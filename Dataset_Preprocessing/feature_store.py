"""
HDF5 Feature Store for precomputed SuperPoint reference tile features.

Design decisions:
  - HDF5 chosen over per-tile NPZ files: single file, fast indexed lookup by
    group name, metadata support, no directory explosion on Windows.
  - HDF5 chosen over memory-mapped arrays: keypoint count varies per tile
    (no fixed row length), so padding would waste space.
  - Each tile stored as a group:  tiles/{x}_{y}/  containing datasets for
    keypoints, descriptors, scores, and scalar attributes.
  - A top-level index group stores arrays of tile_x, tile_y, num_keypoints
    for quick enumeration without opening every tile group.
  - Descriptors stored as float32 by default (SuperPoint native). float16
    option available but must be validated externally.

Schema:
    reference_features.h5
    ├── metadata/
    │   ├── extractor_name   (str)
    │   ├── max_keypoints    (int)
    │   ├── tile_size_px     (int)
    │   ├── zoom             (int)
    │   ├── creation_time    (str)
    │   ├── descriptor_dtype (str)
    │   └── num_tiles        (int)
    ├── tiles/
    │   ├── {x}_{y}/
    │   │   ├── keypoints    (N, 2) float32
    │   │   ├── descriptors  (N, 256) float32 | float16
    │   │   ├── scores       (N,) float32
    │   │   ├── image_height (int attr)
    │   │   ├── image_width  (int attr)
    │   │   └── num_keypoints (int attr)
    │   └── ...
    └── index/
        ├── tile_x          (M,) int32
        ├── tile_y          (M,) int32
        └── num_keypoints   (M,) int32
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Writer — used by the preprocessing pipeline
# ═══════════════════════════════════════════════════════════════════

class FeatureStoreWriter:
    """Create or append to an HDF5 feature store."""

    def __init__(self, h5_path: Path, *,
                 extractor_name: str = "superpoint",
                 max_keypoints: int = 2048,
                 tile_size_px: int = 512,
                 zoom: int = 16,
                 descriptor_dtype: str = "float32"):
        self.h5_path = Path(h5_path)
        self.extractor_name = extractor_name
        self.max_keypoints = max_keypoints
        self.tile_size_px = tile_size_px
        self.zoom = zoom
        self.descriptor_dtype = descriptor_dtype
        self._tile_keys: List[str] = []
        self._fh: Optional[h5py.File] = None

    def open(self):
        """Open (or create) the HDF5 file for writing."""
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = h5py.File(str(self.h5_path), "a")
        # Load existing tile keys if resuming
        if "tiles" in self._fh:
            self._tile_keys = list(self._fh["tiles"].keys())
        return self

    def close(self):
        """Write index + metadata, then close."""
        if self._fh is None:
            return
        self._write_index()
        self._write_metadata()
        self._fh.close()
        self._fh = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def has_tile(self, tile_x: int, tile_y: int) -> bool:
        key = f"{tile_x}_{tile_y}"
        return key in self._tile_keys

    def add_tile(self, tile_x: int, tile_y: int,
                 keypoints: np.ndarray,
                 descriptors: np.ndarray,
                 scores: np.ndarray,
                 image_height: int,
                 image_width: int):
        """
        Store features for one tile.

        Args:
            tile_x, tile_y: TMS tile coordinates.
            keypoints:   (N, 2) float32 — pixel coordinates in tile image.
            descriptors: (N, D) float32 — SuperPoint descriptors.
            scores:      (N,) float32  — keypoint confidence scores.
            image_height, image_width: tile image dimensions.
        """
        key = f"{tile_x}_{tile_y}"
        grp_path = f"tiles/{key}"

        # Remove existing group if overwriting
        if grp_path in self._fh:
            del self._fh[grp_path]

        grp = self._fh.create_group(grp_path)

        # Cast descriptors to target dtype
        desc_np = descriptors.astype(self.descriptor_dtype)

        grp.create_dataset("keypoints", data=keypoints.astype(np.float32),
                           compression="gzip", compression_opts=1)
        grp.create_dataset("descriptors", data=desc_np,
                           compression="gzip", compression_opts=1)
        grp.create_dataset("scores", data=scores.astype(np.float32),
                           compression="gzip", compression_opts=1)
        grp.attrs["image_height"] = image_height
        grp.attrs["image_width"] = image_width
        grp.attrs["num_keypoints"] = len(keypoints)

        if key not in self._tile_keys:
            self._tile_keys.append(key)

    def _write_metadata(self):
        """Write/overwrite metadata group."""
        if "metadata" in self._fh:
            del self._fh["metadata"]
        meta = self._fh.create_group("metadata")
        meta.attrs["extractor_name"] = self.extractor_name
        meta.attrs["max_keypoints"] = self.max_keypoints
        meta.attrs["tile_size_px"] = self.tile_size_px
        meta.attrs["zoom"] = self.zoom
        meta.attrs["descriptor_dtype"] = self.descriptor_dtype
        meta.attrs["num_tiles"] = len(self._tile_keys)
        meta.attrs["creation_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    def _write_index(self):
        """Write/overwrite the quick-lookup index arrays."""
        if "index" in self._fh:
            del self._fh["index"]
        idx = self._fh.create_group("index")

        xs, ys, nkps = [], [], []
        for key in self._tile_keys:
            parts = key.split("_")
            xs.append(int(parts[0]))
            ys.append(int(parts[1]))
            grp = self._fh[f"tiles/{key}"]
            nkps.append(int(grp.attrs["num_keypoints"]))

        idx.create_dataset("tile_x", data=np.array(xs, dtype=np.int32))
        idx.create_dataset("tile_y", data=np.array(ys, dtype=np.int32))
        idx.create_dataset("num_keypoints", data=np.array(nkps, dtype=np.int32))


# ═══════════════════════════════════════════════════════════════════
#  Reader — used by the online pipeline at runtime
# ═══════════════════════════════════════════════════════════════════

class FeatureStoreLoader:
    """
    Read-only loader for precomputed SuperPoint reference tile features.

    Opens the HDF5 file once and provides fast tile-level lookups.
    Returned dicts are compatible with LightGlue input format (PyTorch
    tensors on the requested device).

    Usage:
        store = FeatureStoreLoader("reference_features.h5", device="cuda")
        store.open()
        feats = store.get_features(34500, 45030)
        # feats is a dict with keypoints, descriptors, scores tensors
        store.close()
    """

    def __init__(self, h5_path, device: str = "cuda"):
        self.h5_path = Path(h5_path)
        self.device = device
        self._fh: Optional[h5py.File] = None
        self._tile_set: Optional[set] = None
        self._metadata: Optional[Dict] = None

    def open(self):
        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"Feature store not found: {self.h5_path}\n"
                "Run Dataset_Preprocessing/preprocess_reference.py --superpoint first.")
        self._fh = h5py.File(str(self.h5_path), "r")
        # Build fast lookup set from index
        if "index" in self._fh:
            xs = self._fh["index/tile_x"][:]
            ys = self._fh["index/tile_y"][:]
            self._tile_set = set(zip(xs.tolist(), ys.tolist()))
        else:
            # Fallback: scan tile groups
            self._tile_set = set()
            if "tiles" in self._fh:
                for key in self._fh["tiles"]:
                    parts = key.split("_")
                    self._tile_set.add((int(parts[0]), int(parts[1])))
        # Cache metadata
        if "metadata" in self._fh:
            m = self._fh["metadata"]
            self._metadata = {k: m.attrs[k] for k in m.attrs}
        logger.info("Feature store opened: %s (%d tiles)",
                    self.h5_path.name, len(self._tile_set))
        return self

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._tile_set = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    @property
    def num_tiles(self) -> int:
        return len(self._tile_set) if self._tile_set else 0

    @property
    def metadata(self) -> Optional[Dict]:
        return self._metadata

    def has_tile(self, tile_x: int, tile_y: int) -> bool:
        """Check if tile features exist in the store."""
        if self._tile_set is None:
            return False
        return (tile_x, tile_y) in self._tile_set

    def get_features(self, tile_x: int, tile_y: int) -> Optional[Dict]:
        """
        Load precomputed features for a tile, formatted for LightGlue.

        Returns a dict matching the output of SuperPoint.extract():
            {
                'keypoints':   (1, N, 2) float32 tensor on device,
                'descriptors': (1, N, D) float32 tensor on device,
                'keypoint_scores': (1, N) float32 tensor on device,
                'image_size':  (1, 2) tensor [H, W],
            }

        Returns None if tile not in store.
        """
        if self._fh is None:
            raise RuntimeError("Feature store not opened. Call .open() first.")

        key = f"{tile_x}_{tile_y}"
        grp_path = f"tiles/{key}"
        if grp_path not in self._fh:
            return None

        grp = self._fh[grp_path]

        kpts = torch.from_numpy(grp["keypoints"][:]).unsqueeze(0).to(self.device)
        descs = torch.from_numpy(
            grp["descriptors"][:].astype(np.float32)
        ).unsqueeze(0).to(self.device)
        scores = torch.from_numpy(grp["scores"][:]).unsqueeze(0).to(self.device)

        h = int(grp.attrs["image_height"])
        w = int(grp.attrs["image_width"])
        img_size = torch.tensor([[h, w]], device=self.device)

        return {
            "keypoints": kpts,
            "descriptors": descs,
            "keypoint_scores": scores,
            "image_size": img_size,
        }

    def get_all_tile_coords(self) -> List[Tuple[int, int]]:
        """Return list of all (tile_x, tile_y) in the store."""
        if self._tile_set is None:
            return []
        return list(self._tile_set)


# ═══════════════════════════════════════════════════════════════════
#  Validation utilities
# ═══════════════════════════════════════════════════════════════════

def validate_feature_store(h5_path: Path, num_samples: int = 5) -> Dict:
    """
    Run basic integrity checks on a feature store.

    Returns dict with 'valid' (bool), 'num_tiles', 'errors' (list of str),
    'sample_stats' (list of dicts).
    """
    result = {"valid": True, "num_tiles": 0, "errors": [], "sample_stats": []}

    if not Path(h5_path).exists():
        result["valid"] = False
        result["errors"].append(f"File not found: {h5_path}")
        return result

    try:
        with h5py.File(str(h5_path), "r") as f:
            # Check metadata
            if "metadata" not in f:
                result["errors"].append("Missing metadata group")
                result["valid"] = False

            # Check index
            if "index" not in f:
                result["errors"].append("Missing index group")
            else:
                n = len(f["index/tile_x"])
                result["num_tiles"] = n

            # Check tiles
            if "tiles" not in f:
                result["errors"].append("Missing tiles group")
                result["valid"] = False
                return result

            tile_keys = list(f["tiles"].keys())
            if len(tile_keys) == 0:
                result["errors"].append("No tiles stored")
                result["valid"] = False
                return result

            result["num_tiles"] = len(tile_keys)

            # Sample random tiles
            import random
            samples = random.sample(tile_keys, min(num_samples, len(tile_keys)))
            for key in samples:
                grp = f[f"tiles/{key}"]
                stats = {"key": key}
                try:
                    kpts = grp["keypoints"]
                    descs = grp["descriptors"]
                    scores = grp["scores"]
                    stats["num_keypoints"] = kpts.shape[0]
                    stats["descriptor_dim"] = descs.shape[1] if descs.ndim == 2 else 0
                    stats["descriptor_dtype"] = str(descs.dtype)
                    stats["image_size"] = (int(grp.attrs["image_height"]),
                                           int(grp.attrs["image_width"]))

                    # Sanity checks
                    if kpts.shape[0] == 0:
                        result["errors"].append(f"Tile {key}: 0 keypoints")
                    if kpts.shape[1] != 2:
                        result["errors"].append(f"Tile {key}: keypoints shape {kpts.shape}")
                        result["valid"] = False
                    if descs.shape[0] != kpts.shape[0]:
                        result["errors"].append(
                            f"Tile {key}: desc rows {descs.shape[0]} != kpts {kpts.shape[0]}")
                        result["valid"] = False
                except Exception as e:
                    stats["error"] = str(e)
                    result["errors"].append(f"Tile {key}: {e}")
                    result["valid"] = False
                result["sample_stats"].append(stats)

    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to open HDF5: {e}")

    return result
