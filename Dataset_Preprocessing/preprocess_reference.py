"""
Unified Reference Map Preprocessing Driver.

Runs semantic segmentation and/or SuperPoint feature extraction on the
reference TMS tileset.  Designed to be the single entrypoint for all
offline reference-map preprocessing.

Usage:
    # Run both
    python preprocess_reference.py --all

    # Run only SuperPoint
    python preprocess_reference.py --superpoint

    # Run only semantic
    python preprocess_reference.py --semantic

    # Force re-extraction (overwrite existing)
    python preprocess_reference.py --all --force
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# Add workspace root to path for imports
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from Dataset_Preprocessing import config as cfg
from Dataset_Preprocessing.feature_store import validate_feature_store

logger = logging.getLogger(__name__)


def run_semantic(args):
    """Run semantic preprocessing."""
    from Dataset_Preprocessing.semantic_preprocessor import run_semantic_preprocessing

    print("\n" + "=" * 70)
    print("  SEMANTIC REFERENCE PREPROCESSING")
    print("=" * 70)
    print(f"  Aerial dir:    {args.aerial_dir}")
    print(f"  Prediction dir:{args.prediction_dir}")
    print(f"  Model:         {args.model_path}")
    print(f"  Device:        {args.device}")
    print(f"  Force:         {args.force}")
    print()

    result = run_semantic_preprocessing(
        aerial_dir=args.aerial_dir,
        prediction_dir=args.prediction_dir,
        model_path=args.model_path,
        device=args.device,
        zoom=args.zoom,
        force=args.force,
    )

    print(f"\n  Semantic preprocessing complete:")
    print(f"    Tiles processed: {result['tiles_processed']}")
    print(f"    Tiles skipped:   {result['tiles_skipped']}")
    print(f"    Tiles failed:    {result['tiles_failed']}")
    print(f"    Total:           {result['total_tiles']}")
    print(f"    Time:            {result['elapsed_seconds']:.1f}s")

    if result.get("class_pixel_counts"):
        print(f"\n    Class distribution:")
        total_px = sum(result["class_pixel_counts"].values())
        for cid, count in sorted(result["class_pixel_counts"].items()):
            pct = count / total_px * 100 if total_px > 0 else 0
            name = cfg.SEMANTIC_CLASSES.get(cid, f"class_{cid}")
            print(f"      {name:<15s}: {pct:5.1f}%")

    return result


def run_superpoint(args):
    """Run SuperPoint preprocessing."""
    from Dataset_Preprocessing.superpoint_preprocessor import run_superpoint_preprocessing

    print("\n" + "=" * 70)
    print("  SUPERPOINT REFERENCE PREPROCESSING")
    print("=" * 70)
    print(f"  Aerial dir:    {args.aerial_dir}")
    print(f"  Output HDF5:   {args.features_path}")
    print(f"  Max keypoints: {args.max_keypoints}")
    print(f"  Desc dtype:    {args.dtype}")
    print(f"  Device:        {args.device}")
    print(f"  Force:         {args.force}")
    print()

    result = run_superpoint_preprocessing(
        aerial_dir=args.aerial_dir,
        output_h5_path=args.features_path,
        max_keypoints=args.max_keypoints,
        device=args.device,
        zoom=args.zoom,
        descriptor_dtype=args.dtype,
        force=args.force,
    )

    print(f"\n  SuperPoint preprocessing complete:")
    print(f"    Tiles processed: {result['tiles_processed']}")
    print(f"    Tiles skipped:   {result['tiles_skipped']}")
    print(f"    Tiles failed:    {result['tiles_failed']}")
    print(f"    Total:           {result['total_tiles']}")
    print(f"    Time:            {result['elapsed_seconds']:.1f}s")

    # Validate
    print("\n  Validating feature store...")
    val = validate_feature_store(args.features_path)
    if val["valid"]:
        print(f"    VALID — {val['num_tiles']} tiles")
        for s in val["sample_stats"]:
            print(f"      Tile {s['key']}: {s['num_keypoints']} kpts, "
                  f"desc={s['descriptor_dtype']} {s.get('descriptor_dim', '?')}D, "
                  f"img={s.get('image_size', '?')}")
    else:
        print(f"    INVALID — errors: {val['errors']}")

    # Print file size
    if args.features_path.exists():
        size_mb = args.features_path.stat().st_size / 1024 / 1024
        print(f"\n    Feature store size: {size_mb:.1f} MB")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Unified reference map preprocessing for Pipeline 3")

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true",
                      help="Run both semantic and SuperPoint preprocessing")
    mode.add_argument("--semantic", action="store_true",
                      help="Run only semantic segmentation preprocessing")
    mode.add_argument("--superpoint", action="store_true",
                      help="Run only SuperPoint feature extraction")

    # Paths
    parser.add_argument("--aerial-dir", type=Path,
                        default=cfg.REFERENCE_AERIAL_DIR,
                        help="Path to TMS aerial tiles")
    parser.add_argument("--prediction-dir", type=Path,
                        default=cfg.REFERENCE_PREDICTION_DIR,
                        help="Output path for semantic predictions")
    parser.add_argument("--model-path", type=Path,
                        default=cfg.SEMANTIC_MODEL_PATH,
                        help="Path to semantic model checkpoint")
    parser.add_argument("--features-path", type=Path,
                        default=cfg.REFERENCE_FEATURES_PATH,
                        help="Output path for HDF5 feature store")

    # Parameters
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--zoom", type=int, default=cfg.TMS_ZOOM_LEVEL)
    parser.add_argument("--max-keypoints", type=int,
                        default=cfg.MAX_NUM_KEYPOINTS)
    parser.add_argument("--dtype", default=cfg.DESCRIPTOR_DTYPE,
                        choices=["float32", "float16"])
    parser.add_argument("--force", action="store_true",
                        help="Re-process all tiles, overwriting existing")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.time()

    if args.semantic or args.all:
        run_semantic(args)

    if args.superpoint or args.all:
        run_superpoint(args)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  ALL PREPROCESSING COMPLETE — {elapsed:.1f}s total")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
