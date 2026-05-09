"""
IMU Prior Adapter - Loads simulated IMU prior from ModifiedGPS.kml

Instead of a live IMU pipeline (which requires SimConnect to MSFS),
this adapter reads pre-computed GPS coordinates with ~100m offset
from ground truth, simulating what an IMU dead-reckoning output would
look like.

Each KML Placemark maps a frame filename to (lat, lon, alt).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import warnings


class IMUEstimatorStream:
    """
    Streaming prior provider backed by a KML file of offset GPS coordinates.

    Usage:
        estimator = IMUEstimatorStream(kml_path=Path("ModifiedGPS.kml"))
        for frame_path in query_frames:
            prior = estimator.step(frame_path)
            # prior['lat'], prior['lon'], prior['valid']
    """

    def __init__(self,
                 kml_path: Path,
                 verbose: bool = False,
                 # Legacy kwargs accepted but ignored (for notebook compat)
                 imu_pipeline_dir: Optional[Path] = None,
                 imu_log_path: Optional[Path] = None,
                 algorithm: Optional[str] = None):
        self.verbose = verbose
        self.kml_path = Path(kml_path)
        self.current_index = 0

        # Parse KML into DataFrame
        from ALL_Docs_from_all.old_pipeline.dedode_localization_project.src.io_utils import parse_kml_to_dataframe
        self.results_df = parse_kml_to_dataframe(self.kml_path)

        # Build frame-name → row lookup
        self._lookup: Dict[str, Dict] = {}
        for _, row in self.results_df.iterrows():
            self._lookup[row['frame_name']] = {
                'lat': row['latitude'],
                'lon': row['longitude'],
                'alt': row.get('altitude', 0.0),
            }

        if self.verbose:
            print(f"  ✓ Loaded {len(self._lookup)} IMU prior coordinates from {self.kml_path.name}")

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def step(self, frame_path: Path) -> Dict:
        """Return the IMU prior for a single frame."""
        frame_name = Path(frame_path).name
        entry = self._lookup.get(frame_name)

        if entry is None:
            if self.verbose:
                warnings.warn(f"No IMU prior for frame {frame_name}")
            est = self._invalid_estimate()
            est['frame_path'] = str(frame_path)
            est['imu_index'] = self.current_index
            self.current_index += 1
            return est

        estimate = {
            'lat': entry['lat'],
            'lon': entry['lon'],
            'alt': entry['alt'],
            'heading': 0.0,
            'timestamp': None,
            'confidence': 0.8,
            'valid': True,
            'frame_path': str(frame_path),
            'imu_index': self.current_index,
        }
        self.current_index += 1
        return estimate

    def reset(self):
        """Reset streaming index to beginning."""
        self.current_index = 0

    def get_batch_results(self) -> pd.DataFrame:
        """Return the full DataFrame of prior coordinates."""
        return self.results_df.copy()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _invalid_estimate() -> Dict:
        return {
            'lat': np.nan,
            'lon': np.nan,
            'alt': np.nan,
            'heading': np.nan,
            'timestamp': None,
            'confidence': 0.0,
            'valid': False,
            'frame_path': None,
            'imu_index': -1,
        }
