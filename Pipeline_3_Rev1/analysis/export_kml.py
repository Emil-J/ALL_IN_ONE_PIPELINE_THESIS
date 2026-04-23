"""
analysis/export_kml.py — Export a run's trajectory to KML for Google Earth.

Requires: pip install simplekml

Two LineStrings are written:
  Green  — GPS ground truth  (gps_lat / gps_lon / gps_alt_m)
  Red    — EKF estimate       (final_lat / final_lon / altitude_m)

Usage:
    python analysis/export_kml.py --run-dir outputs/runs/live_test
    python analysis/export_kml.py --run-dir outputs/runs/live_test --output my_flight.kml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent
REPO       = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

import pandas as pd

from config import config


def _parse_args():
    p = argparse.ArgumentParser(description="Export run trajectory to KML")
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Run output directory (contains results.csv)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output KML file path (default: <analysis_dir>/<run_id>.kml)")
    return p.parse_args()


def export_kml(run_dir: Path, output_path: Path | None = None) -> Path:
    """
    Create a KML file from a run's results.csv.

    Returns the path to the written KML file.
    """
    try:
        import simplekml
    except ImportError:
        raise ImportError(
            "simplekml not installed. Run: pip install simplekml"
        )

    results = pd.read_csv(run_dir / "results.csv")
    n = len(results)
    if n == 0:
        raise ValueError(f"results.csv in {run_dir} is empty")

    # ── Build coordinate lists ───────────────────────────────────────────────
    # EKF estimate: always present
    ekf_rows = results[["final_lon", "final_lat", "altitude_m"]].dropna()
    ekf_coords = list(zip(
        ekf_rows["final_lon"].astype(float),
        ekf_rows["final_lat"].astype(float),
        ekf_rows["altitude_m"].astype(float),
    ))

    # GT: only available when GPS columns are present (SimConnect mode)
    gt_coords = []
    if ("gps_lat" in results.columns and "gps_lon" in results.columns
            and "gps_alt_m" in results.columns):
        gt_rows = results[["gps_lon", "gps_lat", "gps_alt_m"]].dropna()
        gt_coords = list(zip(
            gt_rows["gps_lon"].astype(float),
            gt_rows["gps_lat"].astype(float),
            gt_rows["gps_alt_m"].astype(float),
        ))

    # ── Build KML ────────────────────────────────────────────────────────────
    kml = simplekml.Kml()
    kml.document.name = run_dir.name

    if ekf_coords:
        ekf_line = kml.newlinestring(name="EKF Estimated")
        ekf_line.coords = ekf_coords
        ekf_line.style.linestyle.color = simplekml.Color.red
        ekf_line.style.linestyle.width = 4
        ekf_line.altitudemode = simplekml.AltitudeMode.absolute
        ekf_line.extrude = 0

    if gt_coords:
        gt_line = kml.newlinestring(name="GPS Ground Truth")
        gt_line.coords = gt_coords
        gt_line.style.linestyle.color = simplekml.Color.green
        gt_line.style.linestyle.width = 4
        gt_line.altitudemode = simplekml.AltitudeMode.absolute
        gt_line.extrude = 0

    # Start/end placemarks for EKF path
    if ekf_coords:
        start = kml.newpoint(name="Start", coords=[ekf_coords[0]])
        start.style.iconstyle.color = simplekml.Color.lime
        start.altitudemode = simplekml.AltitudeMode.absolute

        end = kml.newpoint(name="End", coords=[ekf_coords[-1]])
        end.style.iconstyle.color = simplekml.Color.red
        end.altitudemode = simplekml.AltitudeMode.absolute

    # ── Save ─────────────────────────────────────────────────────────────────
    if output_path is None:
        out_dir = config.ANALYSIS_OUTPUT_DIR / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"trajectory_{ts}.kml"

    kml.save(str(output_path))

    print(f"KML saved: {output_path}")
    print(f"  EKF path:  {len(ekf_coords)} points")
    if gt_coords:
        print(f"  GT path:   {len(gt_coords)} points")
        if ekf_coords and gt_coords:
            print(f"  Start alt  EKF={ekf_coords[0][2]:.1f}m  GT={gt_coords[0][2]:.1f}m")
    else:
        print("  GT path:   not available (no gps_alt_m column — file-mode run)")

    return output_path


def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    if not (run_dir / "results.csv").exists():
        print(f"ERROR: results.csv not found in {run_dir}", file=sys.stderr)
        sys.exit(1)
    export_kml(run_dir, args.output)


if __name__ == "__main__":
    main()
