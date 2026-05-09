"""
Standalone script to convert KML ground truth file to CSV format.

Usage:
    python convert_kml_to_csv.py <input.kml> [output.csv]
    
Example:
    python convert_kml_to_csv.py coords_fixed_100m_east.kml
    python convert_kml_to_csv.py coords_fixed_100m_east.kml ground_truth.csv
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from io_utils import convert_kml_to_csv


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_kml_to_csv.py <input.kml> [output.csv]")
        print("\nExample:")
        print("  python convert_kml_to_csv.py coords_fixed_100m_east.kml")
        print("  python convert_kml_to_csv.py coords_fixed_100m_east.kml ground_truth.csv")
        sys.exit(1)
    
    kml_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not kml_path.exists():
        print(f"❌ Error: KML file not found: {kml_path}")
        sys.exit(1)
    
    print(f"Converting KML to CSV...")
    print(f"Input: {kml_path.absolute()}")
    
    try:
        output_path = convert_kml_to_csv(kml_path, csv_path)
        print(f"\n✅ Success! CSV saved to:")
        print(f"   {output_path.absolute()}")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
