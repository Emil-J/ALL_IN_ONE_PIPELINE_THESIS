"""Build reference database metadata from TMS directory structure."""
import sys
from pathlib import Path
sys.path.insert(0, 'dedode_localization_project')

from src import tms_utils
import pandas as pd
import numpy as np

# Scan TMS directory
tms_dir = Path(r"C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\REFERENCE_MAP_VEJLE_20260321_162024\aerial")
zoom = 16

tiles = []
for tile_x_dir in sorted((tms_dir / str(zoom)).iterdir()):
    if not tile_x_dir.is_dir():
        continue
    tile_x = int(tile_x_dir.name)
    
    for tile_file in sorted(tile_x_dir.glob("*.png")):
        tile_y = int(tile_file.stem)
        
        # Compute lat/lon from TMS tile coordinates
        lat, lon = tms_utils.tile_to_latlon(tile_x, tile_y, zoom)
        
        tiles.append({
            'tile_x': tile_x,
            'tile_y': tile_y,
            'zoom': zoom,
            'lat': lat,
            'lon': lon,
            'file_path': str(tile_file)
        })

df = pd.DataFrame(tiles)
print(f"Found {len(df)} tiles")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample:")
print(df.head(3))

# Save to CSV
output_path = Path(r"C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\reference_tiles_metadata.csv")
df.to_csv(output_path, index=False)
print(f"\n✓ Saved metadata to {output_path}")
