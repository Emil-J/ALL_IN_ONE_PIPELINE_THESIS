"""Quick diagnostic: when does drone enter the reference map?"""
import sys
sys.path.insert(0, '.')
import pandas as pd
from src.tile_utils import latlon_to_tile, tile_to_latlon, haversine_distance, find_tiles_within_radius

csv = pd.read_csv(r'C:\Users\emilj\Documents\Thesis\All_In_One_Pipeline\Logs_Run_20260321_162024\imu_gps_log_20260321_162024.csv')

ref_xmin, ref_xmax = 34494, 34508
ref_ymin, ref_ymax = 45025, 45042

# Frame 0
tx, ty = latlon_to_tile(csv.iloc[0]['latitude'], csv.iloc[0]['longitude'], 16)
print(f"Frame 0: tile=({tx}, {ty}), in range? X:{ref_xmin<=tx<=ref_xmax} Y:{ref_ymin<=ty<=ref_ymax}")
tiles = find_tiles_within_radius(csv.iloc[0]['latitude'], csv.iloc[0]['longitude'], 350, 16,
                                  x_range=(ref_xmin, ref_xmax), y_range=(ref_ymin, ref_ymax))
print(f"Frame 0 candidates within 350m: {len(tiles)}")

# Find first row where drone is over the reference map
found = False
for i, row in csv.iterrows():
    tx, ty = latlon_to_tile(row['latitude'], row['longitude'], 16)
    if ref_xmin <= tx <= ref_xmax and ref_ymin <= ty <= ref_ymax:
        print(f"\nDrone enters reference map at row {i}")
        print(f"  ts={row['timestamp']:.3f}, lat={row['latitude']:.5f}, lon={row['longitude']:.5f}")
        print(f"  tile=({tx}, {ty})")
        tiles2 = find_tiles_within_radius(row['latitude'], row['longitude'], 350, 16,
                                           x_range=(ref_xmin, ref_xmax), y_range=(ref_ymin, ref_ymax))
        print(f"  candidates within 350m: {len(tiles2)}")
        found = True
        break

if not found:
    print("\nDrone NEVER enters reference map!")
    ref_lat, ref_lon = tile_to_latlon(34501, 45033, 16)
    best_dist = float('inf')
    best_i = 0
    for i, row in csv.iterrows():
        d = haversine_distance(row['latitude'], row['longitude'], ref_lat, ref_lon)
        if d < best_dist:
            best_dist = d
            best_i = i
    print(f"  Closest approach: row {best_i}, dist={best_dist:.0f}m")
    print(f"  lat={csv.iloc[best_i]['latitude']:.5f}, lon={csv.iloc[best_i]['longitude']:.5f}")

# Show tile coverage for a few positions
print("\n--- Tile check for selected IMU rows ---")
for idx in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
    if idx >= len(csv):
        break
    row = csv.iloc[idx]
    tx, ty = latlon_to_tile(row['latitude'], row['longitude'], 16)
    in_x = ref_xmin <= tx <= ref_xmax
    in_y = ref_ymin <= ty <= ref_ymax
    tiles3 = find_tiles_within_radius(row['latitude'], row['longitude'], 350, 16,
                                       x_range=(ref_xmin, ref_xmax), y_range=(ref_ymin, ref_ymax))
    print(f"  Row {idx:4d}: lat={row['latitude']:.4f} lon={row['longitude']:.4f} "
          f"tile=({tx},{ty}) inX={in_x} inY={in_y} candidates={len(tiles3)}")
