import json

with open("Pipeline_3_Rev1/outputs/logs/pipeline3_run_199.325.jsonl") as f:
    lines = f.readlines()

# Check dt values
dt_zero = sum(1 for l in lines if json.loads(l)["dt"] == 0.0)
print(f"dt=0.0 frames: {dt_zero}/{len(lines)}")

# Show timestamps
print("\nFirst 5 frames timestamps and dt:")
for i in range(5):
    d = json.loads(lines[i])
    print(f"  Frame {i}: ts={d['timestamp']:.6f} dt={d['dt']}")

# Particle spread over time
print("\nParticle spread and estimated lat over time:")
for i in range(0, min(len(lines), 200), 20):
    d = json.loads(lines[i])
    pstd = d.get("particle_position_std_m")
    mode = d["mode"]
    est_lat = d.get("estimated_lat")
    imu_lat = d.get("imu_lat", 0)
    top1 = d.get("top1_tile")
    print(f"  Frame {i}: pstd={pstd} mode={mode:20s} top1={top1} est_lat={est_lat} imu_lat={imu_lat:.6f}")

# Unique tiles matched over the run
print("\nAll unique top1 tiles and their frequency:")
tile_counts = {}
for l in lines:
    d = json.loads(l)
    t = d.get("top1_tile")
    if t:
        key = tuple(t)
        tile_counts[key] = tile_counts.get(key, 0) + 1
for tile, cnt in sorted(tile_counts.items(), key=lambda x: -x[1]):
    print(f"  Tile {tile}: {cnt} times")

# GPS ground truth range (from the notebook CSV)
print(f"\nIMU lat range: {min(json.loads(l)['imu_lat'] for l in lines):.6f} to {max(json.loads(l)['imu_lat'] for l in lines):.6f}")
print(f"IMU lon range: {min(json.loads(l)['imu_lon'] for l in lines):.6f} to {max(json.loads(l)['imu_lon'] for l in lines):.6f}")
