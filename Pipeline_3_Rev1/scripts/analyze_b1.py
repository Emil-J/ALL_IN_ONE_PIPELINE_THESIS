"""Quick analysis of Phase B1 results."""
import pandas as pd
import numpy as np

df = pd.read_csv('Pipeline_3_Rev1/outputs/phase_b1/summary.csv')
methods = ['projected_center', 'inlier_centroid', 'trimmed_centroid', 'nadir_corrected', 'weighted_centroid']

print('=== Per-frame: EKF vs Best Rotated vs Best Unrotated ===')
print(f"{'Frame':>5} {'EKF':>8} {'Unrot':>8} {'Rot':>8} {'Rot_method':>22} {'RotBeat':>8}")
for _, r in df.iterrows():
    fi = int(r['frame_idx'])
    ekf = r['ekf_error_m']
    unrot = r['unrot_best_error_m'] if pd.notna(r['unrot_best_error_m']) else 9999
    rot = r['rot_best_error_m'] if pd.notna(r['rot_best_error_m']) else 9999
    rm = str(r['rot_best_method']) if pd.notna(r['rot_best_method']) else 'N/A'
    better = 'YES' if rot < ekf else 'no'
    print(f"{fi:5d} {ekf:8.1f} {unrot:8.1f} {rot:8.1f} {rm:>22} {better:>8}")

print()
print('=== Rotated measurement method comparison ===')
for m in methods:
    col = f'rot_{m}_error_m'
    valid = df[col].dropna()
    beats_ekf = int((valid < df.loc[valid.index, 'ekf_error_m']).sum())
    print(f"  {m:25s}: n={len(valid):2d}  mean={valid.mean():6.1f}m  med={valid.median():6.1f}m  beats_EKF={beats_ekf}/{len(valid)}")

print()
print('=== Oracle: best of ANY method+variant per frame ===')
oracle = []
for _, r in df.iterrows():
    errors = []
    for prefix in ['unrot', 'rot']:
        for m in methods:
            col = f'{prefix}_{m}_error_m'
            if col in r.index and pd.notna(r[col]):
                errors.append(r[col])
    oracle.append(min(errors) if errors else 9999)
df['oracle'] = oracle
print(f"  Oracle mean: {np.mean(oracle):.1f}m  median: {np.median(oracle):.1f}m")
print(f"  Oracle beats EKF: {int(sum(np.array(oracle) < df['ekf_error_m'].values))}/10")

print()
print('=== Quality-gated: use rot if CShape>0.3 & inliers>20, else EKF ===')
gated = []
for _, r in df.iterrows():
    # Check if rotated MAGSAC has good quality
    cs = r.get('rot_magsac_cshape', 0) or 0
    inl = r.get('rot_magsac_inliers', 0) or 0
    cs_dlt = r.get('rot_dlt_cshape', 0) or 0
    inl_dlt = r.get('rot_dlt_inliers', 0) or 0
    best_cs = max(cs, cs_dlt)
    best_inl = max(inl, inl_dlt)
    
    if best_cs > 0.3 and best_inl > 20:
        # Use best rotated measurement
        errors = [r[f'rot_{m}_error_m'] for m in methods if pd.notna(r.get(f'rot_{m}_error_m'))]
        gated.append(min(errors) if errors else r['ekf_error_m'])
    else:
        gated.append(r['ekf_error_m'])
df['gated'] = gated
print(f"  Gated mean: {np.mean(gated):.1f}m  median: {np.median(gated):.1f}m")
print(f"  Frames using visual: {sum(1 for g, e in zip(gated, df['ekf_error_m']) if g != e)}/10")

print()
print('=== CShape-gated with trimmed_centroid as primary ===')
tc_gated = []
for _, r in df.iterrows():
    cs = max(r.get('rot_magsac_cshape', 0) or 0, r.get('rot_dlt_cshape', 0) or 0)
    inl = max(r.get('rot_magsac_inliers', 0) or 0, r.get('rot_dlt_inliers', 0) or 0)
    tc_err = r.get('rot_trimmed_centroid_error_m')
    ic_err = r.get('rot_inlier_centroid_error_m')
    
    if cs > 0.3 and inl > 20:
        if pd.notna(tc_err):
            tc_gated.append(tc_err)
        elif pd.notna(ic_err):
            tc_gated.append(ic_err)
        else:
            tc_gated.append(r['ekf_error_m'])
    else:
        tc_gated.append(r['ekf_error_m'])
df['tc_gated'] = tc_gated
print(f"  TC-gated mean: {np.mean(tc_gated):.1f}m  median: {np.median(tc_gated):.1f}m")

print()
print('=== Nadir-corrected analysis ===')
for _, r in df.iterrows():
    fi = int(r['frame_idx'])
    nc_unrot = r.get('unrot_nadir_corrected_error_m')
    nc_rot = r.get('rot_nadir_corrected_error_m')
    ekf = r['ekf_error_m']
    nc_str_u = f"{nc_unrot:.1f}" if pd.notna(nc_unrot) else "N/A"
    nc_str_r = f"{nc_rot:.1f}" if pd.notna(nc_rot) else "N/A"
    print(f"  Frame {fi}: EKF={ekf:.1f}  nadir_unrot={nc_str_u}  nadir_rot={nc_str_r}")
