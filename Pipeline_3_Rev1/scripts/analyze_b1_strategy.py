"""Refine runtime strategy for Phase B1 integration."""
import pandas as pd
import numpy as np

df = pd.read_csv('Pipeline_3_Rev1/outputs/phase_b1/summary.csv')
methods = ['projected_center', 'inlier_centroid', 'trimmed_centroid', 'nadir_corrected', 'weighted_centroid']

print("=== Fixed-method strategy: quality gate + single method ===")
print("Gate: CShape>0.3 & inliers>20 (rotated), else EKF\n")

for m in methods:
    errors = []
    for _, r in df.iterrows():
        cs = max(r.get('rot_magsac_cshape', 0) or 0, r.get('rot_dlt_cshape', 0) or 0)
        inl = max(r.get('rot_magsac_inliers', 0) or 0, r.get('rot_dlt_inliers', 0) or 0)
        
        if cs > 0.3 and inl > 20:
            err = r.get(f'rot_{m}_error_m')
            if pd.notna(err):
                errors.append(err)
            else:
                errors.append(r['ekf_error_m'])  # method invalid, fallback
        else:
            errors.append(r['ekf_error_m'])
    
    print(f"  {m:25s}: mean={np.mean(errors):6.1f}m  med={np.median(errors):6.1f}m  "
          f"beats_EKF={sum(1 for e, k in zip(errors, df['ekf_error_m']) if e < k)}/10")

# Cascade: trimmed > inlier > weighted > projected
print("\n=== Cascade: trimmed -> inlier -> weighted -> projected ===")
cascade = []
for _, r in df.iterrows():
    cs = max(r.get('rot_magsac_cshape', 0) or 0, r.get('rot_dlt_cshape', 0) or 0)
    inl = max(r.get('rot_magsac_inliers', 0) or 0, r.get('rot_dlt_inliers', 0) or 0)
    
    if cs > 0.3 and inl > 20:
        for m in ['trimmed_centroid', 'inlier_centroid', 'weighted_centroid', 'projected_center']:
            err = r.get(f'rot_{m}_error_m')
            if pd.notna(err):
                cascade.append(err)
                break
        else:
            cascade.append(r['ekf_error_m'])
    else:
        cascade.append(r['ekf_error_m'])
print(f"  Cascade mean: {np.mean(cascade):.1f}m  med: {np.median(cascade):.1f}m  "
      f"beats_EKF={sum(1 for e, k in zip(cascade, df['ekf_error_m']) if e < k)}/10")

# Also try: use unrotated nadir if heading near 0 (frame 4 was heading=-8.9)
print("\n=== Hybrid: rot if |heading|>20, else unrot ===")
hybrid = []
for _, r in df.iterrows():
    hdg = abs(r['heading_deg'])
    if hdg > 20:
        prefix = 'rot'
    else:
        prefix = 'unrot'
    
    cs = max(r.get(f'{prefix}_magsac_cshape', 0) or 0, r.get(f'{prefix}_dlt_cshape', 0) or 0)
    inl = max(r.get(f'{prefix}_magsac_inliers', 0) or 0, r.get(f'{prefix}_dlt_inliers', 0) or 0)
    
    if cs > 0.3 and inl > 20:
        for m in ['trimmed_centroid', 'inlier_centroid', 'weighted_centroid', 'projected_center', 'nadir_corrected']:
            err = r.get(f'{prefix}_{m}_error_m')
            if pd.notna(err):
                hybrid.append(err)
                break
        else:
            hybrid.append(r['ekf_error_m'])
    else:
        hybrid.append(r['ekf_error_m'])
print(f"  Hybrid mean: {np.mean(hybrid):.1f}m  med: {np.median(hybrid):.1f}m  "
      f"beats_EKF={sum(1 for e, k in zip(hybrid, df['ekf_error_m']) if e < k)}/10")

# What if we require BOTH rot and unrot and pick best?
print("\n=== Dual-pipeline: try both rot and unrot, pick lower-error (via CShape proxy) ===")
dual_pick = []
for _, r in df.iterrows():
    best_err = r['ekf_error_m']
    for prefix in ['rot', 'unrot']:
        cs = max(r.get(f'{prefix}_magsac_cshape', 0) or 0, r.get(f'{prefix}_dlt_cshape', 0) or 0)
        inl = max(r.get(f'{prefix}_magsac_inliers', 0) or 0, r.get(f'{prefix}_dlt_inliers', 0) or 0)
        
        if cs > 0.4 and inl > 30:  # stricter gate
            for m in ['trimmed_centroid', 'inlier_centroid', 'weighted_centroid']:
                err = r.get(f'{prefix}_{m}_error_m')
                if pd.notna(err):
                    if err < best_err:
                        best_err = err
                    break
    dual_pick.append(best_err)
print(f"  Dual-pick mean: {np.mean(dual_pick):.1f}m  med: {np.median(dual_pick):.1f}m  "
      f"beats_EKF={sum(1 for e, k in zip(dual_pick, df['ekf_error_m']) if e < k)}/10")

# Also try: pick method by higher CShape winner
print("\n=== CShape-weighted blend: visual_weight = CShape if gate passes ===")
blended = []
for _, r in df.iterrows():
    cs = max(r.get('rot_magsac_cshape', 0) or 0, r.get('rot_dlt_cshape', 0) or 0)
    inl = max(r.get('rot_magsac_inliers', 0) or 0, r.get('rot_dlt_inliers', 0) or 0)
    
    if cs > 0.3 and inl > 20:
        # Get trimmed centroid lat/lon
        tc_err = r.get('rot_trimmed_centroid_error_m')
        ic_err = r.get('rot_inlier_centroid_error_m')
        visual_err = tc_err if pd.notna(tc_err) else ic_err if pd.notna(ic_err) else None
        
        if visual_err is not None:
            # Weight = min(CShape, 0.8) — don't go fully visual
            w = min(cs, 0.8)
            # Blended error estimation (approximate)
            blended.append(w * visual_err + (1 - w) * r['ekf_error_m'])
        else:
            blended.append(r['ekf_error_m'])
    else:
        blended.append(r['ekf_error_m'])
print(f"  CShape-blend mean: {np.mean(blended):.1f}m  med: {np.median(blended):.1f}m")

print("\n=== Summary of strategies ===")
strategies = {
    'EKF baseline': list(df['ekf_error_m']),
    'Rot trimmed_centroid (gated)': [],
    'Cascade (gated)': cascade,
    'Hybrid (heading-aware)': hybrid,
    'Dual-pick (strict gate)': dual_pick,
}
# Fill in gated trimmed
for _, r in df.iterrows():
    cs = max(r.get('rot_magsac_cshape', 0) or 0, r.get('rot_dlt_cshape', 0) or 0)
    inl = max(r.get('rot_magsac_inliers', 0) or 0, r.get('rot_dlt_inliers', 0) or 0)
    if cs > 0.3 and inl > 20:
        err = r.get('rot_trimmed_centroid_error_m')
        if pd.notna(err):
            strategies['Rot trimmed_centroid (gated)'].append(err)
        else:
            strategies['Rot trimmed_centroid (gated)'].append(r['ekf_error_m'])
    else:
        strategies['Rot trimmed_centroid (gated)'].append(r['ekf_error_m'])

print(f"\n{'Strategy':>35} {'Mean':>8} {'Median':>8} {'Beat_EKF':>10}")
ekf = list(df['ekf_error_m'])
for name, errs in strategies.items():
    bt = sum(1 for e, k in zip(errs, ekf) if e < k)
    print(f"  {name:>33}: {np.mean(errs):8.1f} {np.median(errs):8.1f}   {bt}/10")
