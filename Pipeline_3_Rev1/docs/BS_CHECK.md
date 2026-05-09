# BS Check: Is the current localisation claim valid?
**Pipeline 3 — All_In_One_Pipeline**
*Brutally honest assessment, source-grounded. Performed: 2026-05-09.*

This document complements `GPS_DENIED_INTEGRITY_AUDIT.md`. The audit lists every site that touches GPS truth; this document asks whether the thesis-level claim survives the audit.

---

## Three regimes — use the right term

The thesis must distinguish three categories. **Do not describe the system as "GPS-free" or "never uses GPS."** It does use GPS exactly once, by design.

| Term | Definition | Applies to |
|---|---|---|
| **GPS-free from startup** | No GPS sample is consumed at any point, including initialisation. The system has no geodetic prior; e.g. pure visual SLAM that constructs its own map. | **Not** this system. |
| **GPS-denied after initial geodetic prior** | A single GPS sample is consumed at $t_0$ to set the local-NED origin (`lat0, lon0, alt0, heading0`). After $t_0$, no further GPS sample is consumed by the estimator. The system localises by integrating IMU and correcting with visual measurements. | **File mode**, after row 0. **Live mode**, but only on frames where `gate_pass=True`. **This is the headline regime for the thesis claim.** |
| **GPS-aided fallback** | A GPS sample is consumed by the estimator under a defined operational condition (e.g. visual gate failure), with bounded measurement noise so it acts as a soft anchor rather than a position fix. | **Live mode** on frames where `gate_pass=False` (GP8). Operational scaffolding, not part of the core localisation method. |

**Recommended thesis wording:**

> "The proposed system performs visual-inertial localisation without recurring GPS updates after a single initial geodetic prior at $t_0$. The prior defines the local NED reference frame. From $t_1$ onward, position is integrated from IMU sensors and corrected exclusively by visual measurements derived from the orthophoto reference map; no further GPS sample is consumed by the estimator."

This wording is true for file-mode evaluations and for live-mode evaluations with GP8 disabled. It is **not** true for live-mode results that include any not-gate-pass frame, unless those frames are excluded from the aggregate.

---

## Question-by-question assessment

### Q1 — What parts of the pipeline are genuinely GPS-denied?
- The Error-State EKF predict path (IMU integration, baro, accel-mag, airspeed). [`src/ekf_ins.py::predict, update_barometer, update_accel_mag, update_airspeed`]
- The visual measurement chain (rotation → SuperPoint+LightGlue → meta-tile → homography → look-ahead correction → adaptive R). [`src/{visual_measurement, geometric_matcher, meta_tile_builder, position_estimator}.py`]
- The particle filter and tile search (search centre is the EKF estimate, not GPS). [`src/{particle_filter, best_first_search, temporal_searcher}.py`; GP5, GP6, GP11–GP13]
- `update_position` itself — the function does not know whether its input is sim-truth or visual-truth; that's the caller's responsibility. [`src/ekf_ins.py:398-446`]

### Q2 — What parts rely on an initial GPS/geodetic prior?
EKF origin (`lat0, lon0, alt0, heading0`) is initialised from the first valid sample. This is a one-time prior used to define the local NED frame. **Standard and defensible** — without it, no system that reports lat/lon can begin.

- Live mode: `runtime/run_pipeline.py:625-639` (GP1).
- File mode: `runtime/run_pipeline.py:288-305` (GP2).

### Q3 — What parts may use simulator/GPS truth during runtime?
**One specific site: `runtime/run_pipeline.py:752-763` (GP8)**, the live-mode not-gate-pass fallback. It is the only non-prior, non-logging GPS path that reaches `ekf.update_position`.

### Q4 — Which results are valid as GPS-denied results?
- **File-mode results.** Anything produced by `--source file` is a pure GPS-denied evaluation after row-0 init. (Note: file mode is currently broken-as-configured on the active map — see `GPS_DENIED_INTEGRITY_AUDIT.md` § "Notes on file-mode replay availability".)
- **Live-mode results restricted to gate-pass frames.** The 120/125 gate-pass frames in `live_020_Odense_f1` form a clean GPS-denied subset. The remaining 5 not-gate-pass frames are GP8-anchored and must be reported separately or excluded.

### Q5 — Which results may be artificially improved by the fallback correction?
All live-mode aggregate metrics that include any not-gate-pass frame. On `live_020_Odense_f1`, that's 5/125 frames (≈ 4 %). The R = 200 m is generous, so the per-frame bias is small, but it is non-zero and systematic toward truth.

The cleaner replacement experiment is one of:
- (a) Disable GP8 and re-run.
- (b) Report with-fallback and without-fallback aggregates separately.
- (c) Record a file-mode log over Odense and replay it without GP8.

### Q6 — Can I honestly describe the pipeline as GPS-denied after initialization?
- For **file mode**: yes, unambiguously — but use the precise framing from the table above ("localisation without recurring GPS updates after initialisation"), not "never uses GPS".
- For **live mode**: only if GP8 is disabled, *or* it is reported as an auxiliary stabilisation path (not part of the localisation method) and the evaluation excludes its frames.
- **As shipped today, the unmodified live mode is "GPS-aided fallback under failure", not "GPS-denied".**

### Q7 — Is the system GPS-denied only when `gate_pass` is True?
In live mode, **yes**. `runtime/run_pipeline.py:747-763` shows the dichotomy:
- `if gate_pass: ekf.update_position(homo, R = adaptive)` (visual update — GP7, Class B)
- `if not gate_pass: ekf.update_position(sim, R = 200²)` (sim-anchor — GP8, Class C/D)

There is no third branch where the live-mode EKF coasts on IMU alone. (The file-mode branch in `_process_one_frame:374-386` does coast on IMU alone — only the gate-pass visual update is performed.)

### Q8 — Does the fallback make reported live-mode error optimistic?
Yes, on average. R = 200 m is generous, so the bias per fallback frame is small. But the floor is non-zero: across many fallback frames, the EKF cannot drift far from truth, which artificially supports continuity of the trajectory. On `live_020_Odense_f1` (4 % fallback rate) the impact is small; on noisier flights with more visual failures it would be larger.

### Q9 — Should results be reported separately with fallback enabled and disabled?
**Yes.** Two configurations:
- (i) "Method-only" run — file mode, or live mode with GP8 disabled. **This is the headline result.**
- (ii) "Operational" run — live mode as shipped. **This is the safety-net behaviour, useful as a demo but not as the headline thesis result.**

Both should appear in the thesis with clear labels.

### Q10 — Should the fallback be described as a simulator-stabilisation mechanism rather than part of the core method?
**Yes.** The thesis architecture diagram should show GP8 as a dashed/auxiliary edge labelled "live-mode safety anchor (sim GPS, R = 200 m, when visual gate fails)" — not as part of the main fusion path. The main fusion path is: IMU → visual measurement → adaptive R → EKF. The fallback is operational scaffolding.

This is reflected in `docs/Diagrams/03_live_runtime_pipeline.mmd` and `docs/Diagrams/05_ekf_visual_fusion.mmd`.

### Q11 — Should architecture diagrams show the fallback as auxiliary?
**Yes.** Implemented in the Mermaid diagrams shipped alongside this document — GP8 is rendered with `linkStyle … stroke-dasharray:5 5` and an explicit auxiliary label.

### Q12 — What must be changed or reported separately before the thesis claim is defensible?

**Minimum (no code changes):**
1. Document GP8 explicitly in the thesis (cannot quietly reuse current live numbers).
2. Provide a file-mode evaluation on a flight matching the active reference map. Replay produces the clean number.
3. Either disable GP8 for thesis evaluation runs, or partition reported metrics by `gate_pass` and exclude fallback frames from the GPS-denied aggregate.
4. Update `docs/pipeline_breakdown.tex` to mark the live-mode fallback (done in this Phase 1 deliverable).

**Optional cleaner alternatives (require small code changes — out of Phase 1 scope):**
- Add a CLI flag `--disable-sim-anchor` to GP8 so the live-mode evaluation run can opt out without editing source.
- Or replace GP8 with a "freeze EKF and grow σ" branch when visual fails for N consecutive frames. This keeps the search region bounded by the EKF's own σ instead of by GPS.

---

## Bottom line

| Statement | Verdict |
|---|---|
| The algorithm is GPS-denied after a single initialisation prior. | **True.** |
| The current live runtime *with GP8 active* is purely GPS-denied. | **False.** It is GPS-aided fallback under visual failure. |
| The current live runtime *restricted to gate-pass frames* is GPS-denied. | **True.** |
| File-mode results (when the recorded log matches the active map) are scientifically clean GPS-denied results. | **True.** |
| The headline thesis claim is defensible if reported with the framing in the table at the top of this document, plus a clear disclosure of GP8 in live mode. | **True.** |
| The headline thesis claim is defensible if live-mode aggregate numbers (e.g. mean error across all frames) are reported without disclosing GP8. | **False.** |

**The fix is small and entirely on the documentation/evaluation side. No code change is required to make the claim defensible.** What is required is honest framing and either (a) a file-mode evaluation, or (b) a live-mode evaluation with GP8-frames excluded.
