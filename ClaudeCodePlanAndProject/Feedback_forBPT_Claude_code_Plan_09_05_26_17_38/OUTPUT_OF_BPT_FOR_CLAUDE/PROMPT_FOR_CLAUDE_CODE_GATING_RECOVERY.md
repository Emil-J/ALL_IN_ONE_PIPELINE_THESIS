# Prompt for Claude Code

I am giving you an external code review file:

`GATING_RECOVERY_CODE_REVIEW_FOR_CLAUDE.md`

Read it carefully first. Do **not** treat it as automatically correct. Use it as a reviewer report that you must verify against the actual source code and the latest run data.

Main context:

- GP8 simulator-GPS fallback was removed.
- The pipeline is now more scientifically honest for a GPS-denied thesis.
- The latest live run shows a new problem: many visually strong homography measurements are rejected by the EKF innovation gate because the EKF has drifted too far.
- The key symptom is that corrected homography is often closer to ground truth than the online EKF, but the EKF refuses to update due to `innovation_too_large`.
- Frame 126 is the clearest case: strong CShape, many inliers, high semantic confidence, meta-tile verified, corrected homography near GT, but no EKF update is applied.

Your task:

1. Read `GATING_RECOVERY_CODE_REVIEW_FOR_CLAUDE.md`.
2. Verify every major claim against the code and latest outputs.
3. Investigate whether the review is correct, incomplete, or wrong.
4. Evaluate the proposed fixes and your own alternatives.
5. Produce a final staged implementation plan that fixes the actual problem without reintroducing GPS/simulator truth leakage.

Focus area:

- `runtime/run_pipeline.py`
- `src/temporal_searcher.py`
- `src/visual_measurement.py`
- `src/meta_tile_builder.py`
- `src/particle_filter.py`
- `config/config.py`
- latest `results.csv`
- trace JSON files for frames 0000, 0059, 0082, 0126

Important questions to answer:

1. Is the reviewer correct that the current main failure is over-gating/recovery failure rather than visual localisation failure?
2. Is `gate_pass` currently overloaded or inconsistent between `TemporalSearcher` and `run_pipeline.py`?
3. Are the PF innovation gate and EKF innovation gate using inconsistent threshold formulas?
4. Is `vel * 1.0` in `run_pipeline.py` too strict compared with actual frame cadence / `dt`?
5. Are visually strong frames being rejected only because the EKF estimate has drifted too far?
6. Does the current trace output explain this failure clearly enough?
7. Would a simple large-R update actually recover the EKF, or is covariance inflation/relocalization needed?
8. What is the safest recovery design that does not accept random wrong homographies?

Expected plan output:

1. Verified findings from the review.
2. Findings you disagree with, if any.
3. Root-cause diagnosis.
4. Proposed fix plan, staged and safe.
5. Exact files/functions to modify.
6. New columns/diagnostics to add.
7. Recovery logic design.
8. Validation plan using the latest run.
9. Risks and how to detect regressions.
10. A clear recommendation for what to implement first.

Hard requirements:

- Do not reintroduce simulator/GPS correction after initialization.
- Do not start full Phase B homography scoring/cascade changes until the gating/recovery issue is understood.
- Do not refactor folders.
- Keep changes minimal and testable.
- Preserve the ability to compare corrected homography error vs EKF error.
- Make the trace/debug output explain exactly whether a visual measurement was accepted, rejected, or used for relocalization.

After your analysis, ask me for approval before implementing the final plan.

I want a plan that I can approve with something like:

`Accept recovery/gating fix only`

or

`Revise plan before implementation`
