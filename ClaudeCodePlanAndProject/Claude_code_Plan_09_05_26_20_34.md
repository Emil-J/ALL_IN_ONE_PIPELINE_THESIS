# Claude Code Prompt — Recheck `ver_matches` Fix Only

Use this prompt in Claude Code when the goal is only to verify the existing `ver_matches` fix, not to modify the code.

```text
I am using Claude Code. Please inspect the current `runtime/run_pipeline.py` and recheck the `ver_matches` fix only.

Goal:
Verify whether the previous live-mode crash is actually fixed:

NameError: name 'ver_matches' is not defined

This crash previously happened in `run_simconnect_mode()` inside the relocalization condition:

and ver_matches >= cfg_r.RELOCALIZATION_VERIFICATION_MIN

Important:
Do not edit files.
Do not refactor.
Do not apply fixes.
Do not change imports, logic, thresholds, EKF math, relocalization logic, semantic logic, or file structure.
Only inspect and report.

Please check the following:

1. Helper existence
Confirm that `_safe_int()` and `_extract_meta_quality(result)` exist in `runtime/run_pipeline.py`.

2. Helper behavior
Confirm that `_extract_meta_quality(result)` safely extracts:
- `meta_tile_verified`
- `tiles_tested`
- `verification_matches`

It should read from direct result keys first, then fall back to `result["meta_tile_info"]` if present, and default safely to zero/False if missing.

3. File mode check
Inspect `_process_one_frame()` and confirm that it calls:

meta_verified, tiles_tested, ver_matches = _extract_meta_quality(result)

before relocalization logic and before result-row creation.

Confirm that `_process_one_frame()` uses the extracted local variables in:
- the relocalization condition
- `meta_tile_verified`
- `tiles_tested`
- `verification_matches`

4. Live SimConnect mode check
Inspect `run_simconnect_mode()` and confirm that it also calls:

meta_verified, tiles_tested, ver_matches = _extract_meta_quality(result)

before the relocalization condition.

Confirm that `ver_matches` is defined in the same local scope where this condition is evaluated:

and ver_matches >= cfg_r.RELOCALIZATION_VERIFICATION_MIN

5. Live result-row check
Confirm that live-mode `result_row` uses the local extracted variables:

"tiles_tested": tiles_tested,
"verification_matches": ver_matches,

and does not rely on inconsistent direct calls such as:

result.get("verification_matches", 0)

6. Scope check
Explicitly verify that `ver_matches` is not only defined inside `_process_one_frame()`. It must also be defined inside `run_simconnect_mode()`, because Python function scopes are separate.

7. Import check
Confirm that there is no incorrect import such as:

from unittest import result

8. Static validation
Run syntax/static checks only:

python -m py_compile runtime/run_pipeline.py

If available, also run:

pytest tests -q

Do not make code changes if checks fail. Report the failure exactly.

Final report format:

1. Verdict: fixed / not fixed / uncertain
2. Evidence from exact file/function/line references
3. Whether `ver_matches` can still be undefined in live mode
4. Whether file mode and live mode now use consistent metadata extraction
5. Output of `py_compile`
6. Output of `pytest tests -q`, if run
7. Any remaining risk, but no implementation unless I explicitly approve it
```

## Expected conclusion if the code is correct

Claude should conclude that the `ver_matches` `NameError` is fixed only if:

- `_extract_meta_quality(result)` exists.
- `_process_one_frame()` calls it.
- `run_simconnect_mode()` calls it.
- `ver_matches` is defined before the relocalization condition in live mode.
- live-mode `result_row` logs `verification_matches` using the local `ver_matches` variable.
- `python -m py_compile runtime/run_pipeline.py` passes.

