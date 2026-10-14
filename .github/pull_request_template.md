## Description

<!-- Briefly describe what this PR does and why. -->

## Checklist

Please confirm each of the following. If any item is not checked, describe why below.

- [ ] This PR has one clear atomic contribution rather than several logically different changes.
- [ ] New functionality added by this PR is covered by tests.
- [ ] `python -m pytest` passes locally (the CI runs the weight-free `tests/`; `tests_local/` needs model weights and runs only locally).
- [ ] If inference/morphology changed on purpose, the golden baseline was regenerated and reviewed.
- [ ] No runtime artifacts are committed (e.g. `logs/`, `.cache/`, `ui_settings.json`, `app_screenshot.png`).
- [ ] `requirements.txt` / `requirements-dev.txt` are updated if dependencies changed (mind the `numpy>=2.1,<2.3` pin).
- [ ] Code style is consistent with PEP 8 (no dead code / unused imports; new segmenters follow the `BaseSegmenter` seam).

### If any of the above is not checked, explain why

<!-- Leave blank if all boxes are checked. -->
