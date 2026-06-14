# Contributing

Thanks for contributing to Cells Calculator. This document describes what
reviewers will expect when you open a pull request, so you can plan your work
accordingly. See [DEVELOPMENT.md](DEVELOPMENT.md) for setup and the exact
commands referenced below.

## Before opening a PR

The PR template asks you to confirm the following. Each item exists because CI
does not catch it for you.

### One atomic contribution per PR

Each PR should contain one clear, logically self-contained change. If you find
yourself bundling unrelated changes (e.g. a bug fix plus an unrelated
refactor), split them into separate PRs. Smaller, focused PRs are easier to
review, easier to revert, and easier to bisect later.

If you have a good reason to combine changes, describe it in the PR
description.

### New functionality is covered by tests

New behavior should come with tests. For most code, that means a unit test
under `tests/`. For changes that are hard to unit-test in isolation — model
integration, image pipeline output, end-to-end UI workflows — coverage can
take the form of a smoke test or an image golden regression entry under
`tests_local/`. See [DEVELOPMENT.md](DEVELOPMENT.md) for both suites.

If a change genuinely cannot be tested (e.g. a pure refactor that existing
tests already cover, or a docs-only change), leave the box unchecked and say
so in the PR description.

### `pytest -v tests_local` passes locally

CI only runs the `tests/` suite. The `tests_local/` suite — which covers smoke
tests, fuzzing, and image golden regressions — is not run on PRs because it
requires models that are not part of the repo and can take a long time.

Because CI will not catch regressions here, you are expected to run it
locally before opening a PR. See [DEVELOPMENT.md](DEVELOPMENT.md) for how to
run focused subsets while iterating.

### No runtime artifacts committed

The application writes a number of files at runtime that should not be
committed:

- `app.log`
- `logs/`
- `*_output/` directories (e.g. `cellprocesser_output/`, `tracker_output/`)
- `application_settings.backup.*.json`

Please check `git status` before committing.

### `requirements.txt` / `requirements-dev.txt` updated

If your change adds or removes a Python dependency, update the appropriate
requirements file. CI runs `pip check` but will not catch a dependency that
your code imports but never lists.

### Code style is consistent with PEP 8

New and modified code should follow [PEP 8](https://peps.python.org/pep-0008/).
Try to match the surrounding style when editing existing files.

## When something does not apply

If a checklist item does not apply to your PR (for example, you intentionally
combined two changes), leave the box unchecked and explain why in the PR
description. Reviewers would rather see a short explanation than a checked
box that does not match reality.
