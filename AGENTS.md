# Contributor instructions

Read `docs/status/FRAMEWORK_STATUS.md`, `docs/status/BENCHMARK_STATUS.md`,
`README.md`, and the source relevant to the task before editing. `docs/HANDOFF.md`
is navigation only. Inspect `git status --short` and preserve unrelated changes.

## Scope and invariants

- Maintain the core Coordinator + Inspector + Runtime framework.
- Keep graph input independent of benchmarks, gold answers, labels, and evaluators.
- Preserve topology, quantifiers, distinct bindings, contextual sharing, source
  provenance, and complete-result checks. Limits must never imply no solution.
- Every Inspector delegation has fresh context; Coordinator memory is query-local.
- Do not add query-specific routing, prompts, compatibility layers, or experiment
  variants to the main package.
- Keep implementation modular and use existing dependencies where practical.

## Changes and verification

- Back up documents or removed files under the ignored `clean/` directory before
  replacing them, with original relative paths and SHA256 hashes. Do not read
  those backups as current project instructions.
- Add only focused tests for stable interfaces and important regressions.
- Run `python -m pytest -q -p no:cacheprovider`, `ruff check src tests`, and
  `python -m mypy src/masge` after framework changes. Packaging changes also need
  a wheel build and an import/CLI check outside the source tree.
- Update Framework status after changing framework facts. Update Benchmark status
  only when its scope changes. Keep README and HANDOFF aligned with stable entry
  points. Status documents describe current facts, not chronological logs.
- Record only validation actually performed. Local mock tests are not live-model
  acceptance or accuracy measurements.

## Credentials and external calls

- Never print, commit, or copy `.env` or credentials into logs.
- External model calls require explicit authorization in the current task.
- Do not commit user graphs or run artifacts as test data without authorization.
