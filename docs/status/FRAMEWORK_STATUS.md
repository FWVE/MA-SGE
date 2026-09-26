# Framework status

Last verified: 2026-09-26.

Public source repository: [FWVE/MA-SGE](https://github.com/FWVE/MA-SGE),
default branch `main`.

Published release: [v0.1.0](https://github.com/FWVE/MA-SGE/releases/tag/v0.1.0),
source commit `60aa66de3a497f05dbdd02283721f746c854bbc7`. The release includes
a source ZIP, wheel, source distribution, and SHA256 checksums. Its
[release-commit CI run](https://github.com/FWVE/MA-SGE/actions/runs/36217911397)
passed every check. The frozen release packages correspond to that tagged
commit; subsequent publication notes on `main` do not alter those packages.

## Distribution

The distribution is `ma-sge` version `0.1.0`, targeting Python 3.11. It contains
the Coordinator + Inspector + Runtime implementation and the `pattern`,
`shortest_paths`, and `anchored_core` structural interfaces. The installed entry
points are `masge` and `python -m masge`; the public Python exports are `Graph`,
`ExecutionLimits`, `NativeProvider`, `Solver`, and `load_graph`.

The core graph enumeration, workspace propagation, prompts, context assembly,
and Solver were imported unchanged from the current research implementation.
Release integration replaces legacy credential/settings dependencies with
explicit provider environment variables, validates public execution bounds,
adds standalone graph ingestion and CLI commands, and packages the native
transcript replay provider. Benchmark-specific runner and adapter tests are
outside this distribution; runtime regressions remain included.

## Verification

| Check | Actual local result |
| --- | --- |
| Offline pytest suite | 83 passed |
| Ruff on `src` and `tests` | Passed |
| Strict MyPy on `src/masge` | Passed, 18 source files |
| Wheel and source distribution | Built with the setuptools PEP 517 backend |
| Installed wheel | Import, module CLI, executable CLI, and graph validation passed outside the source tree |
| Package contents | No benchmark, experiment, legacy codegen, credentials, or run artifacts included |
| GitHub Actions, clean Ubuntu / Python 3.11 | Tests, Ruff, MyPy, isolated build, and installed-wheel smoke test passed on `3dec7f6` |
| Live model/API validation | Not run during release preparation |

The installed-wheel check used a fresh virtual environment with existing local
dependencies exposed through system site packages; it was not a fresh network
dependency-resolution test. The separate
[GitHub Actions run](https://github.com/FWVE/MA-SGE/actions/runs/36217835010)
installed dependencies in a clean environment and passed every check.
Tests use pytest-managed temporary directories by default. Local Windows sandbox
verification uses an explicit writable `--basetemp` directory because the shared
system pytest directory is not accessible to that sandbox account.

Validation environment: Windows, Python 3.11; OpenAI SDK 2.46.0, Pydantic 2.13.4,
python-dotenv 1.2.2, NetworkX 3.6.1, NumPy 2.4.6, pytest 9.1.1,
pytest-asyncio 1.4.0, Ruff 0.15.22, MyPy 2.3.0, setuptools 83.0.0.

## Current boundaries

- Public JSON input requires dense integer node IDs and unique original edge IDs.
  Graph text, parallel edges, and self-loops are preserved.
- Providers retain the explicit native model/effort configurations described in
  [Usage](../usage.md#provider-configuration). Account access and current service
  availability require a separately authorized live run.
- Structural enumeration can be exponential; time and assignment limits yield
  `unresolved` rather than an incomplete final result.
- Completion is conditional on the accepted Plan and evidence judgments. It does
  not measure semantic accuracy or establish benchmark performance.
- A source hash identifies input bytes. Retain the graph and the complete run
  directory together for later replay; the CLI does not copy the full source graph.
- Native schema adapters use an OpenAI SDK parsing helper; dependency upgrades
  require the transport and replay regression tests.

Scope of excluded research assets is maintained in
[Benchmark status](BENCHMARK_STATUS.md). Method details live in
[Architecture](../architecture.md); operational contracts live in
[Usage](../usage.md). No external API call is required for the local checks.
