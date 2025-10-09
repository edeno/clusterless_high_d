# Setup Tasks for clusterless_high_d

## Phase 1: Directory Structure ✅

- [x] Create `configs/` directory
- [x] Create `src/clusterless_high_d/` directory
- [x] Create `scripts/` directory
- [x] Create `notebooks/` directory
- [x] Create `results/` directory (gitignored)

## Phase 2: Configuration Files ✅

- [x] Create `configs/base.yaml` with decoding parameters (window/step, high-D threshold, feature flags)
- [x] Create `configs/plots.yaml` with plotting parameters (dpi, confidence bands, etc.)
- [x] Create `configs/datasets.yaml` with example dataset registry

## Phase 3: Python Package Structure ✅

- [x] Create `src/clusterless_high_d/__init__.py`
- [x] Create `src/clusterless_high_d/cli.py` with CLI entry points (decode, batch)
- [x] Create `src/clusterless_high_d/decode.py` with main decoding logic
- [x] Create `src/clusterless_high_d/spyglass_io.py` with Spyglass query/load functions
- [x] Create `src/clusterless_high_d/plotting.py` with plotting functions
- [x] Create `src/clusterless_high_d/registry.py` to read datasets.yaml

## Phase 4: Project Configuration ✅

- [x] Create `pyproject.toml` with:
  - Project metadata
  - Dependencies (spyglass-neuro from git, non_local_detector, numpy, pandas, matplotlib, pydantic)
  - Dev dependencies (ruff, black, pytest)
  - Build system configuration
  - CLI entry points

## Phase 5: Scripts & Notebooks ✅

- [x] Create `scripts/run_decode.sh` shell helper script
- [x] Create `notebooks/01_quickstart.ipynb` example notebook

## Phase 6: Documentation & Git ✅

- [x] Create/update `README.md` with project overview and quick start
- [x] Update `.gitignore` to exclude results/, .venv/, __pycache__, etc.

## Phase 7: Environment Setup (manual steps documented) ✅

- [x] Document: `uv python pin 3.11`
- [x] Document: `uv venv`
- [x] Document: `source .venv/bin/activate`
- [x] Document: `uv pip install -e ".[dev]"`

## Notes

- Existing files: .gitignore, .python-version, CLAUDE.md
- All Spyglass credentials should be in environment variables or .env.local (gitignored)
- Keep code minimal and config-driven
- Persist resolved config with every run for reproducibility
