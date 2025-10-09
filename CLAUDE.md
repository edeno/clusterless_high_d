# CLAUDE.md

## Overview

`clusterless_high_d` is a lightweight, config-driven repo for **testing and comparing different clusterless decoding designs** on high-contact probe recordings. The current KDE-based approach in `non_local_detector` doesn't scale well to many-contact probes, so this repo provides a framework to:

- **Benchmark** different decoder implementations (KDE, parametric models, GPU-accelerated variants)
- **Compare** performance and scalability across probe configurations
- **Iterate** quickly on new designs with consistent data loading and evaluation

Data is loaded via **Spyglass** and outputs are minimal (posteriors + quick plots). It's designed to be reproducible, batchable, and config-driven for easy experimentation.

## Directory layout

```
clusterless_high_d/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── plots.yaml
│   └── datasets.yaml        # optional registry (dataset_id → Spyglass query)
├── src/
│   └── clusterless_high_d/
│       ├── __init__.py
│       ├── cli.py           # uv run clusterless-high-d ... or python -m ...
│       ├── decode.py        # config → Spyglass IO → non_local_detector → outputs
│       ├── spyglass_io.py   # Spyglass query/load functions (you fill in)
│       ├── plotting.py      # quick posterior/summary plots
│       └── registry.py      # reads configs/datasets.yaml (optional)
├── scripts/
│   └── run_decode.sh
├── notebooks/
│   └── 01_quickstart.ipynb
└── results/                 # gitignored: per-dataset run outputs
```

---

## Prerequisites

- Python ≥ 3.10 (we pin with uv below)
- Access to Spyglass data source (credentials/config handled outside git)
- Git LFS/DVC **not required**; results are excluded from git

---

## Setup with `uv` (fast Python workflows)

> We use **uv** for environment + dependency management. No `pip`/`venv` required.

```bash
# Install uv (if you don't have it)
# macOS/Linux (recommended):
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
```

### 1) Clone and pin Python

```bash
git clone <YOUR-REPO-URL> clusterless_high_d
cd clusterless_high_d

# Pin a python version for reproducibility
uv python pin 3.11
```

### 2) Create & activate an isolated environment

```bash
uv venv
# Activate:
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\\Scripts\\Activate.ps1
```

### 3) Install the project (editable) and dev extras

```bash
uv pip install -e ".[dev]"
```

> `pyproject.toml` defines runtime deps (Spyglass, non_local_detector, numpy/pandas/matplotlib, pydantic) and optional `[project.optional-dependencies].dev` (ruff/black/pytest, etc.). Keep it small.
>
> **Note:** Spyglass is installed directly from the GitHub repo to ensure latest updates:
>
> ```toml
> [project.dependencies]
> spyglass-neuro = { git = "https://github.com/LorenFrankLab/spyglass.git" }
> ```

---

## Configuration (minimal but strict)

- **`configs/base.yaml`** – decoding knobs (window/step, high-D threshold, feature flags).
- **`configs/plots.yaml`** – plot toggles (dpi, whether to draw confidence bands, etc.).
- **`configs/datasets.yaml` (optional)** – dataset registry used by `batch` mode:

  ```yaml
  rat01_day1:
    spyglass_query: {Animal: rat01, Day: 1}
    region: CA1
    n_contacts: 16
  rat02_day3:
    spyglass_query: {Animal: rat02, Day: 3}
    region: CA3
    n_contacts: 12
  ```

**Best practice:** Treat YAML as the single source of truth. Code should validate and **persist the resolved config** with every run for provenance.

---

## Running decodes

### Decode a single dataset

```bash
uv run python -m clusterless_high_d decode \
  --dataset-id rat01_day1 \
  --config configs/base.yaml \
  --plots configs/plots.yaml \
  --outdir results
```

### Batch over the optional registry

```bash
uv run python -m clusterless_high_d batch \
  --config configs/base.yaml \
  --plots configs/plots.yaml \
  --outdir results
```

### Shell helper (optional)

```bash
bash scripts/run_decode.sh rat01_day1
```

---

## Inputs & Spyglass integration

Implement **`src/clusterless_high_d/spyglass_io.py`** to:

- take `dataset_id` → look up `configs/datasets.yaml` (if present) or pass through a direct identifier,
- execute Spyglass queries (no secrets in git),
- return the **features/events/meta** objects required by `non_local_detector`.

> Keep all Spyglass specifics here so decoding code remains backend-agnostic.

---

## Outputs & reproducibility

Each run writes to:

```
results/<dataset_id>/<YYYYmmdd-HHMMSS>/
  ├── posterior.npz               # minimal numeric outputs (e.g., t, posterior)
  ├── config.resolved.yaml        # exact parameters used for this run
  └── plots/
      ├── posterior_over_time.png
      └── summary.png
```

**Best practice:** Never commit `results/` to git. If you need to track figures for a paper later, consider DVC or Git LFS with explicit include rules.

---

## Plotting

`plotting.py` should produce quick, publication-safe figures (no hardcoded styles that break headless). Keep to a small set of defaults, configurable via `configs/plots.yaml`.

---

## Testing (lean)

Add tests when you start iterating:

- **Unit** (fast, no data): shape checks, config validation, registry behaviors.
- **Integration** (toy data only): stub Spyglass loader → pass minimal arrays through `decode` → verify non-NaN posteriors & saved artifacts.

Run:

```bash
uv run pytest -q
```

---

## Code style & hygiene

- **Format/lint:** ruff + black via dev extras.

  ```bash
  uv run ruff check .
  uv run black --check .
  ```

- **Commits:** small, atomic, with clear messages. Reference dataset IDs/configs if relevant.
- **Branches/PRs:** prefer short-lived branches; open PRs early to record config changes.

---

## Secrets & credentials

- Never commit Spyglass credentials or connection strings.
- Read them from environment variables or a **local**, gitignored file (e.g., `.env.local`).
- If you must ship defaults, make them placeholders with clear instructions.

---

## Common tasks (copy/paste)

```bash
# (Once) create & activate env
uv python pin 3.11 && uv venv && source .venv/bin/activate

# Install
uv pip install -e ".[dev]"

# Lint & test
uv run ruff check .
uv run black --check .
uv run pytest -q

# Decode single dataset
uv run python -m clusterless_high_d decode --dataset-id RAT --config configs/base.yaml --plots configs/plots.yaml

# Batch over registry
uv run python -m clusterless_high_d batch --config configs/base.yaml --plots configs/plots.yaml
```

---

## FAQ

**Q: Do I really need `datasets.yaml`?**
A: No. It’s optional. Use it when you batch across many sessions or want a documented map of datasets → queries/metadata.

**Q: How do I add a new decoder design?**
A: Add a new decoder type in `configs/base.yaml` (e.g., `decoder.type: 'gpu_kde'`), then implement the logic in `decode.py`. Branch on `config['decoder']['type']` to select the implementation. Keep each decoder self-contained for easy comparison.

**Q: Where do high-D (>8 contacts) tweaks live?**
A: Put thresholds & behavior in `configs/base.yaml` and branch on `meta['n_contacts']` inside `decode.py`. Persist the resolved config. Different decoder types may have different scaling behavior.

**Q: Can I keep everything in notebooks?**
A: Keep notebooks thin. Call library functions so your CLI and notebooks produce identical outputs. This makes it easy to run systematic comparisons via CLI and inspect results in notebooks.

**Q: How do I compare decoder performance?**
A: Run multiple configs with different `decoder.type` settings on the same dataset. The timestamped output dirs preserve each run. Add comparison scripts/notebooks that load multiple `posterior.npz` files and compute metrics (accuracy, runtime, memory).

---

## Troubleshooting

- **Missing packages:** Verify `uv run python -c "import spyglass"` works after install. If a package isn’t on PyPI, add a VCS URL in `pyproject.toml`.
- **Plots fail on headless servers:** Ensure your matplotlib backend is compatible (e.g., set `MPLBACKEND=Agg` if needed).
- **Spyglass connection errors:** Confirm env vars/DSN and network access. Keep a small synthetic fallback loader for integration tests.

---

## Contributing (internal)

- Prefer issues for config/method changes; attach small diffs of YAML + a screenshot of resulting plots.
- Add/modify tests when behavior changes (especially config parsing or output formats).
- Keep `spyglass_io.py` backward-compatible if possible; otherwise, bump minor version.

---

**End of file.**
