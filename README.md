![pyChanneLab banner](banner.png)

# pyChanneLab

**Ion Channel Markov State Model fitting**

pyChanneLab fits kinetic (Markov State) models to voltage-clamp electrophysiology data.
It provides:

- A **Streamlit GUI** for model definition, data upload, and interactive fitting.
- A **PyTorch pipeline** (Differential Evolution → Adam → L-BFGS) that runs on CPU, MPS (Apple Silicon), or CUDA.
- A **standalone script exporter** that downloads a self-contained `.py` file for running on an HPC cluster.

---

## Table of contents

1. [Environment setup — uv (recommended)](#1-environment-setup--uv-recommended)
2. [Environment setup — conda / mamba](#2-environment-setup--conda--mamba)
3. [Installation](#3-installation)
4. [Running the GUI](#4-running-the-gui)
5. [Running the optimisation script on HPC](#5-running-the-optimisation-script-on-hpc)
6. [Running the tests](#6-running-the-tests)
7. [Building the documentation](#7-building-the-documentation)

---

## 1. Environment setup — uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager.

```bash
# Install uv (one-time, system-wide)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/lianctrl/pyChanneLab.git
cd pyChanneLab

# Create a virtual environment and install all dependencies
uv sync

```

`uv sync` reads `pyproject.toml`, creates `.venv/`, and installs everything
including the package itself in editable mode.  No separate `pip install -e .`
is needed.

### GPU / CUDA (optional)

By default, PyTorch installs as a CPU-only build.
For CUDA acceleration:

```bash
# Example: CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. Environment setup — conda / mamba

```bash
# Create and activate a conda environment
conda create -n pychannelab python=3.12 
conda activate pychannelab

# Install PyTorch (choose the right build for your hardware)
# CPU only:
pip install torch

# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install the package in editable mode with all dependencies
pip install -e "."

```

---

## 3. Installation

The package installs itself as `pychannelab` with the `pychannelab` CLI command.

| Method | Command |
|--------|---------|
| uv (automatic via `uv sync`) | — |
| pip editable | `pip install -e .` |

After installation `pychannelab` is available in your PATH.

---

## 4. Running the GUI

```bash
# With uv (no prior installation step needed)
uv run pychannelab

# With any activated environment (conda, venv, …)
pychannelab

# Without installation — run directly
cd src/pychannel_lab
streamlit run app.py
```

The browser opens at `http://localhost:8501`.

### GUI tabs

| Tab | Purpose |
|-----|---------|
| 🏗️ MSM Builder | Define states, transitions, parameters; load presets or import JSON. |
| 🧪 Protocols | Set timing and voltage levels for each of the four voltage-clamp protocols. |
| 📁 Data | Upload one CSV per protocol (`x`, `y`, optional `y_err` columns). |
| 🔍 Preview | Sanity-check the initial parameter guess against the loaded data. |
| 🚀 Optimise | Configure DE → Adam → L-BFGS and run the fitting pipeline. Export the run script. |
| 📊 Results | Fitted parameters, AIC/BIC, comparison figures, JSON downloads. |

### Data CSV format

```
x,y,y_err
-90,0.01,0.005
-70,0.05,0.008
...
```

- `x` — independent variable of specific protocols (voltage, time, ...)
- `y` — experimental observable
- `y_err` — experimental error (optional)

---

## 5. Running the optimisation script on HPC

### Step 1 — Export the script from the GUI

In the **🚀 Optimise** tab, configure your optimiser settings (population size,
DE generations, Adam steps, L-BFGS steps, weights), then click:

> 💾 **Export run script (.py)**

This downloads a file like `pychannelab_run_yyyymmdd_hhmmss.py` that embeds:
- the full MSM definition (states, transitions, parameters, bounds)
- all protocol configurations
- the experimental data (as numpy arrays)
- the optimisation weights
- **all optimiser hyperparameters as they were set in the GUI when you clicked export**

The script uses `TorchPipelineOptimizer` (DE → Adam → L-BFGS) and
automatically falls back to a scipy optimiser if PyTorch is not available.

### Step 2 — Retrieve results

The script writes to `pychannelab_output/`:

| File | Contents |
|------|----------|
| `fitted_params.json` | Fitted parameter values |
| `comparison.html` | Interactive Plotly figure (exp vs sim) |
| `comparison.png` | Static matplotlib figure |
| `report.md` | Markdown report (settings, parameters, AIC/BIC, curve-fit comparison) |

---

## 6. Running the tests

```bash
# All tests (unit + integration)
uv run pytest test/ -v

# Unit tests only (fast, no GPU needed)
uv run pytest test/unit/ -v

# Integration tests only (slower, runs the full DE pipeline)
uv run pytest test/integration/ -v
```

---

## 7. Building the documentation

```bash
# Build HTML docs (output in docs/_build/html/)
uv run sphinx-build -b html docs docs/_build/html

# or using the Makefile
cd docs && make html
```

Open `docs/_build/html/index.html` in your browser.

The API reference is generated automatically from docstrings in `core/` via
Sphinx `autodoc`.
