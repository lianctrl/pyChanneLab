#!/usr/bin/env python3
"""
Headless fit + visual verification harness for the WT example dataset.

Fits the 11-state Kv MSM to the four experimental protocols in
``examples/exp_dataset`` with the torch DE -> Adam -> L-BFGS pipeline, then:

  * simulates the fitted model on the experimental x-values,
  * reports per-protocol and total RMSE (printed as JSON on the last line),
  * writes a 2x2 comparison PNG (experimental points w/ SD error bars + fitted
    line) to ``examples/output/fit_wt_comparison.png``.

Usage (run from the pyChanneLab/ project root)::

    python examples/fit_wt.py                        # defaults
    python examples/fit_wt.py --pop 80 --de-gens 300 --starts 5 --seed 0

All hyperparameters are CLI flags so the fit can be swept without editing code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── make the src/ package importable when run from anywhere ──────────────────
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from pychannel_lab.core.data_loader import load_csv  # noqa: E402
from pychannel_lab.core.msm_builder import make_11state_preset  # noqa: E402
from pychannel_lab.core.config import (  # noqa: E402
    ActivationConfig,
    InactivationConfig,
    CSInactivationConfig,
    RecoveryConfig,
    G_K_MAX,
)
from pychannel_lab.core.torch_de import TorchPipelineOptimizer  # noqa: E402
from pychannel_lab.core.torch_simulator import (  # noqa: E402
    TorchProtocolSimulator,
    get_device,
    preferred_dtype,
)

_DATA_DIR = _HERE / "exp_dataset"
_OUT_DIR = _HERE / "output"

# Files -> protocol keys
_FILES = {
    "activation": "Activation_WT.csv",
    "inactivation": "Inactivation_WT.csv",
    "cs_inactivation": "CS-Inactivation_WT.csv",
    "recovery": "Recovery_WT.csv",
}

_PANEL = {
    "activation": ("Activation (G/V)", "pre-pulse voltage (mV)", "norm. conductance"),
    "inactivation": ("Inactivation (h∞/V)", "pre-pulse voltage (mV)", "norm. current"),
    "cs_inactivation": ("Closed-state inactivation", "prepulse duration (ms)", "norm. current"),
    "recovery": ("Recovery from inactivation", "interpulse interval (ms)", "I/Imax"),
}


def load_exp_data() -> dict:
    """Load the four WT CSVs into the {key: (x, y, err)} structure."""
    exp = {}
    for key, fname in _FILES.items():
        x, y, err = load_csv(_DATA_DIR / fname)
        exp[key] = (x, y, err)
    return exp


def _configs():
    """Default protocol configs (they match the WT data grids)."""
    return (
        ActivationConfig(),
        InactivationConfig(),
        CSInactivationConfig(),
        RecoveryConfig(),
    )


def simulate_fit(full_params, exp, csi_cfg, rec_cfg, device, dtype, n_peak_steps):
    """Simulate the fitted model at the experimental x-values.

    Returns {key: (x_exp, y_sim)} aligned with the experimental abscissae so
    residuals line up point-for-point.
    """
    p = torch.tensor(full_params, dtype=dtype, device=device)
    sim = TorchProtocolSimulator(
        p,
        msm_def=make_11state_preset(),
        n_peak_steps=n_peak_steps,
    )
    out = {}
    with torch.no_grad():
        x = np.asarray(exp["activation"][0], float)
        out["activation"] = (x, sim.run_activation(x).cpu().numpy())

        x = np.asarray(exp["inactivation"][0], float)
        out["inactivation"] = (x, sim.run_inactivation(x).cpu().numpy())

        x = np.asarray(exp["cs_inactivation"][0], float)
        x_sim = csi_cfg.t_initial + x / 1000.0
        out["cs_inactivation"] = (x, sim.run_cs_inactivation(x_sim).cpu().numpy())

        x = np.asarray(exp["recovery"][0], float)
        x_sim = rec_cfg.t_pulse + x / 1000.0
        out["recovery"] = (x, sim.run_recovery(x_sim).cpu().numpy())
    return out


def rmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_plot(exp, sim, metrics, out_path: Path, title_suffix=""):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    order = ["activation", "inactivation", "cs_inactivation", "recovery"]
    for ax, key in zip(axes.flat, order):
        title, xlab, ylab = _PANEL[key]
        x_exp, y_exp, err = exp[key]
        x_sim, y_sim = sim[key]
        if err is not None:
            ax.errorbar(
                x_exp, y_exp, yerr=err, fmt="o", ms=4, color="#1f77b4",
                ecolor="#9ecae1", capsize=2, label="experiment", zorder=3,
            )
        else:
            ax.plot(x_exp, y_exp, "o", ms=4, color="#1f77b4", label="experiment", zorder=3)
        ax.plot(x_sim, y_sim, "-", lw=2, color="#d62728", label="fit", zorder=2)
        ax.set_title(f"{title}   (RMSE={metrics[key]:.4f})", fontsize=11)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"pyChanneLab WT fit — total RMSE = {metrics['total']:.4f}{title_suffix}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit WT dataset and verify visually.")
    ap.add_argument("--pop", type=int, default=60, help="DE population size")
    ap.add_argument("--de-gens", type=int, default=250, help="DE generations")
    ap.add_argument("--adam", type=int, default=600, help="Adam steps")
    ap.add_argument("--adam-lr", type=float, default=0.05, help="Adam learning rate")
    ap.add_argument("--lbfgs", type=int, default=200, help="L-BFGS outer steps")
    ap.add_argument("--starts", type=int, default=5, help="top-K multi-start polish")
    ap.add_argument("--peak-steps", type=int, default=50, help="peak substeps")
    ap.add_argument("--strategy", type=str, default="adaptive",
                    help="DE strategy: 'adaptive' | 'rand1'")
    ap.add_argument("--weights", type=str, default="1,1,2,2",
                    help="per-protocol cost weights act,inact,csi,rec (default 1,1,2,2)")
    ap.add_argument("--reg", type=float, default=0.0,
                    help="L2 regularization lambda (scale-invariant pull toward the "
                         "initial guess; 0=off, try 1e-3..1e-1)")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--skip-de", action="store_true", help="skip DE, start local from preset")
    ap.add_argument("--device", type=str, default="cpu",
                    help="'cpu' | 'mps' | 'cuda' | 'auto'. CPU float64 is fastest here "
                         "because matrix_exp has no MPS kernel (falls back to CPU anyway).")
    ap.add_argument("--out", type=str, default=str(_OUT_DIR / "fit_wt_comparison.png"))
    ap.add_argument("--tag", type=str, default="", help="suffix for plot title")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    dtype = preferred_dtype(device)

    exp = load_exp_data()
    act_cfg, inact_cfg, csi_cfg, rec_cfg = _configs()
    msm = make_11state_preset()

    _wv = [float(x) for x in args.weights.split(",")]
    weights = dict(zip(["activation", "inactivation", "cs_inactivation", "recovery"], _wv))

    # Build the pipeline. `strategy` / `n_starts` are optional kwargs added in
    # the optimizer upgrade; fall back gracefully if not yet present.
    pipe_kw = dict(
        experimental_data=exp,
        weights=weights,
        reg_lambda=args.reg,
        msm_def=msm,
        act_cfg=act_cfg,
        inact_cfg=inact_cfg,
        csi_cfg=csi_cfg,
        rec_cfg=rec_cfg,
        g_k_max=G_K_MAX,
        n_peak_steps=args.peak_steps,
        device=device,
        dtype=dtype,
    )
    pipeline = TorchPipelineOptimizer(**pipe_kw)

    opt_kw = dict(
        bounds=list(msm.free_bounds),
        pop_size=args.pop,
        de_maxiter=args.de_gens,
        n_adam=args.adam,
        adam_lr=args.adam_lr,
        n_lbfgs=args.lbfgs,
        skip_de=args.skip_de,
        initial_params=msm.free_initial_guess if args.skip_de else None,
    )
    # Pass upgrade kwargs only if the running code supports them.
    import inspect

    sig = inspect.signature(pipeline.optimize)
    if "strategy" in sig.parameters:
        opt_kw["strategy"] = args.strategy
    if "n_starts" in sig.parameters:
        opt_kw["n_starts"] = args.starts
    if "seed" in sig.parameters:
        opt_kw["seed"] = args.seed

    t0 = time.time()
    last = [(-1, float("inf"))]

    def _cb(it, cost, conv):
        if it - last[0][0] >= 25 or cost < last[0][1] * 0.5:
            print(f"  iter {it:5d}  cost={cost:.6f}  ({time.time()-t0:5.1f}s)", flush=True)
            last[0] = (it, cost)

    result = pipeline.optimize(progress_callback=_cb, **opt_kw)
    elapsed = time.time() - t0

    full_params = msm.expand_params(result.x)
    sim = simulate_fit(full_params, exp, csi_cfg, rec_cfg, device, dtype, args.peak_steps)

    metrics = {}
    for key in _FILES:
        _, y_exp, _ = exp[key]
        _, y_sim = sim[key]
        metrics[key] = rmse(y_exp, y_sim)
    metrics["total"] = float(np.sqrt(np.mean([metrics[k] ** 2 for k in _FILES])))

    out_path = Path(args.out)
    tag = f"  [{args.tag}]" if args.tag else ""
    make_plot(exp, sim, metrics, out_path, title_suffix=tag)

    param_dict = {n: float(v) for n, v in zip(msm.param_names, full_params)}
    summary = {
        "rmse": metrics,
        "final_cost": float(result.fun),
        "elapsed_s": round(elapsed, 1),
        "png": str(out_path),
        "params": param_dict,
        "settings": {
            "pop": args.pop, "de_gens": args.de_gens, "adam": args.adam,
            "lbfgs": args.lbfgs, "starts": args.starts, "strategy": args.strategy,
            "seed": args.seed, "peak_steps": args.peak_steps, "reg": args.reg,
        },
    }
    print("\nPER-PROTOCOL RMSE:")
    for k in _FILES:
        print(f"  {k:16s} {metrics[k]:.4f}")
    print(f"  {'TOTAL':16s} {metrics['total']:.4f}")
    print(f"\nPNG written to: {out_path}")
    print("JSON " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
