#!/usr/bin/env python3
"""
Tabulate fit_wt.py runs of one variant into two CSVs.

  {variant}_params.csv  — one row per run: the 11 MSM parameters, per-protocol
                          and total RMSE (as logged by the run, i.e. at its own
                          --peak-steps, matching its PNG), final cost,
                          resting I(-90 mV).
  {variant}_gating.csv  — Table S1-style gating fits (value and SE) of the
                          experimental points and of each run's simulation:
                            activation    1/(1+exp((V½-v)/k))
                            inactivation  1/(1+exp((v-V½)/k))
                            CSI           C+(1-C)·exp(-t/τ)   (free C)
                            recovery      1-exp(-t/τ)

Usage (run from the pyChanneLab/ project root)::

    python examples/summarize_fits.py --data-dir ~/Desktop/SNS-PhD/Exp-dataset/NoKCHIP \\
        --variant M373I --runs seed7=examples/output/M373I_seed7.log ... --out-dir OUT
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

import fit_wt
from pychannel_lab.core.msm_builder import make_11state_preset
from pychannel_lab.core.torch_simulator import TorchProtocolSimulator

PROTOCOLS = ["activation", "inactivation", "cs_inactivation", "recovery"]

# protocol -> (function, p0, column names)
GATING = {
    "activation": (lambda v, V, k: 1 / (1 + np.exp((V - v) / k)), [15, 15],
                   ["act_V_half_mV", "act_k_mV"]),
    "inactivation": (lambda v, V, k: 1 / (1 + np.exp((v - V) / k)), [-55, 7],
                     ["inact_V_half_mV", "inact_k_mV"]),
    "cs_inactivation": (lambda t, tau, c: c + (1 - c) * np.exp(-t / tau), [200, 0.3],
                        ["csi_tau_ms", "csi_C"]),
    "recovery": (lambda t, tau: 1 - np.exp(-t / tau), [250], ["rec_tau_ms"]),
}


def simulate(params: dict, exp: dict) -> dict:
    """Model curves at the experimental x-values (n_peak_steps=50)."""
    msm = make_11state_preset()
    act, inact, csi, rec = fit_wt._configs()
    sim = TorchProtocolSimulator(
        torch.tensor([params[n] for n in msm.param_names], dtype=torch.float64),
        msm_def=msm, act_cfg=act, inact_cfg=inact, csi_cfg=csi, rec_cfg=rec,
        n_peak_steps=50,
    )
    x = {k: np.asarray(exp[k][0], float) for k in PROTOCOLS}
    with torch.no_grad():
        return {
            "activation": sim.run_activation(x["activation"]).numpy(),
            "inactivation": sim.run_inactivation(x["inactivation"]).numpy(),
            "cs_inactivation": sim.run_cs_inactivation(
                csi.t_initial + x["cs_inactivation"] / 1000).numpy(),
            "recovery": sim.run_recovery(rec.t_pulse + x["recovery"] / 1000).numpy(),
        }


def gating_row(exp: dict, ys: dict) -> dict:
    row = {}
    for key, (f, p0, cols) in GATING.items():
        popt, pcov = curve_fit(f, np.asarray(exp[key][0], float), ys[key], p0=p0, maxfev=50000)
        for col, v, e in zip(cols, popt, np.sqrt(np.diag(pcov))):
            row[col] = v
            row[col + "_SE"] = e
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tabulate fit_wt.py runs of one variant.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--runs", nargs="+", required=True, metavar="LABEL=LOG")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    exp = fit_wt.load_exp_data(Path(args.data_dir).expanduser(), args.variant)
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    names = make_11state_preset().param_names

    param_rows, gating_rows = [], []
    gating_rows.append({"source": "exp_csv", **gating_row(
        exp, {k: np.asarray(exp[k][1], float) for k in PROTOCOLS})})
    for item in args.runs:
        label, log = item.split("=", 1)
        js = json.loads([ln for ln in open(log) if ln.startswith("JSON ")][-1][5:])
        params = js["params"]
        sim = simulate(params, exp)
        param_rows.append({
            "run": label,
            **{n: params[n] for n in names},
            **{f"rmse_{k}": v for k, v in js["rmse"].items()},
            "final_cost": js["final_cost"],
            "resting_I_-90mV": js.get("resting_I", fit_wt.resting_inactivation(
                [params[n] for n in names])),
        })
        gating_rows.append({"source": f"sim_{label}", **gating_row(exp, sim)})

    write_csv(out / f"{args.variant}_params.csv", param_rows)
    write_csv(out / f"{args.variant}_gating.csv", gating_rows)
    print(f"wrote {out / f'{args.variant}_params.csv'} and {args.variant}_gating.csv")


if __name__ == "__main__":
    main()
