"""
Shared constants, helper functions, and run-script generator for pyChanneLab.
All tab modules import from here so the main app.py stays thin.
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.config import (
    ActivationConfig,
    InactivationConfig,
    CSInactivationConfig,
    RecoveryConfig,
)
from core.msm_builder import MSMDefinition, compute_layout
from core.simulator import ProtocolSimulator
from core.curve_fitter import (
    fit_curve,
    eval_curve,
    compute_aic_bic,
    CURVE_FUNCTIONS,
    CURVE_LABELS,
    PROTOCOL_CURVE_DEFAULTS,
)

# ── Protocol metadata ──────────────────────────────────────────────────────────
PROTOCOL_KEYS = ["activation", "inactivation", "cs_inactivation", "recovery"]

PROTOCOL_LABELS = {
    "activation": "Activation (G/V)",
    "inactivation": "Inactivation (h∞/V)",
    "cs_inactivation": "Closed-State Inactivation",
    "recovery": "Recovery from Inactivation",
}
PROTOCOL_X_LABELS = {
    "activation": "Test Voltage (mV)",
    "inactivation": "Conditioning Voltage (mV)",
    "cs_inactivation": "Prepulse Duration (ms)",
    "recovery": "Recovery Interval (ms)",
}
PROTOCOL_Y_LABELS = {
    "activation": "g / g_max",
    "inactivation": "I / I_max",
    "cs_inactivation": "I / I_max",
    "recovery": "I_test / I_pre",
}

TYPE_COLORS = {"closed": "#4878d0", "inactivated": "#ee854a", "open": "#6acc65"}
TYPE_LABELS = {"closed": "Closed", "inactivated": "Inactivated", "open": "Open"}


# ── Session-state helpers ──────────────────────────────────────────────────────


def _ic() -> np.ndarray:
    """Return current initial conditions (from session or msm_def default)."""
    ss = st.session_state
    if (
        ss.initial_conditions is not None
        and len(ss.initial_conditions) == ss.msm_def.n_states
    ):
        return ss.initial_conditions
    return ss.msm_def.default_initial_conditions


def _build_sim(params: np.ndarray) -> ProtocolSimulator:
    ss = st.session_state
    return ProtocolSimulator(
        params,
        msm_def=ss.msm_def,
        act_cfg=ss.act_cfg,
        inact_cfg=ss.inact_cfg,
        csi_cfg=ss.csi_cfg,
        rec_cfg=ss.rec_cfg,
        t_total=ss.t_total,
        dt=ss.dt,
        g_k_max=ss.g_k_max,
        initial_state=_ic(),
    )


def _simulate_all(params: np.ndarray) -> dict:
    ss = st.session_state
    sim = _build_sim(params)
    csi_x_ms = (sim.csi_proto.get_test_times() - ss.csi_cfg.t_initial) * 1000.0
    rec_x_ms = (sim.rec_proto.get_test_times() - ss.rec_cfg.t_pulse) * 1000.0
    return {
        "activation": (sim.act_proto.get_test_voltages(), sim.run_activation()),
        "inactivation": (sim.inact_proto.get_test_voltages(), sim.run_inactivation()),
        "cs_inactivation": (csi_x_ms, sim.run_cs_inactivation()),
        "recovery": (rec_x_ms, sim.run_recovery()),
    }


# ── Figure builders ────────────────────────────────────────────────────────────


def _network_figure(definition: MSMDefinition) -> go.Figure:
    layout = compute_layout(definition)
    edge_pairs = {(t.from_state, t.to_state) for t in definition.transitions}

    fig = go.Figure()
    annotations = []

    for tr in definition.transitions:
        if tr.from_state not in layout or tr.to_state not in layout:
            continue
        x0, y0 = layout[tr.from_state]
        x1, y1 = layout[tr.to_state]

        is_bi = (tr.to_state, tr.from_state) in edge_pairs
        if is_bi:
            dx, dy = x1 - x0, y1 - y0
            d = max(math.hypot(dx, dy), 1e-9)
            px, py = -dy / d * 0.1, dx / d * 0.1
        else:
            px, py = 0.0, 0.0

        shrink = 0.18
        ratio = 1.0 - shrink / max(math.hypot(x1 - x0, y1 - y0), 1e-9)
        ax_ = x0 + px
        ay_ = y0 + py
        x_ = x0 + (x1 - x0) * ratio + px
        y_ = y0 + (y1 - y0) * ratio + py

        label = tr.rate_expr if len(tr.rate_expr) <= 30 else tr.rate_expr[:28] + "…"
        annotations.append(
            dict(
                x=x_,
                y=y_,
                ax=ax_,
                ay=ay_,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor="#888",
                text=label,
                font=dict(size=9, color="#555"),
                bgcolor="rgba(255,255,255,0.7)",
                hovertext=tr.rate_expr,
            )
        )

    for stype in ["closed", "inactivated", "open"]:
        group = [s for s in definition.states if s.state_type == stype]
        if not group:
            continue
        xs = [layout[s.name][0] for s in group if s.name in layout]
        ys = [layout[s.name][1] for s in group if s.name in layout]
        names = [s.name for s in group if s.name in layout]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=TYPE_LABELS[stype],
                text=names,
                textposition="middle center",
                textfont=dict(size=11, color="white", family="monospace"),
                marker=dict(
                    size=42, color=TYPE_COLORS[stype], line=dict(width=2, color="white")
                ),
                hovertext=[f"{n} ({stype})" for n in names],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
        height=420,
        margin=dict(t=20, b=60, l=20, r=20),
        plot_bgcolor="rgba(245,247,250,1)",
    )
    return fig


def _voltage_preview(proto_key: str) -> go.Figure:
    from core.protocols import (
        ActivationProtocol,
        InactivationProtocol,
        CSInactivationProtocol,
        RecoveryProtocol,
    )

    ss = st.session_state
    dt = 0.001
    t = np.arange(0, ss.t_total + dt, dt)
    fig = go.Figure()

    if proto_key == "activation":
        p = ActivationProtocol(ss.act_cfg)
        for V in [
            ss.act_cfg.v_min,
            (ss.act_cfg.v_min + ss.act_cfg.v_max) / 2,
            ss.act_cfg.v_max,
        ]:
            vf = p.get_voltage_function(V)
            fig.add_trace(
                go.Scatter(
                    x=t, y=[vf(ti) for ti in t], mode="lines", name=f"V_test={V:.0f} mV"
                )
            )

    elif proto_key == "inactivation":
        p = InactivationProtocol(ss.inact_cfg)
        for V in [
            ss.inact_cfg.v_min,
            (ss.inact_cfg.v_min + ss.inact_cfg.v_max) / 2,
            ss.inact_cfg.v_max,
        ]:
            vf = p.get_voltage_function(V)
            fig.add_trace(
                go.Scatter(
                    x=t, y=[vf(ti) for ti in t], mode="lines", name=f"V_cond={V:.0f} mV"
                )
            )

    elif proto_key == "cs_inactivation":
        p = CSInactivationProtocol(ss.csi_cfg)
        times = p.get_test_times()
        for tp in [times[0], times[len(times) // 2], times[-1]]:
            vf = p.get_voltage_function(tp)
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=[vf(ti) for ti in t],
                    mode="lines",
                    name=f"t_pulse={tp:.3f} s",
                )
            )

    elif proto_key == "recovery":
        p = RecoveryProtocol(ss.rec_cfg)
        times = p.get_test_times()
        for tp in [times[0], times[len(times) // 2], times[-1]]:
            vf = p.get_voltage_function(tp)
            fig.add_trace(
                go.Scatter(
                    x=t, y=[vf(ti) for ti in t], mode="lines", name=f"t_rec={tp:.3f} s"
                )
            )

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Voltage (mV)",
        height=260,
        margin=dict(t=10, b=40, l=60, r=20),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def _comparison_figure(params: np.ndarray, _sim_data: dict = None) -> go.Figure:
    ss = st.session_state
    sim_data = _sim_data if _sim_data is not None else _simulate_all(params)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[PROTOCOL_LABELS[k] for k in PROTOCOL_KEYS],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (row, col), pk in zip(positions, PROTOCOL_KEYS):
        x_sim, y_sim = sim_data[pk]
        exp = ss.exp_data.get(pk)

        if exp is not None:
            x_exp, y_exp, y_err = exp
            err_kw = (
                dict(error_y=dict(type="data", array=y_err, visible=True))
                if y_err is not None
                else {}
            )
            fig.add_trace(
                go.Scatter(
                    x=x_exp,
                    y=y_exp,
                    mode="markers",
                    name="Experimental",
                    marker=dict(color="#1f77b4", size=8),
                    showlegend=(row == 1 and col == 1),
                    **err_kw,
                ),
                row=row,
                col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=x_sim,
                y=y_sim,
                mode="lines",
                name="Simulated",
                line=dict(color="#d62728", width=2),
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text=PROTOCOL_X_LABELS[pk], row=row, col=col)
        fig.update_yaxes(title_text=PROTOCOL_Y_LABELS[pk], row=row, col=col)

    fig.update_layout(height=700, title_text="Experimental vs Simulated")
    return fig


def _curve_fit_comparison_table(sim_data: dict) -> pd.DataFrame:
    ss = st.session_state
    rows = []
    for pk in PROTOCOL_KEYS:
        if pk not in ss.exp_data or pk not in sim_data:
            continue
        cft = ss.curve_fit_types.get(pk, PROTOCOL_CURVE_DEFAULTS[pk])
        _, formula, param_names = CURVE_FUNCTIONS[cft]

        exp_entry = ss.curve_fit_params_exp.get(pk)
        if exp_entry is not None:
            popt_exp, perr_exp, ok_exp = exp_entry[0], exp_entry[1], exp_entry[2]
        else:
            x_e, y_e, _ = ss.exp_data[pk]
            popt_exp, perr_exp, ok_exp = fit_curve(x_e, y_e, cft)

        x_s, y_s = sim_data[pk]
        popt_sim, perr_sim, ok_sim = fit_curve(x_s, y_s, cft)
        ss.curve_fit_params_sim[pk] = (popt_sim, perr_sim, ok_sim, cft)

        for i, pn in enumerate(param_names):

            def _fmt(ok, popt, perr, idx):
                if not ok or np.isnan(popt[idx]):
                    return "—"
                e = perr[idx]
                if np.isnan(e):
                    return f"{popt[idx]:.4g}"
                return f"{popt[idx]:.4g} ± {e:.4g}"

            rows.append(
                {
                    "Protocol": PROTOCOL_LABELS[pk],
                    "Fit function": formula,
                    "Parameter": pn,
                    "Exp fit": _fmt(ok_exp, popt_exp, perr_exp, i),
                    "Sim fit": _fmt(ok_sim, popt_sim, perr_sim, i),
                }
            )
    return pd.DataFrame(rows)


# ── Run-script generator ───────────────────────────────────────────────────────


def generate_run_script() -> str:
    """
    Return a self-contained Python script that reproduces the current optimisation.
    All settings are read from session state at call time (i.e. when the user
    presses the download button), so the script always reflects the latest UI state.
    """
    import dataclasses

    ss = st.session_state
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Read optimisation hyperparameters from session state ──────────────────
    # These are stored automatically by Streamlit via widget keys in tab_optimise.
    _run_global = bool(ss.get("opt_run_global", True))
    _pop_size = int(ss.get("opt_pop_size", 50))
    _de_maxiter = int(ss.get("opt_de_maxiter", 200))
    _de_F = float(ss.get("opt_de_F", 0.8))
    _de_CR = float(ss.get("opt_de_CR", 0.9))
    _n_adam = int(ss.get("opt_n_adam", 500))
    _adam_lr = float(ss.get("opt_adam_lr", 0.05))
    _n_lbfgs = int(ss.get("opt_lbfgs", 200))
    _n_peak_steps = int(ss.get("opt_peak_steps", 50))

    def cfg_kwargs(cfg) -> str:
        return ", ".join(
            f"{f.name}={getattr(cfg, f.name)!r}" for f in dataclasses.fields(cfg)
        )

    # ── Serialise experimental data as numpy literals ─────────────────────────
    exp_lines: list[str] = []
    for pk in PROTOCOL_KEYS:
        tup = ss.exp_data.get(pk)
        if tup is None:
            exp_lines.append(f'    "{pk}": None,')
            continue
        x, y, y_err = tup
        xl = f"np.array({x.tolist()})"
        yl = f"np.array({y.tolist()})"
        yel = f"np.array({y_err.tolist()})" if y_err is not None else "None"
        exp_lines.append(f'    "{pk}": ({xl}, {yl}, {yel}),')

    ic_arr = _ic()

    L: list[str] = []

    def w(line: str = "") -> None:
        L.append(line)

    # ── Header ────────────────────────────────────────────────────────────────
    w("#!/usr/bin/env python3")
    w(f'"""')
    w(f"pyChanneLab auto-generated run script.")
    w(f"Generated: {ts}")
    w()
    w("Steps executed:")
    w("  1. Global + local optimisation (TorchPipelineOptimizer: DE → Adam → L-BFGS)")
    w("  2. Simulate fitted model")
    w("  3. Compute AIC / BIC")
    w("  4. Fit phenomenological curves to exp and sim data")
    w("  5. Save comparison plots (plotly HTML + matplotlib PNG if available)")
    w("  6. Write Markdown report")
    w('"""')
    w()
    w("import sys, json")
    w("from pathlib import Path")
    w("import numpy as np")
    w()
    w("# ── path setup ──────────────────────────────────────────────────────────")
    w("# Place this script alongside app.py (inside pyChanneLab/src/pychannel_lab/)")
    w("# before running: python pychannelab_run_<ts>.py")
    w("_HERE = Path(__file__).resolve().parent")
    w("sys.path.insert(0, str(_HERE))")
    w()
    w(
        "from core.config import ActivationConfig, InactivationConfig, CSInactivationConfig, RecoveryConfig"
    )
    w("from core.msm_builder import MSMDefinition")
    w("from core.simulator import ProtocolSimulator")
    w(
        "from core.curve_fitter import fit_curve, eval_curve, compute_aic_bic, CURVE_LABELS, CURVE_FUNCTIONS"
    )
    w()
    w('OUT_DIR = Path("pychannelab_output")')
    w("OUT_DIR.mkdir(exist_ok=True)")
    w()

    # ── MSM definition ────────────────────────────────────────────────────────
    msm_json_str = ss.msm_def.to_json()
    w("# ── MSM definition (embedded) ──────────────────────────────────────────")
    w(f"_MSM_JSON = {repr(msm_json_str)}")
    w("msm_def = MSMDefinition.from_json(_MSM_JSON)")
    w()

    # ── Protocol configs ──────────────────────────────────────────────────────
    w("# ── protocol configs ───────────────────────────────────────────────────")
    w(f"act_cfg   = ActivationConfig({cfg_kwargs(ss.act_cfg)})")
    w(f"inact_cfg = InactivationConfig({cfg_kwargs(ss.inact_cfg)})")
    w(f"csi_cfg   = CSInactivationConfig({cfg_kwargs(ss.csi_cfg)})")
    w(f"rec_cfg   = RecoveryConfig({cfg_kwargs(ss.rec_cfg)})")
    w(f"g_k_max = {ss.g_k_max!r}")
    w(f"t_total = {ss.t_total!r}")
    w(f"dt      = {ss.dt!r}")
    w(f"initial_state = np.array({ic_arr.tolist()})")
    w()

    # ── Experimental data ─────────────────────────────────────────────────────
    w("# ── experimental data (embedded) ──────────────────────────────────────")
    w("exp_data = {")
    for line in exp_lines:
        w(line)
    w("}")
    w()

    # ── Optimisation weights ──────────────────────────────────────────────────
    w("# ── optimisation weights ───────────────────────────────────────────────")
    w("weights = {")
    for pk, wt in ss.opt_weights.items():
        w(f'    "{pk}": {wt},')
    w("}")
    w()

    # ── Curve-fit types ───────────────────────────────────────────────────────
    w("# ── curve-fit function per protocol ────────────────────────────────────")
    w("curve_fit_types = {")
    for pk in PROTOCOL_KEYS:
        cft = ss.curve_fit_types.get(pk, PROTOCOL_CURVE_DEFAULTS[pk])
        w(f'    "{pk}": "{cft}",')
    w("}")
    w()

    # ── Display labels ────────────────────────────────────────────────────────
    w('PROTOCOL_KEYS = ["activation", "inactivation", "cs_inactivation", "recovery"]')
    w("PROTOCOL_LABELS = {")
    for pk, lbl in PROTOCOL_LABELS.items():
        w(f'    "{pk}": "{lbl}",')
    w("}")
    w("PROTOCOL_X_LABELS = {")
    for pk, lbl in PROTOCOL_X_LABELS.items():
        w(f'    "{pk}": "{lbl}",')
    w("}")
    w("PROTOCOL_Y_LABELS = {")
    for pk, lbl in PROTOCOL_Y_LABELS.items():
        w(f'    "{pk}": "{lbl}",')
    w("}")
    w()

    # ── Optimisation hyperparameters (from GUI at export time) ────────────────
    w(
        "# ── optimisation hyperparameters (as set in the GUI when script was exported) ─"
    )
    w(f"RUN_GLOBAL   = {_run_global!r}")
    w(f"POP_SIZE     = {_pop_size}")
    w(f"DE_MAXITER   = {_de_maxiter}")
    w(f"DE_F         = {_de_F}")
    w(f"DE_CR        = {_de_CR}")
    w(f"N_ADAM       = {_n_adam}")
    w(f"ADAM_LR      = {_adam_lr}")
    w(f"N_LBFGS      = {_n_lbfgs}")
    w(f"N_PEAK_STEPS = {_n_peak_steps}")
    w()

    # ── Step 1: optimisation (Torch pipeline, scipy fallback) ─────────────────
    w("# " + "─" * 72)
    w("# STEP 1  Optimisation  (DE → Adam → L-BFGS via TorchPipelineOptimizer)")
    w("#         Falls back to scipy differential_evolution + L-BFGS-B if PyTorch")
    w("#         is not available on this machine.")
    w("# " + "─" * 72)
    w()
    w("try:")
    w("    import torch")
    w("    from core.torch_simulator import get_device, preferred_dtype")
    w("    from core.torch_de import TorchPipelineOptimizer")
    w("    _TORCH = True")
    w("except ImportError:")
    w("    _TORCH = False")
    w()
    w("if _TORCH:")
    w("    device = get_device()")
    w("    dtype  = preferred_dtype(device)")
    w('    print(f"Using PyTorch device: {device}")')
    w("    pipeline = TorchPipelineOptimizer(")
    w("        exp_data,")
    w("        weights=weights,")
    w("        msm_def=msm_def,")
    w("        act_cfg=act_cfg, inact_cfg=inact_cfg, csi_cfg=csi_cfg, rec_cfg=rec_cfg,")
    w("        g_k_max=g_k_max, t_total=t_total,")
    w("        n_peak_steps=N_PEAK_STEPS,")
    w("        device=device, dtype=dtype,")
    w("    )")
    w("    result = pipeline.optimize(")
    w("        bounds=list(msm_def.free_bounds),")
    w("        pop_size=POP_SIZE,")
    w("        de_maxiter=DE_MAXITER if RUN_GLOBAL else 0,")
    w("        F=DE_F,")
    w("        CR=DE_CR,")
    w("        n_adam=N_ADAM,")
    w("        adam_lr=ADAM_LR,")
    w("        n_lbfgs=N_LBFGS,")
    w("        skip_de=not RUN_GLOBAL,")
    w("        initial_params=msm_def.free_initial_guess if not RUN_GLOBAL else None,")
    w("    )")
    w("    fitted_params = msm_def.expand_params(result.x)")
    w('    print(f"Final cost (Torch): {result.fun:.6f}")')
    w()
    w("else:")
    w('    print("PyTorch not available — falling back to scipy.")')
    w("    from core.optimizer import CostFunction, ParameterOptimizer")
    w("    cost_fn = CostFunction(")
    w("        exp_data, weights=weights, msm_def=msm_def,")
    w("        act_cfg=act_cfg, inact_cfg=inact_cfg, csi_cfg=csi_cfg, rec_cfg=rec_cfg,")
    w("        g_k_max=g_k_max, t_total=t_total, dt=dt,")
    w("    )")
    w("    optimizer = ParameterOptimizer(cost_fn)")
    w("    bounds = list(msm_def.bounds)")
    w("    if RUN_GLOBAL:")
    w('        print("Running scipy Differential Evolution…")')
    w("        result = optimizer.optimize_global(bounds=bounds, maxiter=DE_MAXITER)")
    w('        print(f"  cost = {result.fun:.6f}")')
    w("    else:")
    w('        result = type("R", (), {"x": msm_def.initial_guess})()')
    w('    print("Running scipy L-BFGS-B…")')
    w("    result = optimizer.optimize_local(initial_guess=result.x, bounds=bounds)")
    w("    fitted_params = result.x")
    w('    print(f"Final cost (scipy): {result.fun:.6f}")')
    w()
    w('print("\\nFitted parameters:")')
    w("for pspec, val in zip(msm_def.parameters, fitted_params):")
    w('    print(f"  {pspec.name:20s} = {val:.6g}")')
    w()
    w("# Save parameters as JSON")
    w('_params_path = OUT_DIR / "fitted_params.json"')
    w("_params_path.write_text(json.dumps(")
    w('    {"timestamp": "' + ts + '",')
    w(
        '     "parameters": {p.name: float(v) for p, v in zip(msm_def.parameters, fitted_params)},'
    )
    w('     "model": msm_def.to_dict()},')
    w("    indent=2,")
    w("))")
    w('print(f"Saved: {_params_path}")')
    w()

    # ── Step 2: simulate ──────────────────────────────────────────────────────
    w("# " + "─" * 72)
    w("# STEP 2  Simulate with fitted parameters")
    w("# " + "─" * 72)
    w("sim = ProtocolSimulator(")
    w("    fitted_params,")
    w("    msm_def=msm_def,")
    w("    act_cfg=act_cfg, inact_cfg=inact_cfg, csi_cfg=csi_cfg, rec_cfg=rec_cfg,")
    w("    t_total=t_total, dt=dt, g_k_max=g_k_max, initial_state=initial_state,")
    w(")")
    w("csi_x_ms = (sim.csi_proto.get_test_times() - csi_cfg.t_initial) * 1000.0")
    w("rec_x_ms = (sim.rec_proto.get_test_times() - rec_cfg.t_pulse)   * 1000.0")
    w("sim_data = {")
    w(
        '    "activation":      (sim.act_proto.get_test_voltages(),   sim.run_activation()),'
    )
    w(
        '    "inactivation":    (sim.inact_proto.get_test_voltages(), sim.run_inactivation()),'
    )
    w('    "cs_inactivation": (csi_x_ms, sim.run_cs_inactivation()),')
    w('    "recovery":        (rec_x_ms, sim.run_recovery()),')
    w("}")
    w()

    # ── Step 3: AIC/BIC ───────────────────────────────────────────────────────
    w("# " + "─" * 72)
    w("# STEP 3  AIC / BIC")
    w("# " + "─" * 72)
    w("ab = compute_aic_bic(fitted_params, exp_data, sim_data)")
    w("print(f\"\\nAIC = {ab['AIC']:.3f}  |  BIC = {ab['BIC']:.3f}\")")
    w(
        "print(f\"  n={int(ab['n_points'])}, k={int(ab['k_params'])}, RSS={ab['RSS']:.4g}\")"
    )
    w()

    # ── Step 4: curve fits ────────────────────────────────────────────────────
    w("# " + "─" * 72)
    w("# STEP 4  Phenomenological curve fits")
    w("# " + "─" * 72)
    w("curve_fit_results = {}")
    w("for pk, cft in curve_fit_types.items():")
    w("    entry = {}")
    w("    if pk in exp_data and exp_data[pk] is not None:")
    w("        x_e, y_e, _ = exp_data[pk]")
    w('        entry["exp"] = fit_curve(x_e, y_e, cft)')
    w("    if pk in sim_data:")
    w("        x_s, y_s = sim_data[pk]")
    w('        entry["sim"] = fit_curve(x_s, y_s, cft)')
    w("    curve_fit_results[pk] = entry")
    w()

    # ── Step 5: plots ─────────────────────────────────────────────────────────
    w("# " + "─" * 72)
    w("# STEP 5  Save comparison plots")
    w("# " + "─" * 72)
    w("try:")
    w("    import plotly.graph_objects as go")
    w("    from plotly.subplots import make_subplots")
    w("    fig = make_subplots(rows=2, cols=2,")
    w("                        subplot_titles=list(PROTOCOL_LABELS.values()),")
    w("                        vertical_spacing=0.18, horizontal_spacing=0.12)")
    w("    for (row, col), pk in zip([(1,1),(1,2),(2,1),(2,2)], PROTOCOL_KEYS):")
    w("        x_s, y_s = sim_data[pk]")
    w("        first = (row == 1 and col == 1)")
    w("        if pk in exp_data and exp_data[pk] is not None:")
    w("            x_e, y_e, y_err = exp_data[pk]")
    w(
        '            ekw = dict(error_y=dict(type="data", array=(y_err.tolist() if y_err is not None else None), visible=True)) if y_err is not None else {}'
    )
    w(
        '            fig.add_trace(go.Scatter(x=x_e.tolist(), y=y_e.tolist(), mode="markers",'
    )
    w(
        '                                     name="Exp", marker=dict(color="#1f77b4", size=8),'
    )
    w(
        "                                     showlegend=first, **ekw), row=row, col=col)"
    )
    w('        fig.add_trace(go.Scatter(x=x_s.tolist(), y=y_s.tolist(), mode="lines",')
    w(
        '                                 name="Sim", line=dict(color="#d62728", width=2),'
    )
    w("                                 showlegend=first), row=row, col=col)")
    w("        entry = curve_fit_results.get(pk, {})")
    w('        popt_s, _, ok_s = entry.get("sim", (None, None, False))')
    w("        if ok_s and popt_s is not None and not np.any(np.isnan(popt_s)):")
    w("            x_fit = np.linspace(float(x_s.min()), float(x_s.max()), 300)")
    w("            y_fit = eval_curve(x_fit, popt_s, curve_fit_types[pk])")
    w(
        '            fig.add_trace(go.Scatter(x=x_fit.tolist(), y=y_fit.tolist(), mode="lines",'
    )
    w(
        '                                     name="Fit (sim)", line=dict(color="#2ca02c", width=1.5, dash="dot"),'
    )
    w("                                     showlegend=first), row=row, col=col)")
    w("        fig.update_xaxes(title_text=PROTOCOL_X_LABELS[pk], row=row, col=col)")
    w("        fig.update_yaxes(title_text=PROTOCOL_Y_LABELS[pk], row=row, col=col)")
    w('    fig.update_layout(height=700, title_text="Experimental vs Simulated")')
    w('    _html = OUT_DIR / "comparison.html"')
    w("    fig.write_html(str(_html))")
    w('    print(f"Saved: {_html}")')
    w("except Exception as _e:")
    w('    print(f"Plotly plot skipped: {_e}")')
    w()
    w("try:")
    w("    import matplotlib.pyplot as plt")
    w("    fig_mpl, axes = plt.subplots(2, 2, figsize=(12, 9))")
    w("    for ax, pk in zip(axes.flat, PROTOCOL_KEYS):")
    w("        x_s, y_s = sim_data[pk]")
    w('        ax.plot(x_s, y_s, color="#d62728", linewidth=2, label="Sim")')
    w("        if pk in exp_data and exp_data[pk] is not None:")
    w("            x_e, y_e, y_err = exp_data[pk]")
    w("            if y_err is not None:")
    w('                ax.errorbar(x_e, y_e, yerr=y_err, fmt="o",')
    w('                            color="#1f77b4", markersize=5, label="Exp")')
    w("            else:")
    w('                ax.scatter(x_e, y_e, color="#1f77b4", s=25, label="Exp")')
    w("            entry = curve_fit_results.get(pk, {})")
    w('            popt_e, _, ok_e = entry.get("exp", (None, None, False))')
    w("            if ok_e and popt_e is not None and not np.any(np.isnan(popt_e)):")
    w("                x_fit = np.linspace(float(x_e.min()), float(x_e.max()), 300)")
    w("                y_fit = eval_curve(x_fit, popt_e, curve_fit_types[pk])")
    w('                ax.plot(x_fit, y_fit, "--", color="#ff7f0e",')
    w('                        linewidth=1.5, label="Fit (exp)")')
    w("        ax.set_xlabel(PROTOCOL_X_LABELS[pk])")
    w("        ax.set_ylabel(PROTOCOL_Y_LABELS[pk])")
    w("        ax.set_title(PROTOCOL_LABELS[pk])")
    w("        ax.legend(fontsize=8)")
    w("    plt.tight_layout()")
    w('    _png = OUT_DIR / "comparison.png"')
    w("    fig_mpl.savefig(str(_png), dpi=150)")
    w("    plt.close(fig_mpl)")
    w('    print(f"Saved: {_png}")')
    w("except ImportError:")
    w('    print("matplotlib not available — PNG skipped")')
    w("except Exception as _e:")
    w('    print(f"matplotlib plot skipped: {_e}")')
    w()

    # ── Step 6: Markdown report ───────────────────────────────────────────────
    w("# " + "─" * 72)
    w("# STEP 6  Markdown report")
    w("# " + "─" * 72)
    w("def _fmt(entry_dict, key, idx):")
    w("    e = entry_dict.get(key)")
    w('    if e is None: return "—"')
    w("    popt, perr, ok = e")
    w('    if not ok or np.isnan(popt[idx]): return "—"')
    w(
        '    return f"{popt[idx]:.4g}" if np.isnan(perr[idx]) else f"{popt[idx]:.4g} ± {perr[idx]:.4g}"'
    )
    w()
    w(f'_ts = "{ts}"')
    w("_n_s = msm_def.n_states")
    w("_n_t = len(msm_def.transitions)")
    w("_n_p = len(msm_def.parameters)")
    w()
    w("md = []")
    w('md.append("# pyChanneLab Optimisation Report")')
    w('md.append("")')
    w('md.append(f"**Generated:** {_ts}  ")')
    w(
        'md.append(f"**Model:** {_n_s} states · {_n_t} transitions · {_n_p} free parameters")'
    )
    w('md.append("")')
    w('md.append("---")')
    w('md.append("")')
    w('md.append("## 1. Optimisation settings")')
    w('md.append("")')
    w('md.append("| Setting | Value |")')
    w('md.append("|---------|-------|")')
    w(f'md.append("| run_global | {_run_global} |")')
    w(f'md.append("| pop_size | {_pop_size} |")')
    w(f'md.append("| de_maxiter | {_de_maxiter} |")')
    w(f'md.append("| de_F | {_de_F} |")')
    w(f'md.append("| de_CR | {_de_CR} |")')
    w(f'md.append("| n_adam | {_n_adam} |")')
    w(f'md.append("| adam_lr | {_adam_lr} |")')
    w(f'md.append("| n_lbfgs | {_n_lbfgs} |")')
    w(f'md.append("| n_peak_steps | {_n_peak_steps} |")')
    w('md.append("")')
    w('md.append("---")')
    w('md.append("")')
    w('md.append("## 2. Optimised parameters")')
    w('md.append("")')
    w('md.append("| Parameter | Fitted value | Bounds |")')
    w('md.append("|-----------|-------------|--------|")')
    w("for pspec, val in zip(msm_def.parameters, fitted_params):")
    w(
        '    md.append(f"| {pspec.name} | {val:.6g} | [{pspec.lower_bound}, {pspec.upper_bound}] |")'
    )
    w('md.append("")')
    w('md.append("---")')
    w('md.append("")')
    w('md.append("## 3. Model information criteria (AIC / BIC)")')
    w('md.append("")')
    w('md.append("| Metric | Value |")')
    w('md.append("|--------|-------|")')
    w("md.append(f\"| AIC | {ab['AIC']:.3f} |\")")
    w("md.append(f\"| BIC | {ab['BIC']:.3f} |\")")
    w("md.append(f\"| RSS | {ab['RSS']:.4g} |\")")
    w("md.append(f\"| n data points | {int(ab['n_points'])} |\")")
    w("md.append(f\"| k free parameters | {int(ab['k_params'])} |\")")
    w('md.append("")')
    w('md.append("> Lower AIC/BIC = better fit relative to model complexity.")')
    w('md.append("")')
    w('md.append("---")')
    w('md.append("")')
    w('md.append("## 4. Phenomenological curve-fit comparison")')
    w('md.append("")')
    w('md.append("| Protocol | Fit function | Parameter | Exp fit | Sim fit |")')
    w('md.append("|----------|-------------|-----------|---------|---------|")')
    w("for pk, cft in curve_fit_types.items():")
    w("    if pk not in exp_data or exp_data[pk] is None:")
    w("        continue")
    w("    _, formula, param_names = CURVE_FUNCTIONS[cft]")
    w("    entry = curve_fit_results.get(pk, {})")
    w("    for i, pn in enumerate(param_names):")
    w('        md.append(f"| {PROTOCOL_LABELS[pk]} | {formula} | {pn} "')
    w("                  f\"| {_fmt(entry, 'exp', i)} | {_fmt(entry, 'sim', i)} |\")")
    w()
    w('_md_path = OUT_DIR / "report.md"')
    w('_md_path.write_text("\\n".join(md))')
    w('print(f"Saved: {_md_path}")')
    w('print(f"\\nAll done. Output in: {OUT_DIR.resolve()}")')

    return "\n".join(L)
