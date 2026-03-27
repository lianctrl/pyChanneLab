"""
pyChanneLab — Streamlit GUI for ion-channel MSM fitting.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import io
import json
import math
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.config import (
    TIME_PARAMS,
    G_K_MAX,
    TEMPERATURE,
    ActivationConfig,
    InactivationConfig,
    CSInactivationConfig,
    RecoveryConfig,
)
from core.msm_builder import (
    MSMDefinition,
    StateSpec,
    TransitionSpec,
    ParamSpec,
    compute_layout,
    PRESETS,
)
from core.simulator import ProtocolSimulator
from core.optimizer import CostFunction, ParameterOptimizer
from core.data_loader import load_from_bytes
from core.curve_fitter import (
    fit_curve,
    eval_curve,
    compute_aic_bic,
    CURVE_FUNCTIONS,
    CURVE_LABELS,
    PROTOCOL_CURVE_DEFAULTS,
)

# PyTorch back-end (optional — graceful fallback if not installed)
try:
    import torch
    from core.torch_simulator import get_device, preferred_dtype
    from core.torch_optimizer import TorchCostFunction, TorchParameterOptimizer

    _TORCH_AVAILABLE = True
    _torch_device = get_device()
    _torch_dtype = preferred_dtype(_torch_device)
except ImportError:
    _TORCH_AVAILABLE = False
    _torch_device = None
    _torch_dtype = None

st.set_page_config(page_title="pyChanneLab", page_icon="🔬", layout="wide")

st.title("🔬 pyChanneLab — Ion Channel MSM Fitting")
st.caption("Build your own Markov State Model, configure protocols, upload data, fit.")


def _init_state():
    if "msm_def" not in st.session_state:
        st.session_state.msm_def = PRESETS["11-state Kv channel (C0-C4, I0-I4, O)"]()

    _DEFAULTS = {
        "initial_conditions": None,  # None → derived from msm_def
        "g_k_max": G_K_MAX,
        "temperature": TEMPERATURE,
        "t_total": TIME_PARAMS["tend"],
        "dt": TIME_PARAMS["dt"],
        "act_cfg": ActivationConfig(),
        "inact_cfg": InactivationConfig(),
        "csi_cfg": CSInactivationConfig(),
        "rec_cfg": RecoveryConfig(),
        "exp_data": {},
        "opt_log": [],
        "opt_result": None,
        "fitted_params": None,
        "opt_costs_initial": None,
        "opt_costs_final": None,
        "curve_fit_types": {},      # {pk: curve_type_key}
        "curve_fit_params_exp": {},  # {pk: (popt, perr, ok, cft)}
        "curve_fit_params_sim": {},  # {pk: (popt, perr, ok, cft)}
        "opt_weights": {
            "activation": 1.0,
            "inactivation": 1.0,
            "cs_inactivation": 2.0,
            "recovery": 2.0,
        },
    }
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = deepcopy(v)


_init_state()
ss = st.session_state


def _ic() -> np.ndarray:
    """Return current initial conditions (from session or msm_def default)."""
    if (
        ss.initial_conditions is not None
        and len(ss.initial_conditions) == ss.msm_def.n_states
    ):
        return ss.initial_conditions
    return ss.msm_def.default_initial_conditions


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


def _build_sim(params: np.ndarray) -> ProtocolSimulator:
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
        solver=ss.get("solver", "ode"),
    )


def _build_cost() -> CostFunction:
    return CostFunction(
        ss.exp_data,
        weights=ss.opt_weights,
        msm_def=ss.msm_def,
        act_cfg=ss.act_cfg,
        inact_cfg=ss.inact_cfg,
        csi_cfg=ss.csi_cfg,
        rec_cfg=ss.rec_cfg,
        g_k_max=ss.g_k_max,
        t_total=ss.t_total,
        dt=ss.dt,
        solver=ss.get("solver", "ode"),
    )


def _simulate_all(params: np.ndarray) -> dict:
    sim = _build_sim(params)
    csi_times_s = sim.csi_proto.get_test_times()
    rec_times_s = sim.rec_proto.get_test_times()
    # Convert simulator absolute-seconds → ms durations/intervals for display
    csi_x_ms = (csi_times_s - ss.csi_cfg.t_initial) * 1000.0
    rec_x_ms = (rec_times_s - ss.rec_cfg.t_pulse) * 1000.0
    return {
        "activation": (sim.act_proto.get_test_voltages(), sim.run_activation()),
        "inactivation": (sim.inact_proto.get_test_voltages(), sim.run_inactivation()),
        "cs_inactivation": (csi_x_ms, sim.run_cs_inactivation()),
        "recovery": (rec_x_ms, sim.run_recovery()),
    }


TYPE_COLORS = {
    "closed": "#4878d0",  # blue
    "inactivated": "#ee854a",  # orange
    "open": "#6acc65",  # green
}
TYPE_LABELS = {"closed": "Closed", "inactivated": "Inactivated", "open": "Open"}


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

        # Offset bidirectional edges so they don't overlap
        is_bi = (tr.to_state, tr.from_state) in edge_pairs
        if is_bi:
            dx, dy = x1 - x0, y1 - y0
            d = max(math.hypot(dx, dy), 1e-9)
            px, py = -dy / d * 0.1, dx / d * 0.1
        else:
            px, py = 0.0, 0.0

        # Shorten arrow so it doesn't overlap the node circle
        shrink = 0.18
        ratio = 1.0 - shrink / max(math.hypot(x1 - x0, y1 - y0), 1e-9)
        ax_ = x0 + px
        ay_ = y0 + py
        x_ = x0 + (x1 - x0) * ratio + px
        y_ = y0 + (y1 - y0) * ratio + py

        # Truncate long expressions for the label
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
        group_states = [s for s in definition.states if s.state_type == stype]
        if not group_states:
            continue
        xs = [layout[s.name][0] for s in group_states if s.name in layout]
        ys = [layout[s.name][1] for s in group_states if s.name in layout]
        names = [s.name for s in group_states if s.name in layout]

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
                    size=42,
                    color=TYPE_COLORS[stype],
                    line=dict(width=2, color="white"),
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
    dt = 0.001
    t = np.arange(0, ss.t_total + dt, dt)
    fig = go.Figure()

    from core.protocols import (
        ActivationProtocol,
        InactivationProtocol,
        CSInactivationProtocol,
        RecoveryProtocol,
    )

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
    """Build a DataFrame comparing exp vs sim curve-fit parameters."""
    rows = []
    for pk in PROTOCOL_KEYS:
        if pk not in ss.exp_data or pk not in sim_data:
            continue
        cft = ss.curve_fit_types.get(pk, PROTOCOL_CURVE_DEFAULTS[pk])
        _, formula, param_names = CURVE_FUNCTIONS[cft]

        # Exp fit (already cached)
        exp_entry = ss.curve_fit_params_exp.get(pk)
        if exp_entry is not None:
            popt_exp, perr_exp, ok_exp, _ = exp_entry
        else:
            x_e, y_e, _ = ss.exp_data[pk]
            popt_exp, perr_exp, ok_exp = fit_curve(x_e, y_e, cft)

        # Sim fit (compute now)
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

            rows.append({
                "Protocol": PROTOCOL_LABELS[pk],
                "Fit function": formula,
                "Parameter": pn,
                "Exp fit": _fmt(ok_exp, popt_exp, perr_exp, i),
                "Sim fit": _fmt(ok_sim, popt_sim, perr_sim, i),
            })
    return pd.DataFrame(rows)


def _generate_run_script() -> str:
    """Return a self-contained Python script that reproduces the current optimisation."""
    import dataclasses

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def cfg_kwargs(cfg) -> str:
        return ", ".join(
            f"{f.name}={getattr(cfg, f.name)!r}" for f in dataclasses.fields(cfg)
        )

    # ── Serialise experimental data as numpy literals ──────────────────────
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

    # ── header ────────────────────────────────────────────────────────────
    w('#!/usr/bin/env python3')
    w(f'"""')
    w(f'pyChanneLab auto-generated run script.')
    w(f'Generated: {ts}')
    w()
    w('Steps executed:')
    w('  1. Global optimisation (Differential Evolution)')
    w('  2. Local refinement (L-BFGS-B)')
    w('  3. Simulate fitted model')
    w('  4. Compute AIC / BIC')
    w('  5. Fit phenomenological curves to exp and sim data')
    w('  6. Save comparison plots (plotly HTML + matplotlib PNG if available)')
    w('  7. Write Markdown report')
    w('"""')
    w()
    w('import sys, json')
    w('from pathlib import Path')
    w('import numpy as np')
    w()
    w('# ── path setup ──────────────────────────────────────────────────────────')
    w('# Place this script alongside app.py (inside pyChanneLab/) before running.')
    w('_HERE = Path(__file__).resolve().parent')
    w('sys.path.insert(0, str(_HERE))')
    w()
    w('from core.config import ActivationConfig, InactivationConfig, CSInactivationConfig, RecoveryConfig')
    w('from core.msm_builder import MSMDefinition')
    w('from core.simulator import ProtocolSimulator')
    w('from core.optimizer import CostFunction, ParameterOptimizer')
    w('from core.curve_fitter import fit_curve, eval_curve, compute_aic_bic, CURVE_LABELS, CURVE_FUNCTIONS')
    w()
    w('OUT_DIR = Path("pychannelab_output")')
    w('OUT_DIR.mkdir(exist_ok=True)')
    w()

    # ── MSM ───────────────────────────────────────────────────────────────
    msm_json_str = ss.msm_def.to_json()
    w('# ── MSM definition (embedded) ──────────────────────────────────────────')
    w(f'_MSM_JSON = {repr(msm_json_str)}')
    w('msm_def = MSMDefinition.from_json(_MSM_JSON)')
    w()

    # ── Protocol configs ──────────────────────────────────────────────────
    w('# ── protocol configs ───────────────────────────────────────────────────')
    w(f'act_cfg   = ActivationConfig({cfg_kwargs(ss.act_cfg)})')
    w(f'inact_cfg = InactivationConfig({cfg_kwargs(ss.inact_cfg)})')
    w(f'csi_cfg   = CSInactivationConfig({cfg_kwargs(ss.csi_cfg)})')
    w(f'rec_cfg   = RecoveryConfig({cfg_kwargs(ss.rec_cfg)})')
    w(f'g_k_max = {ss.g_k_max!r}')
    w(f't_total = {ss.t_total!r}')
    w(f'dt      = {ss.dt!r}')
    w(f'initial_state = np.array({ic_arr.tolist()})')
    w()

    # ── Experimental data ─────────────────────────────────────────────────
    w('# ── experimental data (embedded) ──────────────────────────────────────')
    w('exp_data = {')
    for line in exp_lines:
        w(line)
    w('}')
    w()

    # ── Weights ───────────────────────────────────────────────────────────
    w('# ── optimisation weights ───────────────────────────────────────────────')
    w('weights = {')
    for pk, wt in ss.opt_weights.items():
        w(f'    "{pk}": {wt},')
    w('}')
    w()

    # ── Curve-fit types ───────────────────────────────────────────────────
    w('# ── curve-fit function per protocol ────────────────────────────────────')
    w('curve_fit_types = {')
    for pk in PROTOCOL_KEYS:
        cft = ss.curve_fit_types.get(pk, PROTOCOL_CURVE_DEFAULTS[pk])
        w(f'    "{pk}": "{cft}",')
    w('}')
    w()

    # ── Display labels (needed in plots / report) ─────────────────────────
    w('PROTOCOL_KEYS = ["activation", "inactivation", "cs_inactivation", "recovery"]')
    w('PROTOCOL_LABELS = {')
    for pk, lbl in PROTOCOL_LABELS.items():
        w(f'    "{pk}": "{lbl}",')
    w('}')
    w('PROTOCOL_X_LABELS = {')
    for pk, lbl in PROTOCOL_X_LABELS.items():
        w(f'    "{pk}": "{lbl}",')
    w('}')
    w('PROTOCOL_Y_LABELS = {')
    for pk, lbl in PROTOCOL_Y_LABELS.items():
        w(f'    "{pk}": "{lbl}",')
    w('}')
    w()

    # ── Step 1: optimisation ──────────────────────────────────────────────
    w('# ' + '─' * 72)
    w('# STEP 1  Optimisation')
    w('# ' + '─' * 72)
    w('cost_fn = CostFunction(')
    w('    exp_data,')
    w('    weights=weights,')
    w('    msm_def=msm_def,')
    w('    act_cfg=act_cfg, inact_cfg=inact_cfg, csi_cfg=csi_cfg, rec_cfg=rec_cfg,')
    w('    g_k_max=g_k_max, t_total=t_total, dt=dt,')
    w(')')
    w('optimizer = ParameterOptimizer(cost_fn)')
    w('bounds = list(msm_def.bounds)')
    w()
    w('print("Running global optimisation (Differential Evolution)…")')
    w('result = optimizer.optimize_global(bounds=bounds, maxiter=5000)')
    w('print(f"  cost = {result.fun:.6f}")')
    w()
    w('print("Running local refinement (L-BFGS-B)…")')
    w('result = optimizer.optimize_local(initial_guess=result.x, bounds=bounds)')
    w('print(f"  cost = {result.fun:.6f}")')
    w()
    w('fitted_params = result.x')
    w('print("\\nFitted parameters:")')
    w('for pspec, val in zip(msm_def.parameters, fitted_params):')
    w('    print(f"  {pspec.name:20s} = {val:.6g}")')
    w()

    # ── Step 2: simulate ─────────────────────────────────────────────────
    w('# ' + '─' * 72)
    w('# STEP 2  Simulate with fitted parameters')
    w('# ' + '─' * 72)
    w('sim = ProtocolSimulator(')
    w('    fitted_params,')
    w('    msm_def=msm_def,')
    w('    act_cfg=act_cfg, inact_cfg=inact_cfg, csi_cfg=csi_cfg, rec_cfg=rec_cfg,')
    w('    t_total=t_total, dt=dt, g_k_max=g_k_max, initial_state=initial_state,')
    w(')')
    w('csi_x_ms = (sim.csi_proto.get_test_times() - csi_cfg.t_initial) * 1000.0')
    w('rec_x_ms = (sim.rec_proto.get_test_times() - rec_cfg.t_pulse)   * 1000.0')
    w('sim_data = {')
    w('    "activation":      (sim.act_proto.get_test_voltages(),   sim.run_activation()),')
    w('    "inactivation":    (sim.inact_proto.get_test_voltages(), sim.run_inactivation()),')
    w('    "cs_inactivation": (csi_x_ms, sim.run_cs_inactivation()),')
    w('    "recovery":        (rec_x_ms, sim.run_recovery()),')
    w('}')
    w()

    # ── Step 3: AIC/BIC ───────────────────────────────────────────────────
    w('# ' + '─' * 72)
    w('# STEP 3  AIC / BIC')
    w('# ' + '─' * 72)
    w('ab = compute_aic_bic(fitted_params, exp_data, sim_data)')
    w("print(f\"\\nAIC = {ab['AIC']:.3f}  |  BIC = {ab['BIC']:.3f}\")")
    w("print(f\"  n={int(ab['n_points'])}, k={int(ab['k_params'])}, RSS={ab['RSS']:.4g}\")")
    w()

    # ── Step 4: curve fits ────────────────────────────────────────────────
    w('# ' + '─' * 72)
    w('# STEP 4  Phenomenological curve fits')
    w('# ' + '─' * 72)
    w('curve_fit_results = {}')
    w('for pk, cft in curve_fit_types.items():')
    w('    entry = {}')
    w('    if pk in exp_data and exp_data[pk] is not None:')
    w('        x_e, y_e, _ = exp_data[pk]')
    w('        entry["exp"] = fit_curve(x_e, y_e, cft)')
    w('    if pk in sim_data:')
    w('        x_s, y_s = sim_data[pk]')
    w('        entry["sim"] = fit_curve(x_s, y_s, cft)')
    w('    curve_fit_results[pk] = entry')
    w()

    # ── Step 5: plots ─────────────────────────────────────────────────────
    w('# ' + '─' * 72)
    w('# STEP 5  Save comparison plots')
    w('# ' + '─' * 72)
    w('try:')
    w('    import plotly.graph_objects as go')
    w('    from plotly.subplots import make_subplots')
    w('    fig = make_subplots(rows=2, cols=2,')
    w('                        subplot_titles=list(PROTOCOL_LABELS.values()),')
    w('                        vertical_spacing=0.18, horizontal_spacing=0.12)')
    w('    for (row, col), pk in zip([(1,1),(1,2),(2,1),(2,2)], PROTOCOL_KEYS):')
    w('        x_s, y_s = sim_data[pk]')
    w('        first = (row == 1 and col == 1)')
    w('        if pk in exp_data and exp_data[pk] is not None:')
    w('            x_e, y_e, y_err = exp_data[pk]')
    w('            ekw = dict(error_y=dict(type="data", array=(y_err.tolist() if y_err is not None else None), visible=True)) if y_err is not None else {}')
    w('            fig.add_trace(go.Scatter(x=x_e.tolist(), y=y_e.tolist(), mode="markers",')
    w('                                     name="Exp", marker=dict(color="#1f77b4", size=8),')
    w('                                     showlegend=first, **ekw), row=row, col=col)')
    w('        fig.add_trace(go.Scatter(x=x_s.tolist(), y=y_s.tolist(), mode="lines",')
    w('                                 name="Sim", line=dict(color="#d62728", width=2),')
    w('                                 showlegend=first), row=row, col=col)')
    w('        entry = curve_fit_results.get(pk, {})')
    w('        popt_s, _, ok_s = entry.get("sim", (None, None, False))')
    w('        if ok_s and popt_s is not None and not np.any(np.isnan(popt_s)):')
    w('            x_fit = np.linspace(float(x_s.min()), float(x_s.max()), 300)')
    w('            y_fit = eval_curve(x_fit, popt_s, curve_fit_types[pk])')
    w('            fig.add_trace(go.Scatter(x=x_fit.tolist(), y=y_fit.tolist(), mode="lines",')
    w('                                     name="Fit (sim)", line=dict(color="#2ca02c", width=1.5, dash="dot"),')
    w('                                     showlegend=first), row=row, col=col)')
    w('        fig.update_xaxes(title_text=PROTOCOL_X_LABELS[pk], row=row, col=col)')
    w('        fig.update_yaxes(title_text=PROTOCOL_Y_LABELS[pk], row=row, col=col)')
    w('    fig.update_layout(height=700, title_text="Experimental vs Simulated")')
    w('    _html = OUT_DIR / "comparison.html"')
    w('    fig.write_html(str(_html))')
    w('    print(f"Saved: {_html}")')
    w('except Exception as _e:')
    w('    print(f"Plotly plot skipped: {_e}")')
    w()
    w('try:')
    w('    import matplotlib.pyplot as plt')
    w('    fig_mpl, axes = plt.subplots(2, 2, figsize=(12, 9))')
    w('    for ax, pk in zip(axes.flat, PROTOCOL_KEYS):')
    w('        x_s, y_s = sim_data[pk]')
    w('        ax.plot(x_s, y_s, color="#d62728", linewidth=2, label="Sim")')
    w('        if pk in exp_data and exp_data[pk] is not None:')
    w('            x_e, y_e, y_err = exp_data[pk]')
    w('            if y_err is not None:')
    w('                ax.errorbar(x_e, y_e, yerr=y_err, fmt="o",')
    w('                            color="#1f77b4", markersize=5, label="Exp")')
    w('            else:')
    w('                ax.scatter(x_e, y_e, color="#1f77b4", s=25, label="Exp")')
    w('            entry = curve_fit_results.get(pk, {})')
    w('            popt_e, _, ok_e = entry.get("exp", (None, None, False))')
    w('            if ok_e and popt_e is not None and not np.any(np.isnan(popt_e)):')
    w('                x_fit = np.linspace(float(x_e.min()), float(x_e.max()), 300)')
    w('                y_fit = eval_curve(x_fit, popt_e, curve_fit_types[pk])')
    w('                ax.plot(x_fit, y_fit, "--", color="#ff7f0e",')
    w('                        linewidth=1.5, label="Fit (exp)")')
    w('        ax.set_xlabel(PROTOCOL_X_LABELS[pk])')
    w('        ax.set_ylabel(PROTOCOL_Y_LABELS[pk])')
    w('        ax.set_title(PROTOCOL_LABELS[pk])')
    w('        ax.legend(fontsize=8)')
    w('    plt.tight_layout()')
    w('    _png = OUT_DIR / "comparison.png"')
    w('    fig_mpl.savefig(str(_png), dpi=150)')
    w('    plt.close(fig_mpl)')
    w('    print(f"Saved: {_png}")')
    w('except ImportError:')
    w('    print("matplotlib not available — PNG skipped")')
    w('except Exception as _e:')
    w('    print(f"matplotlib plot skipped: {_e}")')
    w()

    # ── Step 6: Markdown report ───────────────────────────────────────────
    w('# ' + '─' * 72)
    w('# STEP 6  Markdown report')
    w('# ' + '─' * 72)
    w('def _fmt(entry_dict, key, idx):')
    w('    e = entry_dict.get(key)')
    w('    if e is None: return "—"')
    w('    popt, perr, ok = e')
    w('    if not ok or np.isnan(popt[idx]): return "—"')
    w('    return f"{popt[idx]:.4g}" if np.isnan(perr[idx]) else f"{popt[idx]:.4g} ± {perr[idx]:.4g}"')
    w()
    w(f'_ts = "{ts}"')
    w('_n_s = msm_def.n_states')
    w('_n_t = len(msm_def.transitions)')
    w('_n_p = len(msm_def.parameters)')
    w()
    w('md = []')
    w('md.append("# pyChanneLab Optimisation Report")')
    w('md.append("")')
    w('md.append(f"**Generated:** {_ts}  ")')
    w('md.append(f"**Model:** {_n_s} states · {_n_t} transitions · {_n_p} free parameters")')
    w('md.append("")')
    w('md.append("---")')
    w('md.append("")')
    w('md.append("## 1. Optimised parameters")')
    w('md.append("")')
    w('md.append("| Parameter | Fitted value | Bounds |")')
    w('md.append("|-----------|-------------|--------|")')
    w('for pspec, val in zip(msm_def.parameters, fitted_params):')
    w('    md.append(f"| {pspec.name} | {val:.6g} | [{pspec.lower_bound}, {pspec.upper_bound}] |")')
    w('md.append("")')
    w('md.append("---")')
    w('md.append("")')
    w('md.append("## 2. Model information criteria (AIC / BIC)")')
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
    w('md.append("## 3. Phenomenological curve-fit comparison")')
    w('md.append("")')
    w('md.append("| Protocol | Fit function | Parameter | Exp fit | Sim fit |")')
    w('md.append("|----------|-------------|-----------|---------|---------|")')
    w('for pk, cft in curve_fit_types.items():')
    w('    if pk not in exp_data or exp_data[pk] is None:')
    w('        continue')
    w('    _, formula, param_names = CURVE_FUNCTIONS[cft]')
    w('    entry = curve_fit_results.get(pk, {})')
    w('    for i, pn in enumerate(param_names):')
    w('        md.append(f"| {PROTOCOL_LABELS[pk]} | {formula} | {pn} "')
    w('                  f"| {_fmt(entry, \'exp\', i)} | {_fmt(entry, \'sim\', i)} |")')
    w()
    w('_md_path = OUT_DIR / "report.md"')
    w('_md_path.write_text("\\n".join(md))')
    w('print(f"Saved: {_md_path}")')
    w('print(f"\\nAll done. Output in: {OUT_DIR.resolve()}")')

    return "\n".join(L)


# TABS
tab_builder, tab_proto, tab_data, tab_opt, tab_results = st.tabs(
    [
        "🏗️ MSM Builder",
        "🧪 Protocols",
        "📁 Data",
        "🚀 Optimise",
        "📊 Results",
    ]
)

# TAB 1 — MSM Builder
with tab_builder:
    col_preset, col_import, col_export, col_apply = st.columns([2, 1, 1, 1])

    preset_name = col_preset.selectbox(
        "Load preset",
        ["— keep current —"] + list(PRESETS.keys()),
        key="preset_select",
    )
    if col_preset.button("Load preset", key="load_preset"):
        if preset_name in PRESETS:
            ss.msm_def = PRESETS[preset_name]()
            ss.initial_conditions = None
            ss.fitted_params = None
            st.success(f"Loaded preset: {preset_name}")
            st.rerun()

    uploaded_json = col_import.file_uploader(
        "Import JSON", type=["json"], key="import_json", label_visibility="collapsed"
    )
    if uploaded_json is not None:
        try:
            ss.msm_def = MSMDefinition.from_json(uploaded_json.read().decode())
            ss.initial_conditions = None
            ss.fitted_params = None
            st.success("Model imported from JSON.")
            st.rerun()
        except Exception as exc:
            st.error(f"Import failed: {exc}")

    if col_export.download_button(
        "Export JSON",
        data=ss.msm_def.to_json(),
        file_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    ):
        pass

    st.divider()

    with st.expander("Physical & simulation settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        ss.temperature = c1.number_input(
            "Temperature (K)",
            value=float(ss.temperature),
            min_value=200.0,
            max_value=400.0,
            step=1.0,
        )
        ss.g_k_max = c2.number_input(
            "G_K_max (nS)",
            value=float(ss.g_k_max),
            min_value=0.1,
            max_value=500.0,
            step=0.1,
        )
        ss.t_total = c3.number_input(
            "Simulation time (s)",
            value=float(ss.t_total),
            min_value=0.5,
            max_value=20.0,
            step=0.5,
        )
        ss.dt = c4.number_input(
            "dt (s)", value=float(ss.dt), min_value=1e-6, max_value=1e-3, format="%.1e"
        )

    col_states, col_trans, col_params = st.columns([1, 2, 2])

    with col_states:
        st.subheader("States")
        st.caption("Add/remove rows. Type must be closed / inactivated / open.")
        states_df = pd.DataFrame(
            [{"name": s.name, "type": s.state_type} for s in ss.msm_def.states]
        )
        edited_states = st.data_editor(
            states_df,
            width="stretch",
            num_rows="dynamic",
            column_config={
                "name": st.column_config.TextColumn("Name", required=True),
                "type": st.column_config.SelectboxColumn(
                    "Type", options=["closed", "inactivated", "open"], required=True
                ),
            },
            key="states_editor",
            height=400,
        )

    with col_trans:
        st.subheader("Transitions")
        current_names = edited_states["name"].dropna().tolist()
        st.caption(
            "Rate expression: Python with V, EXP_FACTOR, exp(), and param names."
        )
        trans_df = pd.DataFrame(
            [
                {"from": t.from_state, "to": t.to_state, "rate_expr": t.rate_expr}
                for t in ss.msm_def.transitions
            ]
        )
        edited_trans = st.data_editor(
            trans_df,
            width="stretch",
            num_rows="dynamic",
            column_config={
                "from": st.column_config.TextColumn("From", required=True),
                "to": st.column_config.TextColumn("To", required=True),
                "rate_expr": st.column_config.TextColumn(
                    "Rate expression", required=True
                ),
            },
            key="trans_editor",
            height=400,
        )

    with col_params:
        st.subheader("Parameters")
        st.caption("Values used as initial guess for optimisation.")
        params_df = pd.DataFrame(
            [
                {
                    "name": p.name,
                    "initial": p.initial_value,
                    "lower": p.lower_bound,
                    "upper": p.upper_bound,
                }
                for p in ss.msm_def.parameters
            ]
        )
        edited_params = st.data_editor(
            params_df,
            width="stretch",
            num_rows="dynamic",
            column_config={
                "name": st.column_config.TextColumn("Name", required=True),
                "initial": st.column_config.NumberColumn(
                    "Initial value", format="%.4g"
                ),
                "lower": st.column_config.NumberColumn("Lower bound", format="%.4g"),
                "upper": st.column_config.NumberColumn("Upper bound", format="%.4g"),
            },
            key="params_editor",
            height=400,
        )

    st.divider()
    apply_col, msg_col = st.columns([1, 4])

    if apply_col.button("Apply model", type="primary", key="apply_model"):
        try:
            new_states = [
                StateSpec(name=row["name"], state_type=row["type"])
                for _, row in edited_states.dropna(subset=["name"]).iterrows()
                if str(row["name"]).strip()
            ]
            new_trans = [
                TransitionSpec(
                    from_state=str(row["from"]),
                    to_state=str(row["to"]),
                    rate_expr=str(row["rate_expr"]),
                )
                for _, row in edited_trans.dropna(
                    subset=["from", "to", "rate_expr"]
                ).iterrows()
            ]
            new_params = [
                ParamSpec(
                    name=str(row["name"]),
                    initial_value=float(row["initial"]),
                    lower_bound=float(row["lower"]),
                    upper_bound=float(row["upper"]),
                )
                for _, row in edited_params.dropna(subset=["name"]).iterrows()
                if str(row["name"]).strip()
            ]

            candidate = MSMDefinition(
                states=new_states, transitions=new_trans, parameters=new_params
            )
            errors = candidate.validate()

            if errors:
                for e in errors:
                    msg_col.error(e)
            else:
                ss.msm_def = candidate
                ss.initial_conditions = None  # reset to default for new topology
                ss.fitted_params = None
                msg_col.success(
                    f"Model applied: {len(new_states)} states, "
                    f"{len(new_trans)} transitions, {len(new_params)} parameters."
                )

        except Exception as exc:
            msg_col.error(f"Failed to parse model: {exc}")

    with st.expander("Initial state distribution", expanded=False):
        st.caption(
            "Must sum to 1. Leave all zero to use the default (uniform over closed states)."
        )
        ic_vals = _ic().copy()
        ic_names = ss.msm_def.state_names
        ic_cols = st.columns(min(len(ic_names), 12))
        for i, (col, name) in enumerate(zip(ic_cols, ic_names)):
            ic_vals[i] = col.number_input(
                name,
                value=float(ic_vals[i]),
                min_value=0.0,
                max_value=1.0,
                step=0.001,
                format="%.4f",
                key=f"ic_{name}_{i}",
            )
        ic_sum = float(np.sum(ic_vals))
        if abs(ic_sum - 1.0) > 1e-3:
            st.warning(f"Sum = {ic_sum:.4f}  (should be 1.0)")
        else:
            st.success(f"Sum = {ic_sum:.4f}")
        ss.initial_conditions = ic_vals

    st.divider()
    st.subheader("Model topology")

    errors_now = ss.msm_def.validate()
    if errors_now:
        st.warning("Current model has validation issues:")
        for e in errors_now:
            st.error(e)
    else:
        st.plotly_chart(_network_figure(ss.msm_def), width="stretch")

        n_s = ss.msm_def.n_states
        n_t = len(ss.msm_def.transitions)
        n_p = len(ss.msm_def.parameters)
        st.caption(
            f"{n_s} states  •  {n_t} transitions  •  {n_p} free parameters  •  "
            f"open state(s): {[ss.msm_def.state_names[i] for i in ss.msm_def.open_state_indices]}"
        )

# TAB 2 — Protocol Setup
with tab_proto:
    st.info(
        "Adjust timing and voltages for each experimental protocol. "
        "The waveform preview updates instantly."
    )
    proto_subtabs = st.tabs([PROTOCOL_LABELS[k] for k in PROTOCOL_KEYS])

    with proto_subtabs[0]:
        c = ss.act_cfg
        col1, col2, col3 = st.columns(3)
        c.v_hold = col1.number_input(
            "V_hold (mV)", value=float(c.v_hold), step=5.0, key="a_vhold"
        )
        c.v_tail = col2.number_input(
            "V_tail (mV)", value=float(c.v_tail), step=5.0, key="a_vtail"
        )
        c.v_min = col1.number_input(
            "V_min (mV)", value=float(c.v_min), step=5.0, key="a_vmin"
        )
        c.v_max = col2.number_input(
            "V_max (mV)", value=float(c.v_max), step=5.0, key="a_vmax"
        )
        c.v_step = col3.number_input(
            "V_step (mV)", value=float(c.v_step), step=1.0, key="a_vstep", min_value=0.5
        )
        c.t_hold = col1.number_input(
            "t_hold (s)", value=float(c.t_hold), step=0.05, key="a_thold", min_value=0.0
        )
        c.t_test = col2.number_input(
            "t_test (s)",
            value=float(c.t_test),
            step=0.005,
            key="a_ttest",
            min_value=0.001,
        )
        st.plotly_chart(_voltage_preview("activation"), width="stretch")

    with proto_subtabs[1]:
        c = ss.inact_cfg
        col1, col2, col3 = st.columns(3)
        c.v_hold = col1.number_input(
            "V_hold (mV)", value=float(c.v_hold), step=5.0, key="i_vhold"
        )
        c.v_depo = col2.number_input(
            "V_depo (mV)", value=float(c.v_depo), step=5.0, key="i_vdepo"
        )
        c.v_min = col1.number_input(
            "V_cond min (mV)", value=float(c.v_min), step=5.0, key="i_vmin"
        )
        c.v_max = col2.number_input(
            "V_cond max (mV)", value=float(c.v_max), step=5.0, key="i_vmax"
        )
        c.v_step = col3.number_input(
            "V_step (mV)", value=float(c.v_step), step=1.0, key="i_vstep", min_value=0.5
        )
        c.t_hold = col1.number_input(
            "t_hold (s)", value=float(c.t_hold), step=0.05, key="i_thold", min_value=0.0
        )
        c.t_cond = col2.number_input(
            "t_cond (s)", value=float(c.t_cond), step=0.1, key="i_tcond", min_value=0.01
        )
        st.plotly_chart(_voltage_preview("inactivation"), width="stretch")

    with proto_subtabs[2]:
        c = ss.csi_cfg
        col1, col2, col3 = st.columns(3)
        c.v_hold = col1.number_input(
            "V_hold (mV)", value=float(c.v_hold), step=5.0, key="csi_vh"
        )
        c.v_prep = col2.number_input(
            "V_prep (mV)", value=float(c.v_prep), step=5.0, key="csi_vp"
        )
        c.v_depo = col3.number_input(
            "V_depo (mV)", value=float(c.v_depo), step=5.0, key="csi_vd"
        )
        c.t_initial = col1.number_input(
            "t_initial (s)",
            value=float(c.t_initial),
            step=0.01,
            key="csi_ti",
            min_value=0.0,
        )
        c.min_pulse = col2.number_input(
            "Min prepulse (s)",
            value=float(c.min_pulse),
            step=0.01,
            key="csi_mn",
            min_value=0.001,
        )
        c.max_pulse = col3.number_input(
            "Max prepulse (s)", value=float(c.max_pulse), step=0.01, key="csi_mx"
        )
        c.pulse_increment = col1.number_input(
            "Pulse step (s)",
            value=float(c.pulse_increment),
            step=0.005,
            key="csi_pi",
            min_value=0.001,
        )
        c.t_test_end = col2.number_input(
            "t_test_end (s)", value=float(c.t_test_end), step=0.05, key="csi_te"
        )
        st.plotly_chart(_voltage_preview("cs_inactivation"), width="stretch")

    with proto_subtabs[3]:
        c = ss.rec_cfg
        col1, col2, col3 = st.columns(3)
        c.v_hold = col1.number_input(
            "V_hold (mV)", value=float(c.v_hold), step=5.0, key="rec_vh"
        )
        c.v_depo = col2.number_input(
            "V_depo (mV)", value=float(c.v_depo), step=5.0, key="rec_vd"
        )
        c.t_prep = col1.number_input(
            "t_prep (s)", value=float(c.t_prep), step=0.05, key="rec_tp", min_value=0.0
        )
        c.t_pulse = col2.number_input(
            "t_pulse start (s)", value=float(c.t_pulse), step=0.05, key="rec_tps"
        )
        c.min_interval = col1.number_input(
            "Min interval (s)",
            value=float(c.min_interval),
            step=0.01,
            key="rec_mn",
            min_value=0.0,
        )
        c.max_interval = col2.number_input(
            "Max interval (s)", value=float(c.max_interval), step=0.01, key="rec_mx"
        )
        c.interval_increment = col3.number_input(
            "Interval step (s)",
            value=float(c.interval_increment),
            step=0.005,
            key="rec_ii",
            min_value=0.001,
        )
        c.t_end = col1.number_input(
            "t_end (s)", value=float(c.t_end), step=0.05, key="rec_te"
        )
        st.plotly_chart(_voltage_preview("recovery"), width="stretch")

# TAB 3 — Experimental Data
with tab_data:
    st.subheader("Upload experimental data")
    st.markdown(
        "Upload a CSV for each protocol you want to fit.  "
        "**Format**: header row, then columns `x`, `y` (and optionally `y_err`)."
    )

    for pk in PROTOCOL_KEYS:
        with st.expander(PROTOCOL_LABELS[pk], expanded=(pk == "activation")):
            uploaded = st.file_uploader(
                f"CSV — {PROTOCOL_LABELS[pk]}",
                type=["csv"],
                key=f"upload_{pk}",
            )
            if uploaded is not None:
                try:
                    x, y, y_err = load_from_bytes(uploaded.read())
                    ss.exp_data[pk] = (x, y, y_err)
                    st.success(f"Loaded {len(x)} points.")
                except Exception as exc:
                    st.error(f"Failed to load: {exc}")

            if pk in ss.exp_data:
                x, y, y_err = ss.exp_data[pk]

                # ── curve type selector ───────────────────────────────────
                cft_default = ss.curve_fit_types.get(pk, PROTOCOL_CURVE_DEFAULTS[pk])
                cft_keys = list(CURVE_LABELS.keys())
                cft = st.selectbox(
                    "Fit function",
                    cft_keys,
                    index=cft_keys.index(cft_default),
                    format_func=lambda k: CURVE_LABELS[k],
                    key=f"cft_{pk}",
                )
                ss.curve_fit_types[pk] = cft

                # compute fit and cache
                popt, perr, ok = fit_curve(x, y, cft)
                ss.curve_fit_params_exp[pk] = (popt, perr, ok, cft)

                # ── plot: data + fit curve ────────────────────────────────
                fig = go.Figure()
                err_kw = (
                    dict(error_y=dict(type="data", array=y_err, visible=True))
                    if y_err is not None
                    else {}
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        name="Data",
                        marker=dict(size=7, color="#1f77b4"),
                        **err_kw,
                    )
                )
                if ok and not np.any(np.isnan(popt)):
                    x_fit = np.linspace(float(x.min()), float(x.max()), 300)
                    y_fit = eval_curve(x_fit, popt, cft)
                    fig.add_trace(
                        go.Scatter(
                            x=x_fit,
                            y=y_fit,
                            mode="lines",
                            name=f"Fit ({CURVE_LABELS[cft]})",
                            line=dict(color="#ff7f0e", width=2, dash="dash"),
                        )
                    )
                fig.update_layout(
                    xaxis_title=PROTOCOL_X_LABELS[pk],
                    yaxis_title=PROTOCOL_Y_LABELS[pk],
                    height=280,
                    margin=dict(t=10, b=40, l=60, r=20),
                    legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(fig, width="stretch")

                # ── fit parameter table ───────────────────────────────────
                if ok:
                    _, formula, param_names = CURVE_FUNCTIONS[cft]
                    st.caption(f"Fit: {formula}")
                    fit_rows = []
                    for i, pn in enumerate(param_names):
                        val = popt[i] if not np.isnan(popt[i]) else None
                        err = perr[i] if not np.isnan(perr[i]) else None
                        fit_rows.append({
                            "Parameter": pn,
                            "Value": f"{val:.4g}" if val is not None else "—",
                            "± 1σ":  f"{err:.4g}" if err is not None else "—",
                        })
                    st.dataframe(
                        pd.DataFrame(fit_rows),
                        hide_index=True,
                        width='stretch',
                    )
                else:
                    st.warning("Curve fit did not converge.")

                if st.button(f"Remove {PROTOCOL_LABELS[pk]} data", key=f"rm_{pk}"):
                    del ss.exp_data[pk]
                    ss.curve_fit_params_exp.pop(pk, None)
                    st.rerun()

    if ss.exp_data:
        st.success(f"Loaded: {', '.join(ss.exp_data.keys())}")
    else:
        st.warning("No experimental data loaded yet.")

# TAB 4 — Optimisation
with tab_opt:
    model_errors = ss.msm_def.validate()
    if model_errors:
        st.error("Fix model errors in the MSM Builder tab first.")
        for e in model_errors:
            st.write(e)
    elif not ss.exp_data:
        st.warning("Load at least one experimental dataset in the Data tab first.")
    else:
        st.subheader("Optimisation settings")

        _SCIPY_METHODS = [
            "Global (Diff. Evolution)",
            "Local (L-BFGS-B)",
            "Global then Local",
        ]
        _TORCH_LABEL = "PyTorch (Adam → L-BFGS)"
        _method_options = _SCIPY_METHODS + ([_TORCH_LABEL] if _TORCH_AVAILABLE else [])

        col1, col2, col3 = st.columns(3)
        method = col1.selectbox("Method", _method_options, key="opt_method")

        use_torch = method == _TORCH_LABEL

        if not use_torch:
            col1.radio(
                "Solver",
                ["ode", "qmatrix"],
                horizontal=True,
                key="solver",
                help=(
                    "**ode** — scipy odeint (numerical integration). "
                    "**qmatrix** — matrix exponential P(t+dt)=expm(Q·dt)@P(t), "
                    "exact for piecewise-constant voltage within each dt step."
                ),
            )

        if use_torch:
            _dev_name = str(_torch_device) if _torch_device else "cpu"
            col1.caption(f"Device: **{_dev_name}**")
            maxiter = col2.number_input(
                "Adam steps",
                value=500,
                min_value=0,
                max_value=10000,
                step=50,
                key="opt_maxiter",
            )
            n_lbfgs = col3.number_input(
                "L-BFGS steps",
                value=200,
                min_value=0,
                max_value=2000,
                step=20,
                key="opt_lbfgs",
            )
            col_lr, col_peaks = st.columns(2)
            adam_lr = col_lr.number_input(
                "Adam learning rate",
                value=0.05,
                min_value=1e-4,
                max_value=1.0,
                format="%.4f",
                key="opt_adam_lr",
            )
            n_peak_steps = col_peaks.number_input(
                "Peak-detection substeps",
                value=50,
                min_value=10,
                max_value=200,
                step=10,
                key="opt_peak_steps",
            )
        else:
            maxiter = col2.number_input(
                "Max iterations",
                value=5000,
                min_value=10,
                max_value=50000,
                step=100,
                key="opt_maxiter",
            )
            workers = col3.number_input(
                "Workers (-1 = all cores)",
                value=-1,
                min_value=-1,
                max_value=64,
                step=1,
                key="opt_workers",
            )

        st.subheader("Protocol weights")
        wc = st.columns(4)
        ss.opt_weights["activation"] = wc[0].number_input(
            "Activation",
            value=float(ss.opt_weights["activation"]),
            min_value=0.0,
            step=0.5,
            key="w_act",
        )
        ss.opt_weights["inactivation"] = wc[1].number_input(
            "Inactivation",
            value=float(ss.opt_weights["inactivation"]),
            min_value=0.0,
            step=0.5,
            key="w_inact",
        )
        ss.opt_weights["cs_inactivation"] = wc[2].number_input(
            "CS Inact.",
            value=float(ss.opt_weights["cs_inactivation"]),
            min_value=0.0,
            step=0.5,
            key="w_csi",
        )
        ss.opt_weights["recovery"] = wc[3].number_input(
            "Recovery",
            value=float(ss.opt_weights["recovery"]),
            min_value=0.0,
            step=0.5,
            key="w_rec",
        )

        st.divider()

        if st.button("Run Optimisation", type="primary", key="run_opt"):
            defn = ss.msm_def
            bounds = list(defn.bounds)
            log: list = []

            if use_torch:
                _total_iters = int(maxiter) + int(n_lbfgs)
            elif "Global then Local" in method:
                _total_iters = int(maxiter) * 2  # rough estimate
            else:
                _total_iters = int(maxiter)

            _prog = st.progress(0.0, text="Starting…")
            _mc1, _mc2, _mc3 = st.columns(3)
            _m_iter = _mc1.empty()
            _m_cost = _mc2.empty()
            _m_phase = _mc3.empty()
            _chart = st.empty()

            _cost_hist = []
            _global_iter = [0]

            def _cb(iteration, cost, convergence):
                _global_iter[0] += 1
                g = _global_iter[0]

                msg = f"Iter {g:5d} | cost = {cost:.6e}"
                if convergence is not None:
                    msg += f" | conv = {convergence:.4f}"
                log.append(msg)

                # Phase label
                if use_torch:
                    phase = "Adam" if iteration <= int(maxiter) else "L-BFGS"
                elif convergence is not None:
                    phase = "Global (DE)"
                else:
                    phase = "Local (L-BFGS-B)"

                # Progress bar
                pct = min(g / max(_total_iters, 1), 1.0)
                _prog.progress(pct, text=f"{phase}  ·  iter {g} / {_total_iters}")

                # Metric cards
                _m_iter.metric("Iteration", f"{g:,}")
                _m_cost.metric("Cost", f"{cost:.4e}")
                _m_phase.metric("Phase", phase)

                _cost_hist.append(math.log10(max(cost, 1e-15)))
                if len(_cost_hist) == 1 or len(_cost_hist) % 5 == 0:
                    _chart.line_chart(
                        pd.DataFrame(
                            {"log10(cost)": _cost_hist},
                            index=range(1, len(_cost_hist) + 1),
                        ),
                        height=200,
                        x_label="callback #",
                        y_label="log10(cost)",
                    )

            result = None

            if use_torch:
                torch_cost = TorchCostFunction(
                    ss.exp_data,
                    weights=ss.opt_weights,
                    msm_def=defn,
                    act_cfg=ss.act_cfg,
                    inact_cfg=ss.inact_cfg,
                    csi_cfg=ss.csi_cfg,
                    rec_cfg=ss.rec_cfg,
                    g_k_max=ss.g_k_max,
                    t_total=ss.t_total,
                    n_peak_steps=int(n_peak_steps),
                    device=_torch_device,
                    dtype=_torch_dtype,
                )
                torch_opt = TorchParameterOptimizer(torch_cost)
                ss.opt_costs_initial = torch_opt.cost_breakdown(defn.initial_guess)

                result = torch_opt.optimize(
                    initial_guess=defn.initial_guess,
                    bounds=bounds,
                    n_adam=int(maxiter),
                    adam_lr=float(adam_lr),
                    n_lbfgs=int(n_lbfgs),
                    progress_callback=_cb,
                )

                ss.opt_costs_final = torch_opt.cost_breakdown(result.x)

            else:
                cost_fn = _build_cost()
                optimizer = ParameterOptimizer(cost_fn)
                ss.opt_costs_initial = optimizer.cost_breakdown(defn.initial_guess)

                if "Global" in method:
                    result = optimizer.optimize_global(
                        bounds=bounds,
                        maxiter=int(maxiter),
                        workers=int(workers),
                        progress_callback=_cb,
                    )
                    if "then Local" in method:
                        log.append("--- Switching to local refinement ---")
                        result = optimizer.optimize_local(
                            initial_guess=result.x,
                            bounds=bounds,
                            progress_callback=_cb,
                        )
                else:
                    result = optimizer.optimize_local(
                        initial_guess=defn.initial_guess,
                        bounds=bounds,
                        progress_callback=_cb,
                    )

                ss.opt_costs_final = optimizer.cost_breakdown(result.x)

            _prog.progress(1.0, text="Done")

            ss.fitted_params = result.x
            ss.opt_result = result
            ss.opt_log = log
            st.success(f"Done! Final cost: {result.fun:.6f}")

        if ss.opt_result is not None:
            ca, cb = st.columns(2)
            ca.metric("Final cost", f"{ss.opt_result.fun:.6f}")
            cb.metric("Iterations", str(ss.opt_result.get("nit", "—")))

            if ss.opt_costs_initial and ss.opt_costs_final:
                st.subheader("Cost breakdown")
                rows = []
                for pk in ss.opt_costs_final:
                    ci = ss.opt_costs_initial.get(pk, float("nan"))
                    cf = ss.opt_costs_final[pk]
                    imp = (ci - cf) / ci * 100 if ci else float("nan")
                    rows.append(
                        {
                            "Protocol": pk,
                            "Initial": f"{ci:.6f}",
                            "Final": f"{cf:.6f}",
                            "Improvement (%)": f"{imp:.1f}",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            if ss.opt_log:
                with st.expander("Optimisation log (last 200 iterations)"):
                    st.text("\n".join(ss.opt_log[-200:]))

# TAB 5 — Results and comparison
with tab_results:
    if ss.fitted_params is None and not ss.exp_data:
        st.info("Run the optimisation first, or load data to preview simulations.")
    else:
        params = (
            ss.fitted_params
            if ss.fitted_params is not None
            else ss.msm_def.initial_guess
        )
        label = "fitted" if ss.fitted_params is not None else "initial guess"

        if ss.fitted_params is not None:
            st.subheader("Fitted parameters")
            rows = []
            for i, pspec in enumerate(ss.msm_def.parameters):
                init_v = float(pspec.initial_value)
                final_v = float(ss.fitted_params[i])
                chg = (final_v - init_v) / init_v * 100 if init_v != 0 else float("nan")
                rows.append(
                    {
                        "Parameter": pspec.name,
                        "Initial": round(init_v, 6),
                        "Fitted": round(final_v, 6),
                        "Change (%)": round(chg, 2),
                        "Bounds": f"[{pspec.lower_bound}, {pspec.upper_bound}]",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dict = {
                "timestamp": ts,
                "final_cost": float(ss.opt_result.fun),
                "parameters": {
                    p.name: float(v)
                    for p, v in zip(ss.msm_def.parameters, ss.fitted_params)
                },
                "model": ss.msm_def.to_dict(),
            }

            # ── AIC / BIC ─────────────────────────────────────────────────
            if ss.exp_data:
                st.subheader("Model information criteria (AIC / BIC)")
                with st.spinner("Computing AIC/BIC…"):
                    _ab_sim = _simulate_all(ss.fitted_params)
                    ab = compute_aic_bic(ss.fitted_params, ss.exp_data, _ab_sim)
                _ca, _cb, _cc, _cd, _ce = st.columns(5)
                _ca.metric("AIC", f"{ab['AIC']:.2f}" if not math.isnan(ab["AIC"]) else "—")
                _cb.metric("BIC", f"{ab['BIC']:.2f}" if not math.isnan(ab["BIC"]) else "—")
                _cc.metric("RSS", f"{ab['RSS']:.4g}")
                _cd.metric("n points", int(ab["n_points"]))
                _ce.metric("k params", int(ab["k_params"]))
                st.caption(
                    "Lower AIC/BIC = better model relative to its complexity. "
                    "Use these scores to compare models with different numbers of states."
                )

            # ── downloads ─────────────────────────────────────────────────
            dl1, dl2, dl3 = st.columns(3)
            dl1.download_button(
                "Download parameters (JSON)",
                data=json.dumps(save_dict, indent=2),
                file_name=f"params_{ts}.json",
                mime="application/json",
            )
            dl2.download_button(
                "Download model definition (JSON)",
                data=ss.msm_def.to_json(),
                file_name=f"model_{ts}.json",
                mime="application/json",
            )
            dl3.download_button(
                "Download run script (.py)",
                data=_generate_run_script(),
                file_name=f"pychannelab_run_{ts}.py",
                mime="text/plain",
                help=(
                    "Self-contained Python script that reproduces this optimisation, "
                    "saves plots (HTML + PNG), and writes a Markdown report."
                ),
            )

        st.divider()

        # ── Simulate once, reuse for figure + comparison table ────────────
        with st.spinner("Simulating…"):
            sim_data = _simulate_all(params)

        st.subheader(f"Experimental vs Simulated  ({label} parameters)")
        fig = _comparison_figure(params, _sim_data=sim_data)
        st.plotly_chart(fig, width="stretch")

        # ── Phenomenological curve-fit comparison table ───────────────────
        if ss.exp_data:
            st.subheader("Phenomenological curve-fit comparison")
            st.caption(
                "Fits of the selected function (chosen in the Data tab) to "
                "experimental data and to the simulated model output."
            )
            with st.spinner("Fitting curves…"):
                cmp_df = _curve_fit_comparison_table(sim_data)
            if not cmp_df.empty:
                st.dataframe(cmp_df, hide_index=True, width='stretch')
            else:
                st.info("Load experimental data and select fit functions in the Data tab.")
