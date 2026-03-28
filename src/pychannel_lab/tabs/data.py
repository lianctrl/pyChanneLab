"""Tab 3 — Experimental Data upload and curve fitting."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import load_from_bytes
from core.curve_fitter import (
    fit_curve,
    eval_curve,
    CURVE_FUNCTIONS,
    CURVE_LABELS,
    PROTOCOL_CURVE_DEFAULTS,
)
from helpers import PROTOCOL_KEYS, PROTOCOL_LABELS, PROTOCOL_X_LABELS, PROTOCOL_Y_LABELS


def render() -> None:
    ss = st.session_state

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
                    ss.curve_fit_params_exp.pop(pk, None)
                    st.success(f"Loaded {len(x)} points.")
                except Exception as exc:
                    st.error(f"Failed to load: {exc}")

            if pk in ss.exp_data:
                x, y, y_err = ss.exp_data[pk]

                cft_default = ss.curve_fit_types.get(pk, PROTOCOL_CURVE_DEFAULTS[pk])
                cft_keys = list(CURVE_LABELS.keys())
                _cft_col, _fix_col = st.columns([3, 2])
                cft = _cft_col.selectbox(
                    "Fit function",
                    cft_keys,
                    index=cft_keys.index(cft_default),
                    format_func=lambda k: CURVE_LABELS[k],
                    key=f"cft_{pk}",
                )
                ss.curve_fit_types[pk] = cft
                fix_base = _fix_col.checkbox(
                    "Fix baseline = 0",
                    value=ss.curve_fix_baseline.get(pk, False),
                    key=f"fix_base_{pk}",
                )
                fix_amp = _fix_col.checkbox(
                    "Fix amplitude = 1",
                    value=ss.curve_fix_amplitude.get(pk, False),
                    key=f"fix_amp_{pk}",
                )
                ss.curve_fix_baseline[pk] = fix_base
                ss.curve_fix_amplitude[pk] = fix_amp

                _cached = ss.curve_fit_params_exp.get(pk)
                if (
                    _cached is None
                    or _cached[3] != cft
                    or _cached[4] != fix_base
                    or _cached[5] != fix_amp
                ):
                    popt, perr, ok = fit_curve(
                        x, y, cft, fix_baseline=fix_base, fix_amplitude=fix_amp
                    )
                    ss.curve_fit_params_exp[pk] = (
                        popt,
                        perr,
                        ok,
                        cft,
                        fix_base,
                        fix_amp,
                    )
                else:
                    popt, perr, ok = _cached[0], _cached[1], _cached[2]

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

                if ok:
                    _, formula, param_names = CURVE_FUNCTIONS[cft]
                    st.caption(f"Fit: {formula}")
                    fit_rows = []
                    for i, pn in enumerate(param_names):
                        val = popt[i] if not np.isnan(popt[i]) else None
                        err = perr[i] if not np.isnan(perr[i]) else None
                        fit_rows.append(
                            {
                                "Parameter": pn,
                                "Value": f"{val:.4g}" if val is not None else "—",
                                "± 1σ": f"{err:.4g}" if err is not None else "—",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(fit_rows), hide_index=True, width="stretch"
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
