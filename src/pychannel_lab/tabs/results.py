"""Tab 6 — Results and comparison."""

import json
import math
from datetime import datetime

import streamlit as st

from core.curve_fitter import compute_aic_bic
from helpers import _simulate_all, _comparison_figure, _curve_fit_comparison_table
import pandas as pd


def render() -> None:
    ss = st.session_state

    if ss.fitted_params is None:
        st.info("Run the optimisation first to see results here.")
        return

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
            p.name: float(v) for p, v in zip(ss.msm_def.parameters, ss.fitted_params)
        },
        "model": ss.msm_def.to_dict(),
    }

    # ── Simulate once, reuse for figure + AIC/BIC + comparison table ──────────
    with st.spinner("Simulating fitted model…"):
        sim_data = _simulate_all(ss.fitted_params)

    # ── AIC / BIC ──────────────────────────────────────────────────────────────
    if ss.exp_data:
        st.subheader("Model information criteria (AIC / BIC)")
        ab = compute_aic_bic(ss.fitted_params, ss.exp_data, sim_data)
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

    st.divider()

    # ── Experimental vs Simulated figure ──────────────────────────────────────
    st.subheader("Experimental vs Simulated (fitted parameters)")
    st.plotly_chart(
        _comparison_figure(ss.fitted_params, _sim_data=sim_data), width="stretch"
    )

    # ── Phenomenological curve-fit comparison ──────────────────────────────────
    if ss.exp_data:
        st.subheader("Phenomenological curve-fit comparison")
        st.caption(
            "Fits of the selected function (chosen in the Data tab) to "
            "experimental data and to the simulated model output."
        )
        with st.spinner("Fitting curves…"):
            cmp_df = _curve_fit_comparison_table(sim_data)
        if not cmp_df.empty:
            st.dataframe(cmp_df, hide_index=True, width="stretch")

    st.divider()

    # ── Downloads ──────────────────────────────────────────────────────────────
    dl1, dl2 = st.columns(2)
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
