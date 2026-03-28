"""Tab 4 — Preview / Sanity Check."""
import pandas as pd
import streamlit as st

from helpers import _simulate_all, _comparison_figure, _curve_fit_comparison_table


def render() -> None:
    ss = st.session_state

    model_errors = ss.msm_def.validate()
    if model_errors:
        st.error("Fix model errors in the MSM Builder tab first.")
        for e in model_errors:
            st.write(e)
        return

    init_params = ss.msm_def.initial_guess
    st.subheader("Initial parameters")
    _rows_init = []
    for pspec in ss.msm_def.parameters:
        _rows_init.append({
            "Parameter": pspec.name,
            "Value":     round(float(pspec.initial_value), 6),
            "Bounds":    f"[{pspec.lower_bound}, {pspec.upper_bound}]",
        })
    st.dataframe(pd.DataFrame(_rows_init), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Simulated curves with initial parameters")
    if not ss.exp_data:
        st.info("Load experimental data in the Data tab to overlay it here.")

    if st.button("▶ Run preview simulation", key="btn_preview_sim"):
        with st.spinner("Simulating with initial parameters…"):
            ss["_preview_sim"]        = _simulate_all(init_params)
            ss["_preview_sim_params"] = init_params.tolist()

    _prev_sim = ss.get("_preview_sim")
    if _prev_sim is not None:
        st.plotly_chart(_comparison_figure(init_params, _sim_data=_prev_sim), width="stretch")

        st.divider()
        st.subheader("Phenomenological curve-fit parameters (initial)")
        st.caption(
            "Fits of the selected function (chosen in the Data tab) to "
            "experimental data and to the simulated model output with initial parameters."
        )
        with st.spinner("Fitting curves…"):
            _prev_cmp_df = _curve_fit_comparison_table(_prev_sim)
        if not _prev_cmp_df.empty:
            st.dataframe(_prev_cmp_df, hide_index=True, width="stretch")
        else:
            st.info("Load experimental data in the Data tab to compare fits here.")
    else:
        st.info("Click the button above to run the simulation.")
