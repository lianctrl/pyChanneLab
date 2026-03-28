"""Tab 1 — MSM Builder."""
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from core.msm_builder import MSMDefinition, StateSpec, TransitionSpec, ParamSpec, PRESETS
from helpers import _ic, _network_figure


def render() -> None:
    ss = st.session_state

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
            "Temperature (K)", value=float(ss.temperature),
            min_value=200.0, max_value=400.0, step=1.0,
        )
        ss.g_k_max = c2.number_input(
            "G_K_max (nS)", value=float(ss.g_k_max),
            min_value=0.1, max_value=500.0, step=0.1,
        )
        ss.t_total = c3.number_input(
            "Simulation time (s)", value=float(ss.t_total),
            min_value=0.5, max_value=20.0, step=0.5,
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
        st.caption("Rate expression: Python with V, EXP_FACTOR, exp(), and param names.")
        trans_df = pd.DataFrame(
            [{"from": t.from_state, "to": t.to_state, "rate_expr": t.rate_expr}
             for t in ss.msm_def.transitions]
        )
        edited_trans = st.data_editor(
            trans_df,
            width="stretch",
            num_rows="dynamic",
            column_config={
                "from":      st.column_config.TextColumn("From",            required=True),
                "to":        st.column_config.TextColumn("To",              required=True),
                "rate_expr": st.column_config.TextColumn("Rate expression", required=True),
            },
            key="trans_editor",
            height=400,
        )

    with col_params:
        st.subheader("Parameters")
        st.caption("Values used as initial guess for optimisation.")
        params_df = pd.DataFrame(
            [{"name":    p.name,
              "initial": p.initial_value,
              "lower":   p.lower_bound,
              "upper":   p.upper_bound,
              "freeze":  p.frozen}
             for p in ss.msm_def.parameters]
        )
        edited_params = st.data_editor(
            params_df,
            width="stretch",
            num_rows="dynamic",
            column_config={
                "name":    st.column_config.TextColumn("Name", required=True),
                "initial": st.column_config.NumberColumn("Initial value", format="%.4g"),
                "lower":   st.column_config.NumberColumn("Lower bound",   format="%.4g"),
                "upper":   st.column_config.NumberColumn("Upper bound",   format="%.4g"),
                "freeze":  st.column_config.CheckboxColumn(
                    "Freeze", help="Exclude this parameter from optimisation"
                ),
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
                for _, row in edited_trans.dropna(subset=["from", "to", "rate_expr"]).iterrows()
            ]
            new_params = [
                ParamSpec(
                    name=str(row["name"]),
                    initial_value=float(row["initial"]),
                    lower_bound=float(row["lower"]),
                    upper_bound=float(row["upper"]),
                    frozen=bool(row.get("freeze", False)),
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
                ss.initial_conditions = None
                ss.fitted_params = None
                msg_col.success(
                    f"Model applied: {len(new_states)} states, "
                    f"{len(new_trans)} transitions, {len(new_params)} parameters."
                )
        except Exception as exc:
            msg_col.error(f"Failed to parse model: {exc}")

    with st.expander("Initial state distribution", expanded=False):
        st.caption("Must sum to 1. Leave all zero to use the default (uniform over closed states).")
        ic_vals  = _ic().copy()
        ic_names = ss.msm_def.state_names
        ic_cols  = st.columns(min(len(ic_names), 12))
        for i, (col, name) in enumerate(zip(ic_cols, ic_names)):
            ic_vals[i] = col.number_input(
                name, value=float(ic_vals[i]),
                min_value=0.0, max_value=1.0, step=0.001, format="%.4f",
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
