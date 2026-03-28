"""Tab 2 — Protocol Setup."""

import streamlit as st

from helpers import PROTOCOL_KEYS, PROTOCOL_LABELS, _voltage_preview


def render() -> None:
    ss = st.session_state

    st.info(
        "Adjust timing and voltages for each experimental protocol. "
        "The waveform preview updates instantly."
    )
    proto_subtabs = st.tabs([PROTOCOL_LABELS[k] for k in PROTOCOL_KEYS])

    # ── Activation ─────────────────────────────────────────────────────────────
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

    # ── Inactivation ───────────────────────────────────────────────────────────
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

    # ── Closed-state inactivation ──────────────────────────────────────────────
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

    # ── Recovery ───────────────────────────────────────────────────────────────
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
