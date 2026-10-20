"""Tab 3 — Custom Protocol Builder."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.config import CustomProtocolConfig, ProtocolSegment
from core.protocols import CustomProtocol
from core.data_loader import load_from_bytes


# ── Helpers ────────────────────────────────────────────────────────────────────


def _default_protocol(n: int) -> CustomProtocolConfig:
    return CustomProtocolConfig(
        name=f"Protocol {n}",
        segments=[
            ProtocolSegment(name="Hold", voltage=-90.0, duration=0.5),
            ProtocolSegment(name="Test", voltage=-90.0, duration=0.1),
            ProtocolSegment(name="Tail", voltage=-50.0, duration=0.5),
        ],
        sweep_type="voltage",
        sweep_segment_idx=1,
        sweep_min=-90.0,
        sweep_max=60.0,
        sweep_step=10.0,
        obs_segment_idx=1,
        obs_offset=0.0,
        obs_window_length=None,
        obs_type="peak_open",
    )


def _segs_to_df(segs: list) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Name": s.name, "Voltage (mV)": float(s.voltage), "Duration (s)": float(s.duration)}
         for s in segs]
    )


def _df_to_segs(df: pd.DataFrame) -> list:
    segs = []
    for _, row in df.iterrows():
        v = float(row["Voltage (mV)"]) if pd.notna(row.get("Voltage (mV)")) else -90.0
        d = float(row["Duration (s)"]) if pd.notna(row.get("Duration (s)")) else 0.1
        n = str(row["Name"]) if pd.notna(row.get("Name")) else ""
        if d > 0:
            segs.append(ProtocolSegment(name=n, voltage=v, duration=d))
    return segs


def _seg_label(idx: int, seg: ProtocolSegment) -> str:
    label = seg.name.strip() or f"Segment {idx}"
    return f"[{idx}] {label}"


def _waveform_preview(cfg: CustomProtocolConfig) -> go.Figure:
    """Step-chart of V(t) for min, mid, and max sweep values with window shading."""
    fig = go.Figure()

    if not cfg.segments:
        fig.update_layout(
            xaxis_title="Time (s)", yaxis_title="Voltage (mV)",
            height=260, margin=dict(t=10, b=40, l=60, r=20),
        )
        return fig

    proto = CustomProtocol(cfg)
    sweep_vals = proto.get_sweep_values()
    if len(sweep_vals) == 0:
        return fig

    # Pick at most 3 representative sweep values
    indices = [0]
    if len(sweep_vals) > 2:
        indices.append(len(sweep_vals) // 2)
    if len(sweep_vals) > 1:
        indices.append(len(sweep_vals) - 1)
    preview_sweeps = [sweep_vals[i] for i in indices]

    dt_prev = 0.001  # coarse step for preview

    for sv in preview_sweeps:
        total_dur = proto.total_duration(sv)
        t_arr = np.arange(0, total_dur + dt_prev, dt_prev)
        vf = proto.get_voltage_function(sv)
        v_arr = [vf(ti) for ti in t_arr]

        if cfg.sweep_type == "voltage":
            label = f"V_sweep = {sv:.0f} mV"
        else:
            label = f"dur_sweep = {sv:.3f} s"

        fig.add_trace(go.Scatter(x=t_arr, y=v_arr, mode="lines", name=label))

    # Shade observable window for the middle preview sweep
    mid_sv = preview_sweeps[len(preview_sweeps) // 2]
    t0_obs, t1_obs = proto.obs_window(mid_sv)
    if t1_obs > t0_obs:
        fig.add_vrect(
            x0=t0_obs, x1=t1_obs,
            fillcolor="green", opacity=0.12, line_width=0,
            annotation_text="obs", annotation_position="top left",
            annotation_font_size=11,
        )
    if cfg.obs_type == "ratio":
        t0_ref, t1_ref = proto.ref_window(mid_sv)
        if t1_ref > t0_ref:
            fig.add_vrect(
                x0=t0_ref, x1=t1_ref,
                fillcolor="blue", opacity=0.10, line_width=0,
                annotation_text="ref", annotation_position="top right",
                annotation_font_size=11,
            )

    # Segment boundary lines for the middle sweep
    segs_mid = proto._apply_sweep(mid_sv)
    t_boundaries = np.cumsum([s.duration for s in segs_mid])
    for tb in t_boundaries[:-1]:
        fig.add_vline(x=float(tb), line_dash="dot", line_color="gray", line_width=1)

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Voltage (mV)",
        height=260,
        margin=dict(t=10, b=40, l=60, r=20),
        legend=dict(orientation="h", y=1.15),
    )
    return fig


# ── Main render ────────────────────────────────────────────────────────────────


def render() -> None:
    ss = st.session_state

    st.info(
        "Build custom voltage-clamp protocols from constant-voltage segments. "
        "Sweep one segment's **voltage** (Family 1, like Activation/Inactivation) "
        "or **duration** (Family 2, like CSI/Recovery). "
        "Custom protocol simulations appear in the **Preview** tab."
    )

    col_lib, col_edit = st.columns([1, 2])

    # ── Protocol library ───────────────────────────────────────────────────────
    with col_lib:
        st.subheader("Protocol library")

        protos = ss.custom_protocols
        if not protos:
            st.caption("No custom protocols yet.")
        else:
            for i, p in enumerate(protos):
                is_active = (i == ss.active_custom_protocol_idx)
                btn_label = f"{'▶ ' if is_active else ''}{p.name}"
                c1, c2 = st.columns([4, 1])
                if c1.button(btn_label, key=f"sel_proto_{i}", use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    ss.active_custom_protocol_idx = i
                    st.rerun()
                if c2.button("✕", key=f"del_proto_{i}", help="Delete"):
                    # Clean up experimental data for this and re-index later ones
                    ss.exp_data.pop(f"custom_{i}", None)
                    # Shift experimental data keys for higher-indexed protocols
                    for j in range(i + 1, len(protos)):
                        if f"custom_{j}" in ss.exp_data:
                            ss.exp_data[f"custom_{j-1}"] = ss.exp_data.pop(f"custom_{j}")
                    protos.pop(i)
                    ss.active_custom_protocol_idx = max(0, min(ss.active_custom_protocol_idx, len(protos) - 1))
                    st.rerun()

        if st.button("＋ New protocol", use_container_width=True):
            protos.append(_default_protocol(len(protos) + 1))
            ss.active_custom_protocol_idx = len(protos) - 1
            st.rerun()

    # ── Editor ────────────────────────────────────────────────────────────────
    with col_edit:
        if not ss.custom_protocols:
            st.info("Click **＋ New protocol** to get started.")
            return

        idx = ss.active_custom_protocol_idx
        if idx >= len(ss.custom_protocols):
            idx = 0
            ss.active_custom_protocol_idx = 0

        cfg = ss.custom_protocols[idx]

        # Name
        cfg.name = st.text_input("Protocol name", value=cfg.name, key=f"proto_name_{idx}")

        # ── Segment table ──────────────────────────────────────────────────────
        st.markdown("**Segments** (rows are constant-V blocks in sequence)")

        df = _segs_to_df(cfg.segments)
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            key=f"segs_editor_{idx}",
            column_config={
                "Name": st.column_config.TextColumn("Name", width="medium"),
                "Voltage (mV)": st.column_config.NumberColumn(
                    "Voltage (mV)", min_value=-300.0, max_value=300.0,
                    step=1.0, default=-90.0,
                ),
                "Duration (s)": st.column_config.NumberColumn(
                    "Duration (s)", min_value=0.001, max_value=100.0,
                    step=0.01, format="%.3f", default=0.1,
                ),
            },
        )
        cfg.segments = _df_to_segs(edited_df)

        if not cfg.segments:
            st.warning("Add at least one segment.")
            return

        seg_labels = [_seg_label(i, s) for i, s in enumerate(cfg.segments)]

        st.divider()

        # ── Sweep configuration ───────────────────────────────────────────────
        st.markdown("**Sweep axis** — which parameter varies across simulations")

        s_col1, s_col2 = st.columns(2)
        sweep_type_opts = ["voltage", "duration"]
        cfg.sweep_type = s_col1.selectbox(
            "Sweep type",
            sweep_type_opts,
            index=sweep_type_opts.index(cfg.sweep_type) if cfg.sweep_type in sweep_type_opts else 0,
            key=f"sweep_type_{idx}",
            help="voltage → V of one segment varies (Family 1)  |  duration → timing of one segment varies (Family 2)",
        )

        safe_sweep_idx = min(cfg.sweep_segment_idx, len(seg_labels) - 1)
        swept_label = s_col2.selectbox(
            "Swept segment",
            seg_labels,
            index=safe_sweep_idx,
            key=f"swept_seg_{idx}",
        )
        cfg.sweep_segment_idx = seg_labels.index(swept_label)

        mn_col, mx_col, st_col = st.columns(3)
        if cfg.sweep_type == "voltage":
            cfg.sweep_min = mn_col.number_input(
                "Min (mV)", value=float(cfg.sweep_min), step=5.0, key=f"sw_mn_{idx}"
            )
            cfg.sweep_max = mx_col.number_input(
                "Max (mV)", value=float(cfg.sweep_max), step=5.0, key=f"sw_mx_{idx}"
            )
            cfg.sweep_step = st_col.number_input(
                "Step (mV)", value=float(cfg.sweep_step), min_value=0.5,
                step=1.0, key=f"sw_st_{idx}"
            )
        else:
            cfg.sweep_min = mn_col.number_input(
                "Min (s)", value=float(cfg.sweep_min), min_value=0.001,
                step=0.01, format="%.3f", key=f"sw_mn_{idx}"
            )
            cfg.sweep_max = mx_col.number_input(
                "Max (s)", value=float(cfg.sweep_max), step=0.01,
                format="%.3f", key=f"sw_mx_{idx}"
            )
            cfg.sweep_step = st_col.number_input(
                "Step (s)", value=float(cfg.sweep_step), min_value=0.001,
                step=0.005, format="%.3f", key=f"sw_st_{idx}"
            )

        st.divider()

        # ── Observable configuration ──────────────────────────────────────────
        st.markdown("**Observable extraction** — what to measure and where")

        obs_type_opts = ["peak_open", "conductance", "current", "ratio"]
        obs_labels = {
            "peak_open": "Peak open probability",
            "conductance": "Peak conductance (g / g_max)",
            "current": "Peak current (I / I_max)",
            "ratio": "Ratio (peak in window / peak in reference window)",
        }
        cfg.obs_type = st.selectbox(
            "Observable type",
            obs_type_opts,
            index=obs_type_opts.index(cfg.obs_type) if cfg.obs_type in obs_type_opts else 0,
            format_func=lambda k: obs_labels[k],
            key=f"obs_type_{idx}",
        )

        st.caption("Measurement window (anchored to start of selected segment + offset)")
        ow_col1, ow_col2, ow_col3 = st.columns(3)

        safe_obs_idx = min(cfg.obs_segment_idx, len(seg_labels) - 1)
        obs_label = ow_col1.selectbox(
            "Measured in segment",
            seg_labels,
            index=safe_obs_idx,
            key=f"obs_seg_{idx}",
        )
        cfg.obs_segment_idx = seg_labels.index(obs_label)

        cfg.obs_offset = ow_col2.number_input(
            "Offset (s)", value=float(cfg.obs_offset),
            min_value=0.0, step=0.005, format="%.3f", key=f"obs_off_{idx}",
        )

        full_window = ow_col3.checkbox(
            "Full segment", value=(cfg.obs_window_length is None), key=f"obs_full_{idx}",
            help="Use the entire segment duration (minus offset) as the measurement window"
        )
        if full_window:
            cfg.obs_window_length = None
        else:
            default_wl = cfg.obs_window_length if cfg.obs_window_length is not None else 0.05
            cfg.obs_window_length = ow_col3.number_input(
                "Length (s)", value=float(default_wl),
                min_value=0.001, step=0.005, format="%.3f", key=f"obs_len_{idx}",
            )

        if cfg.obs_type == "ratio":
            st.caption("Reference window")
            rw_col1, rw_col2, rw_col3 = st.columns(3)

            safe_ref_idx = min(cfg.obs_ref_segment_idx, len(seg_labels) - 1)
            ref_label = rw_col1.selectbox(
                "Reference segment",
                seg_labels,
                index=safe_ref_idx,
                key=f"ref_seg_{idx}",
            )
            cfg.obs_ref_segment_idx = seg_labels.index(ref_label)

            cfg.obs_ref_offset = rw_col2.number_input(
                "Ref offset (s)", value=float(cfg.obs_ref_offset),
                min_value=0.0, step=0.005, format="%.3f", key=f"ref_off_{idx}",
            )

            full_ref = rw_col3.checkbox(
                "Full segment (ref)", value=(cfg.obs_ref_window_length is None),
                key=f"ref_full_{idx}",
            )
            if full_ref:
                cfg.obs_ref_window_length = None
            else:
                default_rl = cfg.obs_ref_window_length if cfg.obs_ref_window_length is not None else 0.05
                cfg.obs_ref_window_length = rw_col3.number_input(
                    "Ref length (s)", value=float(default_rl),
                    min_value=0.001, step=0.005, format="%.3f", key=f"ref_len_{idx}",
                )

        st.divider()

        cfg.enabled = st.checkbox(
            "Include in optimisation (scipy exported script)",
            value=cfg.enabled,
            key=f"enabled_{idx}",
            help="When enabled and experimental data is uploaded, this protocol contributes to the cost function in the exported run script.",
        )
        cfg.steady_state_start = st.checkbox(
            "Start each sweep from steady state at the first segment's voltage",
            value=bool(cfg.steady_state_start),
            key=f"ss_start_{idx}",
            help="The first segment is treated as the holding potential the cell rests at between sweeps.",
        )

        # ── Waveform preview ──────────────────────────────────────────────────
        st.markdown("**Waveform preview** (green = measurement window)")
        st.plotly_chart(_waveform_preview(cfg), width="stretch")

        # ── Experimental data upload ──────────────────────────────────────────
        proto_key = f"custom_{idx}"
        with st.expander("Upload experimental data for this protocol", expanded=False):
            st.markdown(
                "CSV format: header row, then columns `x` (sweep parameter values), "
                "`y` (normalised observable), optionally `y_err`."
            )
            uploaded = st.file_uploader(
                f"CSV — {cfg.name}",
                type=["csv"],
                key=f"upload_custom_{idx}",
            )
            if uploaded is not None:
                try:
                    x, y, y_err = load_from_bytes(uploaded.read())
                    ss.exp_data[proto_key] = (x, y, y_err)
                    st.success(f"Loaded {len(x)} data points.")
                except Exception as exc:
                    st.error(f"Failed to load: {exc}")

            if proto_key in ss.exp_data:
                x, y, y_err = ss.exp_data[proto_key]
                fig_data = go.Figure()
                err_kw = (
                    dict(error_y=dict(type="data", array=y_err, visible=True))
                    if y_err is not None else {}
                )
                fig_data.add_trace(
                    go.Scatter(x=x, y=y, mode="markers",
                               marker=dict(size=7, color="#1f77b4"), name="Data", **err_kw)
                )
                x_label = "Voltage (mV)" if cfg.sweep_type == "voltage" else "Duration (s)"
                y_label = {"peak_open": "P_open", "conductance": "g / g_max",
                           "current": "I / I_max", "ratio": "Ratio"}.get(cfg.obs_type, "Observable")
                fig_data.update_layout(
                    xaxis_title=x_label, yaxis_title=y_label,
                    height=240, margin=dict(t=10, b=40, l=60, r=20),
                )
                st.plotly_chart(fig_data, width="stretch")

                if st.button(f"Remove data for {cfg.name}", key=f"rm_custom_{idx}"):
                    del ss.exp_data[proto_key]
                    st.rerun()
