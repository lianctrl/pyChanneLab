"""Tab 5 — Optimisation."""
import math
from datetime import datetime

import pandas as pd
import streamlit as st

from helpers import generate_run_script

try:
    import torch
    from core.torch_simulator import get_device, preferred_dtype
    from core.torch_de import TorchPipelineOptimizer
    _TORCH_AVAILABLE = True
    _torch_device    = get_device()
    _torch_dtype     = preferred_dtype(_torch_device)
except ImportError:
    _TORCH_AVAILABLE = False
    _torch_device    = None
    _torch_dtype     = None


def render() -> None:
    ss = st.session_state

    model_errors = ss.msm_def.validate()
    if model_errors:
        st.error("Fix model errors in the MSM Builder tab first.")
        for e in model_errors:
            st.write(e)
        return

    if not ss.exp_data:
        st.warning("Load at least one experimental dataset in the Data tab first.")
        return

    st.subheader("Optimisation settings")

    if not _TORCH_AVAILABLE:
        st.error(
            "PyTorch is not installed. "
            "Install it with `pip install torch` to use the optimiser."
        )
        return

    _dev_name = str(_torch_device) if _torch_device else "cpu"
    st.caption(f"Device: **{_dev_name}**")

    run_global = st.toggle(
        "Run global search (Differential Evolution) before local refinement",
        value=True,
        key="opt_run_global",
        help="Disable to skip DE and start Adam → L-BFGS directly from the initial parameter guess.",
    )

    if run_global:
        st.markdown("**Differential Evolution (global)**")
        col1, col2, col3, col4 = st.columns(4)
        pop_size   = col1.number_input("Population size",     value=50,   min_value=10,  max_value=500,  step=10,   key="opt_pop_size")
        de_maxiter = col2.number_input("DE generations",      value=200,  min_value=1,   max_value=5000, step=10,   key="opt_de_maxiter")
        de_F       = col3.number_input("Mutation scale F",    value=0.8,  min_value=0.1, max_value=2.0,  format="%.2f", key="opt_de_F")
        de_CR      = col4.number_input("Crossover prob. CR",  value=0.9,  min_value=0.1, max_value=1.0,  format="%.2f", key="opt_de_CR")
    else:
        st.info("Skipping DE — local refinement will start from the initial parameter guess.")
        pop_size   = 50
        de_maxiter = 0
        de_F       = 0.8
        de_CR      = 0.9

    st.markdown("**Adam (warm-up)**")
    col5, col6 = st.columns(2)
    n_adam  = col5.number_input("Adam steps",      value=500,  min_value=0, max_value=10000, step=50,    key="opt_n_adam")
    adam_lr = col6.number_input("Learning rate",   value=0.05, min_value=1e-4, max_value=1.0, format="%.4f", key="opt_adam_lr")

    st.markdown("**L-BFGS (refinement)**")
    col7, col8 = st.columns(2)
    n_lbfgs      = col7.number_input("L-BFGS steps",              value=200, min_value=0, max_value=2000, step=20,  key="opt_lbfgs")
    n_peak_steps = col8.number_input("Peak-detection substeps",   value=50,  min_value=10, max_value=200, step=10, key="opt_peak_steps")

    st.subheader("Protocol weights")
    wc = st.columns(4)
    ss.opt_weights["activation"]    = wc[0].number_input("Activation",  value=float(ss.opt_weights["activation"]),    min_value=0.0, step=0.5, key="w_act")
    ss.opt_weights["inactivation"]  = wc[1].number_input("Inactivation", value=float(ss.opt_weights["inactivation"]), min_value=0.0, step=0.5, key="w_inact")
    ss.opt_weights["cs_inactivation"] = wc[2].number_input("CS Inact.",  value=float(ss.opt_weights["cs_inactivation"]), min_value=0.0, step=0.5, key="w_csi")
    ss.opt_weights["recovery"]      = wc[3].number_input("Recovery",    value=float(ss.opt_weights["recovery"]),      min_value=0.0, step=0.5, key="w_rec")

    st.divider()

    _frozen = [p for p in ss.msm_def.parameters if p.frozen]
    if _frozen:
        st.info(
            "Frozen (fixed) parameters: "
            + ", ".join(f"**{p.name}** = {p.initial_value:.4g}" for p in _frozen)
        )

    _btn_col, _dl_col = st.columns([2, 1])

    # ── Export run script (reads all current session state at click time) ──────
    _dl_col.download_button(
        "💾 Export run script (.py)",
        data=generate_run_script(),
        file_name=f"pychannelab_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
        mime="text/plain",
        help=(
            "Self-contained Python script with all current settings (model, protocols, "
            "data, weights, optimiser hyperparameters). "
            "Run it on a cluster or workstation to reproduce the optimisation."
        ),
    )

    if _btn_col.button("▶ Run Optimisation", type="primary", key="run_opt"):
        defn = ss.msm_def
        bounds = list(defn.free_bounds)
        log: list = []

        _de_gens     = int(de_maxiter) if run_global else 0
        _adam_steps  = int(n_adam)
        _lbfgs_steps = int(n_lbfgs)
        _total_iters = _de_gens + _adam_steps + _lbfgs_steps

        _prog            = st.progress(0.0, text="Starting…")
        _mc1, _mc2, _mc3 = st.columns(3)
        _m_iter  = _mc1.empty()
        _m_cost  = _mc2.empty()
        _m_phase = _mc3.empty()
        _chart   = st.empty()

        _cost_hist    = []
        _global_iter  = [0]

        def _cb(iteration, cost, convergence):
            _global_iter[0] += 1
            g = _global_iter[0]

            msg = f"Iter {g:5d} | cost = {cost:.6e}"
            log.append(msg)

            if iteration <= _de_gens:
                phase = "DE (global)"
            elif iteration <= _de_gens + _adam_steps:
                phase = "Adam"
            else:
                phase = "L-BFGS"

            pct = min(iteration / max(_total_iters, 1), 1.0)
            _prog.progress(pct, text=f"{phase}  ·  iter {iteration} / {_total_iters}")
            _m_iter.metric("Iteration", f"{iteration:,}")
            _m_cost.metric("Cost",      f"{cost:.4e}")
            _m_phase.metric("Phase",    phase)

            _cost_hist.append(math.log10(max(cost, 1e-15)))
            if len(_cost_hist) == 1 or len(_cost_hist) % 5 == 0:
                _chart.line_chart(
                    pd.DataFrame(
                        {"log10(cost)": _cost_hist},
                        index=range(1, len(_cost_hist) + 1),
                    ),
                    height=200, x_label="callback #", y_label="log10(cost)",
                )

        pipeline = TorchPipelineOptimizer(
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
        ss.opt_costs_initial = pipeline.cost_breakdown(defn.free_initial_guess)

        result = pipeline.optimize(
            bounds=bounds,
            pop_size=int(pop_size),
            de_maxiter=_de_gens,
            F=float(de_F),
            CR=float(de_CR),
            n_adam=_adam_steps,
            adam_lr=float(adam_lr),
            n_lbfgs=_lbfgs_steps,
            progress_callback=_cb,
            skip_de=not run_global,
            initial_params=defn.free_initial_guess if not run_global else None,
        )

        ss.opt_costs_final = pipeline.cost_breakdown(result.x)
        _prog.progress(1.0, text="Done")
        ss.fitted_params = defn.expand_params(result.x)
        ss.opt_result    = result
        ss.opt_log       = log
        st.success(f"Done! Final cost: {result.fun:.6f}")

    if ss.opt_result is not None:
        ca, cb = st.columns(2)
        ca.metric("Final cost", f"{ss.opt_result.fun:.6f}")
        cb.metric("Iterations", str(ss.opt_result.get("nit", "—")))

        if ss.opt_costs_initial and ss.opt_costs_final:
            st.subheader("Cost breakdown")
            rows = []
            for pk in ss.opt_costs_final:
                ci  = ss.opt_costs_initial.get(pk, float("nan"))
                cf  = ss.opt_costs_final[pk]
                imp = (ci - cf) / ci * 100 if ci else float("nan")
                rows.append({
                    "Protocol":        pk,
                    "Initial":         f"{ci:.6f}",
                    "Final":           f"{cf:.6f}",
                    "Improvement (%)": f"{imp:.1f}",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        if ss.opt_log:
            with st.expander("Optimisation log (last 200 iterations)"):
                st.text("\n".join(ss.opt_log[-200:]))
