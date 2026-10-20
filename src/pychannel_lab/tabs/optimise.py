"""Tab 5 — Optimisation."""

import math
from datetime import datetime

import pandas as pd
import streamlit as st

from helpers import generate_run_script, _simulate_all, _comparison_figure

try:
    import torch
    from core.torch_simulator import get_device, preferred_dtype
    from core.torch_de import TorchPipelineOptimizer

    _TORCH_AVAILABLE = True
    _torch_device = get_device()
    _torch_dtype = preferred_dtype(_torch_device)
except ImportError:
    _TORCH_AVAILABLE = False
    _torch_device = None
    _torch_dtype = None


def _activate_seed(entry: dict) -> None:
    """Make one multi-seed result the active fit (feeds the Results tab)."""
    ss = st.session_state
    ss.fitted_params = entry["params_full"]
    ss.opt_result = entry["result"]
    ss.opt_costs_final = entry["breakdown"]
    ss.active_seed = entry["seed"]


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
        pop_size = col1.number_input(
            "Population size",
            value=50,
            min_value=10,
            max_value=500,
            step=10,
            key="opt_pop_size",
        )
        de_maxiter = col2.number_input(
            "DE generations",
            value=200,
            min_value=1,
            max_value=5000,
            step=10,
            key="opt_de_maxiter",
        )
        de_F = col3.number_input(
            "Mutation scale F",
            value=0.8,
            min_value=0.1,
            max_value=2.0,
            format="%.2f",
            key="opt_de_F",
        )
        de_CR = col4.number_input(
            "Crossover prob. CR",
            value=0.9,
            min_value=0.1,
            max_value=1.0,
            format="%.2f",
            key="opt_de_CR",
        )
        colA, colB = st.columns(2)
        de_strategy = colA.selectbox(
            "DE strategy",
            options=["adaptive", "rand1"],
            index=0,
            key="opt_de_strategy",
            help=(
                "adaptive: JADE-style DE/current-to-pbest/1 with self-adapting "
                "F/CR (more robust, ignores the fixed F/CR above). "
                "rand1: classic DE/rand/1/bin with the fixed F/CR."
            ),
        )
        n_starts = colB.number_input(
            "Local multi-starts (top-K)",
            value=5,
            min_value=1,
            max_value=50,
            step=1,
            key="opt_n_starts",
            help=(
                "Polish the K best DE members with Adam → L-BFGS and keep the "
                "lowest-cost result. Higher K is more robust but slower."
            ),
        )
    else:
        st.info(
            "Skipping DE — local refinement will start from the initial parameter guess."
        )
        pop_size = 50
        de_maxiter = 0
        de_F = 0.8
        de_CR = 0.9
        de_strategy = "adaptive"
        n_starts = 1

    st.markdown("**Adam (warm-up)**")
    col5, col6 = st.columns(2)
    n_adam = col5.number_input(
        "Adam steps", value=500, min_value=0, max_value=10000, step=50, key="opt_n_adam"
    )
    adam_lr = col6.number_input(
        "Learning rate",
        value=0.05,
        min_value=1e-4,
        max_value=1.0,
        format="%.4f",
        key="opt_adam_lr",
    )

    st.markdown("**L-BFGS (refinement)**")
    col7, col8 = st.columns(2)
    n_lbfgs = col7.number_input(
        "L-BFGS steps", value=200, min_value=0, max_value=2000, step=20, key="opt_lbfgs"
    )
    n_peak_steps = col8.number_input(
        "Peak-detection substeps",
        value=50,
        min_value=10,
        max_value=200,
        step=10,
        key="opt_peak_steps",
    )

    st.subheader("Regularization")
    reg_lambda = st.number_input(
        "L2 regularization λ",
        value=float(ss.get("opt_reg_lambda", 0.0)),
        min_value=0.0,
        format="%.6f",
        step=1e-5,
        key="opt_reg_lambda",
        help=(
            "Scale-invariant pull toward the initial-guess prior: penalises the "
            "squared log-fold deviation of each rate (linear params normalised by "
            "their bound width). 0 = off. Try 1e-3…1e-1 to make fits more "
            "reproducible / less seed-dependent without biasing rates toward zero."
        ),
    )
    rest_lambda = st.number_input(
        "Resting-inactivation penalty λ",
        value=float(ss.get("opt_rest_lambda", 0.0)),
        min_value=0.0,
        format="%.4f",
        step=0.01,
        key="opt_rest_lambda",
        help=(
            "Adds λ·I_rest² to the loss, where I_rest is the inactivated fraction of "
            "the steady state at the resting voltage below. Normalised data cannot "
            "pin how many channels are available at rest, so without it fits may "
            "leave many inactivated. "
            "0 = off; 0.1 brought WT to ~3% resting inactivation."
        ),
    )
    v_rest = st.number_input(
        "Resting voltage for the penalty (mV)",
        value=float(ss.get("opt_v_rest", -90.0)),
        step=5.0,
        key="opt_v_rest",
        help="Voltage at which the resting inactivated fraction is evaluated "
        "(normally the holding potential).",
        disabled=rest_lambda == 0.0,
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

    st.subheader("Reproducibility")
    multi_seed = st.toggle(
        "Multi-seed sweep",
        value=False,
        key="opt_multi_seed",
        help=(
            "Run the whole pipeline once per random seed and keep every result, "
            "so you can inspect the different basins and pick the best fit. "
            "Different seeds can give qualitatively different curves "
            "(e.g. exponential vs sigmoidal recovery)."
        ),
    )
    if multi_seed:
        _sc1, _sc2 = st.columns(2)
        base_seed = int(
            _sc1.number_input("Base seed", value=42, min_value=0, step=1, key="opt_seed")
        )
        n_seeds = int(
            _sc2.number_input(
                "Number of seeds",
                value=3,
                min_value=1,
                max_value=5,  # hard cap
                step=1,
                key="opt_n_seeds",
            )
        )
        seeds = [base_seed + i for i in range(n_seeds)]
        st.caption(f"Seeds to run: {seeds}")
    else:
        base_seed = int(
            st.number_input("Random seed", value=42, min_value=0, step=1, key="opt_seed")
        )
        seeds = [base_seed]

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

        _de_gens = int(de_maxiter) if run_global else 0
        _adam_steps = int(n_adam)
        _lbfgs_steps = int(n_lbfgs)
        _total_iters = _de_gens + _adam_steps + _lbfgs_steps
        _n_seeds = len(seeds)

        _prog = st.progress(0.0, text="Starting…")
        _mc1, _mc2, _mc3 = st.columns(3)
        _m_iter = _mc1.empty()
        _m_cost = _mc2.empty()
        _m_phase = _mc3.empty()
        _chart = st.empty()

        _cost_hist = []
        _seed_ctx = {"idx": 0, "seed": seeds[0]}

        def _cb(iteration, cost, convergence):
            si = _seed_ctx["idx"]
            msg = f"[seed {_seed_ctx['seed']}] iter {iteration:5d} | cost = {cost:.6e}"
            log.append(msg)

            if iteration <= _de_gens:
                phase = "DE (global)"
            elif iteration <= _de_gens + _adam_steps:
                phase = "Adam"
            else:
                phase = "L-BFGS"

            seed_label = f"Seed {si + 1}/{_n_seeds}  ·  " if _n_seeds > 1 else ""
            frac = min(iteration / max(_total_iters, 1), 1.0)
            pct = (si + frac) / _n_seeds
            _prog.progress(
                pct, text=f"{seed_label}{phase}  ·  iter {iteration} / {_total_iters}"
            )
            _m_iter.metric("Iteration", f"{iteration:,}")
            _m_cost.metric("Cost", f"{cost:.4e}")
            _m_phase.metric("Phase", f"{seed_label}{phase}")

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
            reg_lambda=float(reg_lambda),
            rest_lambda=float(rest_lambda),
            v_rest=float(v_rest),
        )
        ss.opt_costs_initial = pipeline.cost_breakdown(defn.free_initial_guess)

        results = []
        for _i, _s in enumerate(seeds):
            _seed_ctx["idx"] = _i
            _seed_ctx["seed"] = _s
            _res = pipeline.optimize(
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
                strategy=de_strategy,
                n_starts=int(n_starts),
                seed=int(_s),
            )
            results.append(
                {
                    "seed": int(_s),
                    "result": _res,
                    "params_full": defn.expand_params(_res.x),
                    "fun": float(_res.fun),
                    "breakdown": pipeline.cost_breakdown(_res.x),
                }
            )

        _prog.progress(1.0, text="Done")
        ss.multiseed_results = results
        ss.opt_log = log

        # Activate the best-by-objective run by default (user can re-pick below).
        best = min(results, key=lambda r: r["fun"])
        _activate_seed(best)
        st.success(
            f"Done! Ran {len(results)} seed(s). "
            f"Best objective {best['fun']:.6f} (seed {best['seed']})."
        )

    # ── Multi-seed inspection & manual selection ──────────────────────────────
    _ms = ss.get("multiseed_results", [])
    if len(_ms) > 1:
        st.subheader("Multi-seed results")
        st.caption(
            "Per-seed fit quality (RMSE = √MSE per protocol; objective = weighted "
            "cost that was minimised). Pick the seed whose curves look best below — "
            "lowest objective is not always the best-shaped fit."
        )
        _protos = [k for k in ss.exp_data if ss.exp_data.get(k) is not None]
        _rows = []
        for r in _ms:
            row = {"seed": r["seed"], "objective": f"{r['fun']:.6f}"}
            for pk in _protos:
                mse = r["breakdown"].get(pk)
                row[f"RMSE {pk}"] = f"{mse ** 0.5:.4f}" if mse is not None else "—"
            _rows.append(row)
        st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)

        _seed_opts = [r["seed"] for r in _ms]
        _active = ss.get("active_seed", _seed_opts[0])
        _sel = st.selectbox(
            "Active seed (its parameters feed the Results tab)",
            options=_seed_opts,
            index=_seed_opts.index(_active) if _active in _seed_opts else 0,
            key="multiseed_select",
        )
        _chosen = next(r for r in _ms if r["seed"] == _sel)
        if ss.get("active_seed") != _sel:
            _activate_seed(_chosen)

        with st.spinner(f"Simulating seed {_sel}…"):
            _sim = _simulate_all(_chosen["params_full"])
        st.plotly_chart(
            _comparison_figure(_chosen["params_full"], _sim_data=_sim),
            width="stretch",
        )

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
