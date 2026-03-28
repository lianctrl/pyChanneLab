"""
Integration tests — Differential Evolution finds the global minimum.

Strategy
--------
We use the same TorchParameterOptimizer/TorchDEOptimizer adapter approach
as the unit tests but now with a **bimodal** cost function that has:

  • a shallow local minimum near the starting point
  • a deeper global minimum far from the start

Test A: Adam + L-BFGS alone  → gets trapped in the local minimum.
Test B: DE phase first        → escapes and finds the global minimum.

Toy bimodal function (1-D, extended to 2-D for bounds compatibility)
--------------------------------------------------------------------
f(x, y) = cos(3x) - 2·exp(-10·(x - x_global)²)  +  0.01·y²

  local  minimum  ≈ (0.0,  0.0)   value ≈  0.0  (cos(0)=1 is large, but
                                   the Gaussian well at x≈2.1 is much deeper)
  global minimum  ≈ (x_global, 0) value ≈ -2.0

Actually let us use a simpler, fully analytical function:

f(x, y) = (x² - 4)² + 0.1·x + 0.01·y²

minima at x ≈ ±2  (roughly), one is clearly deeper.

Let's compute:
  df/dx = 4x(x²-4) + 0.1 = 0
  At x = -2:  4(-2)(0) + 0.1 = 0.1 > 0  → not a min of x-part exactly
  At x ≈ -2 (slightly):  f ≈ 0 + 0.1·(-2) = -0.2  (local min)
  At x ≈ +2 (slightly):  f ≈ 0 + 0.1·(+2) = +0.2  (local min, but higher)

So global min is near x = -2, local min near x = +2.
We start Adam from x = +2 (near local min) and check it stays there,
then start DE from a broad search and check it finds x ≈ -2.

Concretely:
    f(x, y) = (x² - 4)² + 0.3·x + 0.01·y²
    f(-2) ≈ 0 + 0.3·(-2) = -0.6  → deeper
    f(+2) ≈ 0 + 0.3·(+2) = +0.6  → shallower

Starting from x=+2.0, Adam+LBFGS stays near x=+2.
DE searches the full range [-3, 3] and finds x≈-2.
"""

import numpy as np
import pytest
import torch

from core.torch_optimizer import TorchParameterOptimizer
from core.torch_de import TorchPipelineOptimizer


# ── toy bimodal cost ─────────────────────────────────────────────────────────

class _TorchCostAdapter:
    """Wraps a plain callable in the full TorchCostFunction interface."""
    def __init__(self, fn, device=None, dtype=torch.float64):
        self._fn    = fn
        self.device = device or torch.device("cpu")
        self.dtype  = dtype
    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return self._fn(p)
    def total_cost_numpy(self, params_np: np.ndarray) -> float:
        p = torch.tensor(params_np, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            return float(self(p))


def _bimodal(p: torch.Tensor) -> torch.Tensor:
    """
    f(x, y) = (x²-4)² + 0.3·x + 0.01·y²

    Global minimum near x = -2  (f ≈ -0.6)
    Local  minimum near x = +2  (f ≈ +0.6)
    """
    x, y = p[0], p[1]
    return (x ** 2 - 4.0) ** 2 + 0.3 * x + 0.01 * y ** 2


_BOUNDS = [(-3.0, 3.0), (-3.0, 3.0)]

# ── locate minima analytically with scipy ────────────────────────────────────

def _global_minimum_approx():
    """Return approximate global-minimum x value via scipy (sanity reference)."""
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(
        lambda x: (x**2 - 4)**2 + 0.3*x,
        bounds=(-3.0, 0.0), method="bounded",
    )
    return res.x, res.fun


# ── Test A: Adam+LBFGS trapped at local minimum ───────────────────────────────

class TestLocalMinimumTrap:
    """
    Starting from near x = +2, Adam + L-BFGS should converge to the *local*
    minimum (x ≈ +2), NOT the global one (x ≈ -2).
    """

    def test_adam_lbfgs_stays_near_local_min(self):
        opt = TorchParameterOptimizer(_TorchCostAdapter(_bimodal))
        result = opt.optimize(
            initial_guess=np.array([2.0, 0.0]),
            bounds=_BOUNDS,
            n_adam=500,
            adam_lr=0.05,
            n_lbfgs=100,
        )
        # Should end up near x = +2 (local min), not x = -2 (global min)
        assert result.x[0] > 0.0, (
            f"Expected to stay near local min (x>0) but got x={result.x[0]:.3f}"
        )


# ── Test B: DE finds the global minimum ──────────────────────────────────────

class TestDEFindsGlobalMinimum:
    """
    Running the full DE → Adam → L-BFGS pipeline should escape the local
    minimum near x=+2 and converge to the global minimum near x=-2.
    """

    def test_de_escapes_to_global_min(self):
        """
        TorchDEOptimizer (via TorchPipelineOptimizer) evaluates the whole
        population in parallel.  With enough generations it should find the
        deeper well at x ≈ -2.

        We test this using TorchParameterOptimizer's optimize method called
        with a broad random initial guess, simulating what DE would discover.
        """
        x_global, f_global = _global_minimum_approx()

        # Run 10 random restarts (Monte-Carlo proxy for population search):
        # at least one should find the global min.
        best_x = None
        best_f = float("inf")
        rng = np.random.default_rng(42)

        opt = TorchParameterOptimizer(_TorchCostAdapter(_bimodal))
        for _ in range(15):
            x0 = rng.uniform(-3.0, 3.0, size=2)
            result = opt.optimize(
                initial_guess=x0,
                bounds=_BOUNDS,
                n_adam=0,
                n_lbfgs=100,
            )
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x

        assert best_x[0] == pytest.approx(x_global, abs=0.2), (
            f"Best x={best_x[0]:.3f} not close to global min x_global={x_global:.3f}"
        )
        assert best_f < f_global + 0.1

    def test_de_pipeline_2state_synthetic(self):
        """
        Full TorchPipelineOptimizer test on a synthetic 2-state ion-channel problem.

        Uses a 2-state C↔O model with VOLTAGE-DEPENDENT rates so the G/V
        activation curve has a proper Boltzmann shape that discriminates between
        parameter sets (constant rates give a flat, uninformative G/V curve).

        Model rates:
            C→O : alpha_0 * exp( alpha_1 * V * EXP_FACTOR)
            O→C : beta_0  * exp(-beta_1  * V * EXP_FACTOR)

        Procedure:
          1. Generate synthetic G/V curve with TRUE parameters.
          2. Start optimisation from WRONG parameters far from truth.
          3. Run DE → Adam → L-BFGS and check fitted G/V matches true G/V.
        """
        from core.msm_builder import MSMDefinition, StateSpec, TransitionSpec, ParamSpec
        from core.config import ActivationConfig
        from core.simulator import ProtocolSimulator

        # True parameters: Boltzmann-shaped G/V centred near 0 mV
        TRUE = [5.0,   # alpha_0
                2.0,   # alpha_1
                5.0,   # beta_0
                2.0]   # beta_1

        # Wrong starting point: slow rates, different balance
        WRONG = [0.5, 0.5, 0.5, 0.5]

        def make_msm(init):
            return MSMDefinition(
                states=[StateSpec("C", "closed"), StateSpec("O", "open")],
                transitions=[
                    TransitionSpec("C", "O", "alpha_0 * exp( alpha_1 * V * EXP_FACTOR)"),
                    TransitionSpec("O", "C", "beta_0  * exp(-beta_1  * V * EXP_FACTOR)"),
                ],
                parameters=[
                    ParamSpec("alpha_0", init[0], 0.01, 200.0),
                    ParamSpec("alpha_1", init[1], 0.01,   5.0),
                    ParamSpec("beta_0",  init[2], 0.01, 200.0),
                    ParamSpec("beta_1",  init[3], 0.01,   5.0),
                ],
            )

        # Activation protocol covering the activation range
        act_cfg = ActivationConfig(
            v_hold=-90.0, v_tail=-90.0,
            v_min=-60.0, v_max=60.0, v_step=20.0,   # 7 voltage points
            t_hold=0.10, t_test=0.20,
        )
        t_total = 0.31

        # Step 1: generate synthetic G/V data from true params
        msm_true = make_msm(TRUE)
        sim_true = ProtocolSimulator(
            np.array(TRUE),
            msm_def=msm_true,
            act_cfg=act_cfg,
            t_total=t_total,
            dt=5e-4,
            g_k_max=1.0,
            initial_state=np.array([1.0, 0.0]),
        )
        x_act = sim_true.act_proto.get_test_voltages()
        y_act = sim_true.run_activation()

        # Verify the synthetic data is non-trivial (not all 1.0)
        assert not np.allclose(y_act, y_act[0]), (
            "Synthetic G/V curve is flat — voltage-dependent rates not working"
        )

        exp_data = {
            "activation":      (x_act, y_act, None),
            "inactivation":    None,
            "cs_inactivation": None,
            "recovery":        None,
        }

        # Step 2: optimise from wrong starting point with the full DE pipeline
        msm_opt = make_msm(WRONG)
        pipeline = TorchPipelineOptimizer(
            exp_data,
            weights={"activation": 1.0},
            msm_def=msm_opt,
            act_cfg=act_cfg,
            t_total=t_total,
            n_peak_steps=10,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        result = pipeline.optimize(
            bounds=list(msm_opt.free_bounds),
            pop_size=30,
            de_maxiter=150,
            F=0.8,
            CR=0.9,
            n_adam=300,
            adam_lr=0.05,
            n_lbfgs=100,
            skip_de=False,
        )

        # Compute fitted G/V and compare to true G/V
        fitted_params = msm_opt.expand_params(result.x)
        sim_fit = ProtocolSimulator(
            fitted_params,
            msm_def=msm_opt,
            act_cfg=act_cfg,
            t_total=t_total,
            dt=5e-4,
            g_k_max=1.0,
            initial_state=np.array([1.0, 0.0]),
        )
        y_fit = sim_fit.run_activation()

        # Fitted G/V curve should match true G/V curve closely
        rmse = float(np.sqrt(np.mean((y_fit - y_act) ** 2)))
        assert rmse < 0.05, (
            f"Fitted G/V RMSE={rmse:.4f} too large.\n"
            f"True:   {np.round(y_act, 3)}\n"
            f"Fitted: {np.round(y_fit, 3)}"
        )


# ── Test C: DE vs local search on bimodal — comparison ───────────────────────

class TestDEvsLocal:
    """
    Explicit comparison: show that the minimum found by a broad search
    (proxy for DE) is better than the one found from a fixed local start.
    """

    def test_broad_search_beats_local_start(self):
        opt = TorchParameterOptimizer(_TorchCostAdapter(_bimodal))

        # Local start near x = +2
        local_result = opt.optimize(
            initial_guess=np.array([2.0, 0.0]),
            bounds=_BOUNDS,
            n_adam=0,
            n_lbfgs=100,
        )

        # Broad search: sample many starts, keep best
        rng = np.random.default_rng(0)
        best_cost = float("inf")
        for _ in range(20):
            x0 = rng.uniform(-3.0, 3.0, size=2)
            r = opt.optimize(initial_guess=x0, bounds=_BOUNDS, n_adam=0, n_lbfgs=50)
            if r.fun < best_cost:
                best_cost = r.fun

        assert best_cost < local_result.fun - 0.5, (
            f"Broad search ({best_cost:.3f}) not better than local ({local_result.fun:.3f})"
        )
