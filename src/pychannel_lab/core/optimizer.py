"""
Cost function and parameter optimiser.

Works with either the default 11-state model or a user-defined MSMDefinition.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Callable, Tuple

from .config import (
    ActivationConfig,
    InactivationConfig,
    CSInactivationConfig,
    RecoveryConfig,
)
from .simulator import ProtocolSimulator, stationary_distribution
from .param_transform import prior_penalty_np


class CostFunction:
    """
    Weighted sum of MSE between simulated and experimental curves.

    Parameters
    ----------
    experimental_data : dict
        Keys: 'activation', 'inactivation', 'cs_inactivation', 'recovery'
        Values: tuple (x_data, y_data, y_err) of numpy arrays, or None to skip.
    weights : dict, optional
        Per-protocol weights.
    msm_def : MSMDefinition, optional
        If provided, the optimiser uses the dynamic model.
    act_cfg … rec_cfg : protocol config dataclasses
    g_k_max, t_total, dt : simulation settings
    """

    _DEFAULT_WEIGHTS = {
        "activation": 1.0,
        "inactivation": 1.0,
        "cs_inactivation": 2.0,
        "recovery": 2.0,
    }

    def __init__(
        self,
        experimental_data: dict,
        weights: dict = None,
        msm_def=None,
        act_cfg: ActivationConfig = None,
        inact_cfg: InactivationConfig = None,
        csi_cfg: CSInactivationConfig = None,
        rec_cfg: RecoveryConfig = None,
        custom_protocols: list = None,  # list[CustomProtocolConfig]
        g_k_max: float = None,
        t_total: float = None,
        dt: float = None,
        reg_lambda: float = 0.0,
        rest_lambda: float = 0.0,
        v_rest: float = -90.0,
    ):
        self.exp = experimental_data
        self.w = weights or self._DEFAULT_WEIGHTS
        self.reg_lambda = reg_lambda
        # Resting-inactivation penalty, matching the torch cost functions.
        self.rest_lambda = rest_lambda
        self.v_rest = v_rest
        self._inact_idx = (
            [i for i, s in enumerate(msm_def.states) if s.state_type == "inactivated"]
            if msm_def is not None
            else []
        )
        self._msm_def = msm_def
        self._act_cfg = act_cfg
        self._inact_cfg = inact_cfg
        self._csi_cfg = csi_cfg
        self._rec_cfg = rec_cfg
        self._custom_protocols = custom_protocols or []
        self._g_k_max = g_k_max
        self._t_total = t_total
        self._dt = dt

    # ------------------------------------------------------------------

    def _expand(self, parameters: np.ndarray) -> np.ndarray:
        """Expand free-parameter vector to full length by inserting frozen values."""
        if self._msm_def is None or not self._msm_def.frozen_indices:
            return parameters
        return self._msm_def.expand_params(parameters)

    def _build_sim(self, parameters: np.ndarray) -> ProtocolSimulator:
        kw = dict(
            msm_def=self._msm_def,
            act_cfg=self._act_cfg,
            inact_cfg=self._inact_cfg,
            csi_cfg=self._csi_cfg,
            rec_cfg=self._rec_cfg,
        )
        if self._g_k_max is not None:
            kw["g_k_max"] = self._g_k_max
        if self._t_total is not None:
            kw["t_total"] = self._t_total
        if self._dt is not None:
            kw["dt"] = self._dt
        return ProtocolSimulator(parameters, **kw)

    @staticmethod
    def _mse(predicted: np.ndarray, observed: np.ndarray) -> float:
        return float(np.mean((predicted - observed) ** 2))

    # ------------------------------------------------------------------

    def individual_costs(self, parameters: np.ndarray) -> dict:
        sim = self._build_sim(self._expand(parameters))
        active = {k: v for k, v in self.exp.items() if v is not None}
        costs = {}

        if "activation" in active:
            x, y, _ = active["activation"]
            costs["activation"] = self._mse(sim.run_activation(x), y)

        if "inactivation" in active:
            x, y, _ = active["inactivation"]
            costs["inactivation"] = self._mse(sim.run_inactivation(x), y)

        if "cs_inactivation" in active:
            x, y, _ = active["cs_inactivation"]
            csi_cfg = self._csi_cfg or CSInactivationConfig()
            x_sim = csi_cfg.t_initial + np.asarray(x) / 1000.0
            costs["cs_inactivation"] = self._mse(sim.run_cs_inactivation(x_sim), y)

        if "recovery" in active:
            x, y, _ = active["recovery"]
            rec_cfg = self._rec_cfg or RecoveryConfig()
            x_sim = rec_cfg.t_pulse + np.asarray(x) / 1000.0
            costs["recovery"] = self._mse(sim.run_recovery(x_sim), y)

        for i, cfg in enumerate(self._custom_protocols):
            if not cfg.enabled:
                continue
            key = f"custom_{i}"
            if key in active:
                x_exp, y_exp, _ = active[key]
                x_arr = np.asarray(x_exp, dtype=float)
                _, y_sim = sim.run_custom_protocol(cfg, sweep_vals=x_arr)
                costs[key] = self._mse(y_sim, np.asarray(y_exp, dtype=float))

        return costs

    def total_cost(self, parameters: np.ndarray) -> float:
        costs = self.individual_costs(parameters)
        data_cost = sum(self.w.get(k, 1.0) * v for k, v in costs.items())
        reg_cost = 0.0
        if self.reg_lambda > 0.0 and self._msm_def is not None:
            # Scale-invariant pull toward the prior (initial guess), matching the
            # torch path; `parameters` is the free-length vector.
            fb = self._msm_def.free_bounds
            reg_cost = self.reg_lambda * prior_penalty_np(
                parameters,
                self._msm_def.free_initial_guess,
                [b[0] for b in fb],
                [b[1] for b in fb],
                self._msm_def.free_log_mask,
            )
        rest_cost = 0.0
        if self.rest_lambda > 0.0 and self._inact_idx:
            sim = self._build_sim(self._expand(parameters))
            p_rest = stationary_distribution(sim._build_Q(self.v_rest))
            rest_cost = self.rest_lambda * float(p_rest[self._inact_idx].sum()) ** 2
        return data_cost + reg_cost + rest_cost

    def __call__(self, parameters: np.ndarray) -> float:
        return self.total_cost(parameters)


# ---------------------------------------------------------------------------


class ParameterOptimizer:
    """Wraps differential_evolution (global) and L-BFGS-B (local)."""

    def __init__(self, cost_function: CostFunction):
        self.cost = cost_function

    # ------------------------------------------------------------------

    def optimize_global(
        self,
        bounds: tuple,
        maxiter: int = 5000,
        workers: int = -1,
        progress_callback: Callable = None,
    ):
        iteration = [0]

        def _cb(xk, convergence):
            iteration[0] += 1
            if progress_callback:
                cost = self.cost.total_cost(xk)
                progress_callback(iteration[0], cost, convergence)

        return differential_evolution(
            self.cost,
            bounds=bounds,
            maxiter=maxiter,
            workers=workers,
            callback=_cb,
            seed=42,
            polish=True,
        )

    def optimize_local(
        self,
        initial_guess: np.ndarray,
        bounds: tuple,
        maxiter: int = 15000,
        method: str = "L-BFGS-B",
        progress_callback: Callable = None,
    ):
        iteration = [0]

        def _cb(xk):
            iteration[0] += 1
            if progress_callback:
                cost = self.cost.total_cost(xk)
                progress_callback(iteration[0], cost, None)

        return minimize(
            self.cost,
            initial_guess,
            bounds=bounds,
            method=method,
            callback=_cb,
            options={"maxiter": maxiter, "maxfev": 50000},
        )

    def cost_breakdown(self, parameters: np.ndarray) -> dict:
        ind = self.cost.individual_costs(parameters)
        ind["total"] = self.cost.total_cost(parameters)
        return ind
