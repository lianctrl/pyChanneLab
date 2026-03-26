"""
Cost function and parameter optimiser.

Works with either the default 11-state model or a user-defined MSMDefinition.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Callable, Tuple

from core.config import (
    ActivationConfig,
    InactivationConfig,
    CSInactivationConfig,
    RecoveryConfig,
)
from core.simulator import ProtocolSimulator


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
        g_k_max: float = None,
        t_total: float = None,
        dt: float = None,
    ):
        self.exp = experimental_data
        self.w = weights or self._DEFAULT_WEIGHTS
        self._msm_def = msm_def
        self._act_cfg = act_cfg
        self._inact_cfg = inact_cfg
        self._csi_cfg = csi_cfg
        self._rec_cfg = rec_cfg
        self._g_k_max = g_k_max
        self._t_total = t_total
        self._dt = dt

    # ------------------------------------------------------------------

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
        sim = self._build_sim(parameters)
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

        return costs

    def total_cost(self, parameters: np.ndarray) -> float:
        costs = self.individual_costs(parameters)
        return sum(self.w.get(k, 1.0) * v for k, v in costs.items())

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
