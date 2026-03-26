"""
PyTorch Adam + L-BFGS optimiser for ion channel MSMs.

Strategy
--------
1. Adam phase   — fast gradient-descent exploration of the loss landscape.
2. L-BFGS phase — second-order convergence to a tight local minimum.

Bounds are enforced via a sigmoid (logistic) transform on unconstrained raw
parameters, so the optimisation is unconstrained in raw space:

    param = lb + (ub - lb) * sigmoid(raw)

This is fully differentiable and avoids projection / clamping hacks.
"""

import numpy as np
import torch
from typing import Callable, Optional

from core.config import (
    ActivationConfig, InactivationConfig,
    CSInactivationConfig, RecoveryConfig,
    G_K_MAX, TIME_PARAMS,
)
from core.torch_simulator import (
    TorchProtocolSimulator, get_device, preferred_dtype,
)


class OptimizeResult:
    """Attribute- and dict-style access to optimisation results."""

    def __init__(self, x, fun, nit, success=True):
        self.x       = x        # np.ndarray — best parameters
        self.fun     = fun      # float — best cost
        self.nit     = nit      # int  — total iterations
        self.success = success

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        return (f"OptimizeResult(fun={self.fun:.6g}, nit={self.nit}, "
                f"success={self.success})")


class TorchCostFunction:
    """
    Weighted MSE between matrix-exponential simulations and experimental data.

    Calling an instance with a torch.Tensor of parameters returns a
    differentiable scalar loss that can be backpropagated.
    """

    _DEFAULT_WEIGHTS = {
        "activation":      1.0,
        "inactivation":    1.0,
        "cs_inactivation": 2.0,
        "recovery":        2.0,
    }

    def __init__(
        self,
        experimental_data: dict,
        weights:   dict                 = None,
        msm_def                         = None,
        act_cfg:   ActivationConfig     = None,
        inact_cfg: InactivationConfig   = None,
        csi_cfg:   CSInactivationConfig = None,
        rec_cfg:   RecoveryConfig       = None,
        g_k_max:   float                = G_K_MAX,
        t_total:   float                = TIME_PARAMS["tend"],
        n_peak_steps: int               = 50,
        device:    torch.device         = None,
        dtype:     torch.dtype          = None,
    ):
        self.exp          = experimental_data
        self.w            = weights or self._DEFAULT_WEIGHTS
        self._msm_def     = msm_def
        self._act_cfg     = act_cfg
        self._inact_cfg   = inact_cfg
        self._csi_cfg     = csi_cfg
        self._rec_cfg     = rec_cfg
        self._g_k_max     = g_k_max
        self._t_total     = t_total
        self._n_peak_steps = n_peak_steps
        self.device       = device or get_device()
        self.dtype        = dtype or preferred_dtype(self.device)

    # ------------------------------------------------------------------

    def _build_sim(self, params: torch.Tensor) -> TorchProtocolSimulator:
        return TorchProtocolSimulator(
            params,
            msm_def      = self._msm_def,
            act_cfg      = self._act_cfg,
            inact_cfg    = self._inact_cfg,
            csi_cfg      = self._csi_cfg,
            rec_cfg      = self._rec_cfg,
            g_k_max      = self._g_k_max,
            t_total      = self._t_total,
            n_peak_steps = self._n_peak_steps,
        )

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        """Return weighted MSE as a differentiable scalar."""
        sim    = self._build_sim(params)
        loss   = params.new_zeros(())          # scalar, same device/dtype
        active = {k: v for k, v in self.exp.items() if v is not None}

        if "activation" in active:
            x, y, _ = active["activation"]
            pred   = sim.run_activation(x)
            target = torch.tensor(y, dtype=params.dtype, device=params.device)
            loss   = loss + self.w.get("activation", 1.0) * ((pred - target) ** 2).mean()

        if "inactivation" in active:
            x, y, _ = active["inactivation"]
            pred   = sim.run_inactivation(x)
            target = torch.tensor(y, dtype=params.dtype, device=params.device)
            loss   = loss + self.w.get("inactivation", 1.0) * ((pred - target) ** 2).mean()

        if "cs_inactivation" in active:
            x, y, _ = active["cs_inactivation"]
            _csi = self._csi_cfg or CSInactivationConfig()
            x_sim = _csi.t_initial + np.asarray(x) / 1000.0
            pred   = sim.run_cs_inactivation(x_sim)
            target = torch.tensor(y, dtype=params.dtype, device=params.device)
            loss   = loss + self.w.get("cs_inactivation", 1.0) * ((pred - target) ** 2).mean()

        if "recovery" in active:
            x, y, _ = active["recovery"]
            _rec = self._rec_cfg or RecoveryConfig()
            x_sim = _rec.t_pulse + np.asarray(x) / 1000.0
            pred   = sim.run_recovery(x_sim)
            target = torch.tensor(y, dtype=params.dtype, device=params.device)
            loss   = loss + self.w.get("recovery", 1.0) * ((pred - target) ** 2).mean()

        return loss

    def total_cost_numpy(self, params_np: np.ndarray) -> float:
        params = torch.tensor(params_np, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            return float(self(params))

    def individual_costs_numpy(self, params_np: np.ndarray) -> dict:
        params = torch.tensor(params_np, dtype=self.dtype, device=self.device)
        sim    = self._build_sim(params)
        active = {k: v for k, v in self.exp.items() if v is not None}
        costs  = {}

        with torch.no_grad():
            def _mse(pred, y_np):
                t = torch.tensor(y_np, dtype=self.dtype, device=self.device)
                return float(((pred - t) ** 2).mean())

            if "activation" in active:
                x, y, _ = active["activation"]
                costs["activation"] = _mse(sim.run_activation(x), y)

            if "inactivation" in active:
                x, y, _ = active["inactivation"]
                costs["inactivation"] = _mse(sim.run_inactivation(x), y)

            if "cs_inactivation" in active:
                x, y, _ = active["cs_inactivation"]
                _csi = self._csi_cfg or CSInactivationConfig()
                x_sim = _csi.t_initial + np.asarray(x) / 1000.0
                costs["cs_inactivation"] = _mse(sim.run_cs_inactivation(x_sim), y)

            if "recovery" in active:
                x, y, _ = active["recovery"]
                _rec = self._rec_cfg or RecoveryConfig()
                x_sim = _rec.t_pulse + np.asarray(x) / 1000.0
                costs["recovery"] = _mse(sim.run_recovery(x_sim), y)

        return costs


class TorchParameterOptimizer:
    """
    Adam (warm-up) → L-BFGS (refinement) optimiser.

    Parameters are transformed to an unconstrained raw space via
        raw = logit((p − lb) / (ub − lb))
    and recovered via
        p = lb + (ub − lb) × sigmoid(raw)
    so that gradient steps never violate the parameter bounds.
    """

    def __init__(self, cost_function: TorchCostFunction):
        self.cost = cost_function

    @staticmethod
    def _to_raw(
        params_np: np.ndarray,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.tensor(params_np, dtype=lb.dtype, device=lb.device)
        normalized = ((p - lb) / (ub - lb)).clamp(0.001, 0.999)
        return torch.logit(normalized)

    @staticmethod
    def _from_raw(raw: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        return lb + (ub - lb) * torch.sigmoid(raw)

    def optimize(
        self,
        initial_guess:     np.ndarray,
        bounds:            list,
        n_adam:            int            = 500,
        adam_lr:           float          = 0.05,
        n_lbfgs:           int            = 200,
        progress_callback: Optional[Callable] = None,
    ) -> OptimizeResult:
        """
        Run Adam followed by L-BFGS.

        Parameters
        ----------
        initial_guess : shape (n_params,) numpy array
        bounds        : list of (lb, ub) tuples
        n_adam        : number of Adam steps
        adam_lr       : Adam learning rate
        n_lbfgs       : number of L-BFGS outer steps (each runs up to 20 inner)
        progress_callback : callable(iteration, cost, convergence) or None

        Returns
        -------
        OptimizeResult with .x, .fun, .nit
        """
        device = self.cost.device
        dtype  = self.cost.dtype

        lb = torch.tensor([b[0] for b in bounds], dtype=dtype, device=device)
        ub = torch.tensor([b[1] for b in bounds], dtype=dtype, device=device)

        raw = self._to_raw(initial_guess, lb, ub).detach().requires_grad_(True)

        def get_params() -> torch.Tensor:
            return self._from_raw(raw, lb, ub)

        def compute_loss() -> torch.Tensor:
            return self.cost(get_params())

        total_iters = [0]

        if n_adam > 0:
            adam = torch.optim.Adam([raw], lr=adam_lr)
            for i in range(n_adam):
                adam.zero_grad()
                loss = compute_loss()
                loss.backward()
                adam.step()
                total_iters[0] += 1
                if progress_callback and (i % 10 == 0 or i == n_adam - 1):
                    progress_callback(total_iters[0], loss.detach().item(), None)

        if n_lbfgs > 0:
            lbfgs = torch.optim.LBFGS(
                [raw],
                lr=1.0,
                max_iter=20,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                line_search_fn="strong_wolfe",
            )
            last_loss = [float("nan")]

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                loss = compute_loss()
                loss.backward()
                last_loss[0] = loss.detach().item()
                return loss

            for outer in range(n_lbfgs):
                lbfgs.step(closure)
                total_iters[0] += 1
                if progress_callback and (outer % 5 == 0 or outer == n_lbfgs - 1):
                    progress_callback(total_iters[0], last_loss[0], None)

        final_params = get_params().detach().cpu().numpy()
        final_cost   = self.cost.total_cost_numpy(final_params)

        return OptimizeResult(
            x       = final_params,
            fun     = final_cost,
            nit     = total_iters[0],
            success = True,
        )

    def cost_breakdown(self, params_np: np.ndarray) -> dict:
        ind = self.cost.individual_costs_numpy(params_np)
        ind["total"] = self.cost.total_cost_numpy(params_np)
        return ind