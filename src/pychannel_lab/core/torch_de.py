"""
CUDA-accelerated Differential Evolution + Adam + L-BFGS pipeline.

Pipeline
--------
1. DE/rand/1/bin (batched, CUDA)   — global exploration
2. Adam          (CUDA)            — fast gradient-descent warm-up
3. L-BFGS        (CUDA)            — tight local convergence

DE evaluates the entire population in a single batched matrix-exponential pass.
Adam and L-BFGS reuse TorchCostFunction / TorchParameterOptimizer from
torch_optimizer.py so gradient flow is identical to the standalone torch path.
"""

import math
from typing import Callable, Optional

import numpy as np
import torch

from .config import (
    EXP_FACTOR,
    INITIAL_CONDITIONS,
    G_K_MAX,
    ActivationConfig,
    InactivationConfig,
    CSInactivationConfig,
    RecoveryConfig,
)
from .protocols import (
    ActivationProtocol,
    InactivationProtocol,
    CSInactivationProtocol,
    RecoveryProtocol,
)
from .torch_simulator import (
    _get_basis_11,
    _precompute_dynamic_basis,
    get_device,
    preferred_dtype,
)
from .torch_optimizer import (
    OptimizeResult,
    TorchCostFunction,
    TorchParameterOptimizer,
)
from .param_transform import (
    to_raw as _pt_to_raw,
    from_raw as _pt_from_raw,
    build_log_mask,
    prior_penalty,
)

# ─── batched Q-matrix builders ────────────────────────────────────────────────


def build_Q_11state_batch(params: torch.Tensor, V: float) -> torch.Tensor:
    """
    Build batched Q matrices for the hardcoded 11-state Kv model.

    Parameters
    ----------
    params : (P, 11) — one parameter set per population member
    V      : float   — membrane voltage (mV)

    Returns
    -------
    Q : (P, 11, 11)   Q[p, j, i] = rate i→j for member p
    """
    EF = EXP_FACTOR
    Vt = float(V)

    a0, a1 = params[:, 0], params[:, 1]
    b0, b1 = params[:, 2], params[:, 3]
    kco0, kco1 = params[:, 4], params[:, 5]
    koc0, koc1 = params[:, 6], params[:, 7]
    kCI, kIC, f = params[:, 8], params[:, 9], params[:, 10]

    alpha = a0 * torch.exp(a1 * Vt * EF)
    beta = b0 * torch.exp(-b1 * Vt * EF)
    k_CO = kco0 * torch.exp(kco1 * Vt * EF)
    k_OC = koc0 * torch.exp(-koc1 * Vt * EF)

    # Order must match _TRANSITIONS_11 in torch_simulator.py (28 entries)
    rates = torch.stack(
        [
            4 * alpha,
            beta,  # C0 ↔ C1
            3 * alpha,
            2 * beta,  # C1 ↔ C2
            2 * alpha,
            3 * beta,  # C2 ↔ C3
            alpha,
            4 * beta,  # C3 ↔ C4
            4 * alpha / f,
            beta * f,  # I0 ↔ I1
            3 * alpha / f,
            2 * beta * f,  # I1 ↔ I2
            2 * alpha / f,
            3 * beta * f,  # I2 ↔ I3
            alpha / f,
            4 * beta * f,  # I3 ↔ I4
            kCI * f**4,
            kIC / f**4,  # C0 ↔ I0
            kCI * f**3,
            kIC / f**3,  # C1 ↔ I1
            kCI * f**2,
            kIC / f**2,  # C2 ↔ I2
            kCI * f,
            kIC / f,  # C3 ↔ I3
            kCI,
            kIC,  # C4 ↔ I4
            k_CO,
            k_OC,  # C4 ↔ O
        ],
        dim=1,
    )  # (P, 28)

    basis = _get_basis_11().to(device=params.device, dtype=params.dtype)  # (28, 11, 11)
    return torch.einsum("pt,tij->pij", rates, basis)  # (P, 11, 11)


def build_Q_dynamic_batch(
    params: torch.Tensor,
    V: float,
    msm_def,
    basis: torch.Tensor,
) -> torch.Tensor:
    """
    Build batched Q matrices for a user-defined MSMDefinition.

    Parameters
    ----------
    params : (P, n_params)
    V      : float
    msm_def : MSMDefinition
    basis  : (n_transitions, n_states, n_states) — precomputed constant basis

    Returns
    -------
    Q : (P, n_states, n_states)
    """
    device, dtype = params.device, params.dtype
    P = params.shape[0]
    EF_t = torch.tensor(EXP_FACTOR, device=device, dtype=dtype)
    Vt = torch.tensor(float(V), device=device, dtype=dtype)

    ns: dict = {
        "V": Vt,
        "EXP_FACTOR": EF_t,
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
        "abs": torch.abs,
        "pi": math.pi,
        "__builtins__": {},
    }
    for i, name in enumerate(msm_def.param_names):
        ns[name] = params[:, i]  # (P,) — vectorises over population

    rate_list = []
    for tr in msm_def.transitions:
        r = eval(tr.rate_expr, ns)  # noqa: S307
        if not isinstance(r, torch.Tensor):
            r = torch.full((P,), float(r), device=device, dtype=dtype)
        elif r.dim() == 0:
            r = r.expand(P)
        rate_list.append(r)

    rates = torch.stack(rate_list, dim=1)  # (P, T)
    return torch.einsum("pt,tij->pij", rates, basis.to(device=device, dtype=dtype))


# ─── frozen-parameter expansion (batched) ────────────────────────────────────


def _expand_batch(params_free: torch.Tensor, msm_def) -> torch.Tensor:
    """
    Insert frozen-parameter values into a batch of free-parameter vectors.

    Parameters
    ----------
    params_free : (P, n_free)
    msm_def     : MSMDefinition or None

    Returns
    -------
    (P, n_full)  — if no frozen params, returns params_free unchanged
    """
    if msm_def is None or not msm_def.frozen_indices:
        return params_free
    P = params_free.shape[0]
    device, dtype = params_free.device, params_free.dtype
    n_total = msm_def.n_params
    full = torch.zeros(P, n_total, dtype=dtype, device=device)
    fi = 0
    for i, p in enumerate(msm_def.parameters):
        if p.frozen:
            full[:, i] = float(p.initial_value)
        else:
            full[:, i] = params_free[:, fi]
            fi += 1
    return full


# ─── sigmoid / logit transforms (batched) ────────────────────────────────────
# Thin wrappers over the shared param_transform helpers so DE, Adam and L-BFGS
# all use exactly the same bounded<->raw mapping (linear or per-parameter log).


def _to_raw_batch(
    params: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, log_mask=None
) -> torch.Tensor:
    """Map (P, n) param tensor → raw space. lb/ub/log_mask : (n,)."""
    return _pt_to_raw(params, lb, ub, log_mask)


def _from_raw_batch(
    raw: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, log_mask=None
) -> torch.Tensor:
    """Map (P, n) raw tensor → param space. lb/ub/log_mask : (n,)."""
    return _pt_from_raw(raw, lb, ub, log_mask)


# ─── batched protocol simulator ──────────────────────────────────────────────


class BatchedProtocolSimulator:
    """
    Evaluates all P population members simultaneously.

    All matrix_exp calls operate on (P, n, n) tensors; bmm propagates
    (P, n) state vectors through the same batched transition matrices.

    Parameters
    ----------
    params    : (P, n_full_params) — one parameter set per member (fully expanded)
    t_total   : float              — total simulation time (s), needed for
                                     inactivation test-pulse duration and recovery.
    """

    def __init__(
        self,
        params: torch.Tensor,
        msm_def=None,
        act_cfg: ActivationConfig = None,
        inact_cfg: InactivationConfig = None,
        csi_cfg: CSInactivationConfig = None,
        rec_cfg: RecoveryConfig = None,
        g_k_max: float = G_K_MAX,
        t_total: float = 3.0,
        n_peak_steps: int = 50,
        initial_state=None,
    ):
        self.params = params
        self.P = params.shape[0]
        self.device = params.device
        self.dtype = params.dtype
        self.g_k_max = g_k_max
        self.t_total = t_total
        self.n_peak_steps = n_peak_steps
        self.msm_def = msm_def

        if msm_def is not None:
            self._open_idx = msm_def.open_state_indices
            s0 = (
                msm_def.default_initial_conditions
                if initial_state is None
                else np.asarray(initial_state, float)
            )
            self._dyn_basis, _ = _precompute_dynamic_basis(msm_def, self.dtype)
        else:
            self._open_idx = [10]
            s0 = (
                INITIAL_CONDITIONS
                if initial_state is None
                else np.asarray(initial_state, float)
            )
            self._dyn_basis = None

        s0_t = torch.tensor(s0, dtype=self.dtype, device=self.device)
        self.s0 = s0_t.unsqueeze(0).expand(self.P, -1).clone()  # (P, n_states)

        self.act_proto = ActivationProtocol(act_cfg)
        self.inact_proto = InactivationProtocol(inact_cfg)
        self.csi_proto = CSInactivationProtocol(csi_cfg)
        self.rec_proto = RecoveryProtocol(rec_cfg)

    # ── low-level batched primitives ──────────────────────────────────────────

    def _Q_batch(self, V: float) -> torch.Tensor:
        """(P, n, n) Q matrices at voltage V."""
        if self.msm_def is None:
            return build_Q_11state_batch(self.params, V)
        return build_Q_dynamic_batch(self.params, V, self.msm_def, self._dyn_basis)

    def _expm_batch(self, Qdt: torch.Tensor) -> torch.Tensor:
        """matrix_exp with MPS fallback. Input/output: (P, n, n)."""
        if Qdt.device.type == "mps":
            return torch.linalg.matrix_exp(Qdt.cpu()).to(Qdt.device)
        return torch.linalg.matrix_exp(Qdt)

    def _prop(self, P_batch: torch.Tensor, V: float, dt: float) -> torch.Tensor:
        """Exact one-shot propagation. P_batch: (P, n) → (P, n)."""
        M = self._expm_batch(self._Q_batch(V) * dt)  # (P, n, n)
        return torch.bmm(M, P_batch.unsqueeze(-1)).squeeze(-1)

    def _prop_peak(
        self, P_batch: torch.Tensor, V: float, total_dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate with n_peak_steps substeps, tracking peak open probability.

        Returns
        -------
        P_final : (P, n_states)
        peak    : (P,)  — max open probability over all substeps
        """
        n = self.n_peak_steps
        dt_sub = total_dt / n
        M = self._expm_batch(self._Q_batch(V) * dt_sub)  # (P, n, n) — computed once

        open_probs = [P_batch[:, self._open_idx].sum(dim=1)]  # initial
        cur = P_batch
        for _ in range(n):
            cur = torch.bmm(M, cur.unsqueeze(-1)).squeeze(-1)
            open_probs.append(cur[:, self._open_idx].sum(dim=1))

        peak = torch.stack(open_probs, dim=0).max(dim=0).values  # (P,)
        return cur, peak

    def _open(self, P_batch: torch.Tensor) -> torch.Tensor:
        """(P,) — total open probability for each member."""
        return P_batch[:, self._open_idx].sum(dim=1)

    # ── per-protocol MSE losses ───────────────────────────────────────────────

    def activation_loss(self, x_data: np.ndarray, y_data: np.ndarray) -> torch.Tensor:
        """Weighted-normalised MSE for all P members. Returns (P,)."""
        cfg = self.act_proto.cfg
        P_hold = self._prop(self.s0, cfg.v_hold, cfg.t_hold)

        peaks = []
        for V in x_data.tolist():
            _, pk = self._prop_peak(P_hold, float(V), cfg.t_test)
            peaks.append(pk)  # (P,) each

        cond = self.g_k_max * torch.stack(peaks, dim=0)  # (N_V, P)
        mx = cond.max(dim=0).values.clamp(min=1e-12)
        norm = cond / mx.unsqueeze(0)
        target = torch.tensor(y_data, dtype=self.dtype, device=self.device)  # (N_V,)
        return ((norm - target.unsqueeze(1)) ** 2).mean(dim=0)  # (P,)

    def inactivation_loss(self, x_data: np.ndarray, y_data: np.ndarray) -> torch.Tensor:
        cfg = self.inact_proto.cfg
        t_test_dur = max(self.t_total - cfg.t_hold - cfg.t_cond, 1e-3)
        P_hold = self._prop(self.s0, cfg.v_hold, cfg.t_hold)

        currents = []
        for V_cond in x_data.tolist():
            P_cond = self._prop(P_hold, float(V_cond), cfg.t_cond)
            baseline = self._open(P_cond)
            _, peak = self._prop_peak(P_cond, cfg.v_depo, t_test_dur)
            g = self.g_k_max * (peak - baseline)
            currents.append(g * (cfg.v_depo - cfg.v_hold))

        curr = torch.stack(currents, dim=0)  # (N_V, P)
        mx = curr.max(dim=0).values.clamp(min=1e-12)
        norm = curr / mx.unsqueeze(0)
        target = torch.tensor(y_data, dtype=self.dtype, device=self.device)
        return ((norm - target.unsqueeze(1)) ** 2).mean(dim=0)

    def cs_inactivation_loss(
        self, x_sim: np.ndarray, y_data: np.ndarray
    ) -> torch.Tensor:
        """x_sim: absolute pulse-end times in seconds (same units as config)."""
        cfg = self.csi_proto.cfg
        t_initial = cfg.t_initial

        _, baseline_peak = self._prop_peak(self.s0, cfg.v_hold, t_initial)
        baseline_peak = baseline_peak.clamp(min=1e-12)
        P_after = self._prop(self.s0, cfg.v_hold, t_initial)

        currents = []
        for t_pulse in x_sim.tolist():
            t_prep = float(t_pulse) - t_initial
            t_depo = max(cfg.t_test_end - float(t_pulse), 1e-3)
            P_prep = self._prop(P_after, cfg.v_prep, t_prep)
            _, pk = self._prop_peak(P_prep, cfg.v_depo, t_depo)
            g = self.g_k_max * pk / baseline_peak
            currents.append(g * (cfg.v_depo - cfg.v_prep))

        curr = torch.stack(currents, dim=0)  # (N_T, P)
        mx = curr.max(dim=0).values.clamp(min=1e-12)
        norm = curr / mx.unsqueeze(0)
        target = torch.tensor(y_data, dtype=self.dtype, device=self.device)
        return ((norm - target.unsqueeze(1)) ** 2).mean(dim=0)

    def recovery_loss(self, x_sim: np.ndarray, y_data: np.ndarray) -> torch.Tensor:
        """x_sim: absolute test-pulse start times in seconds."""
        cfg = self.rec_proto.cfg
        P0 = self._prop(self.s0, cfg.v_hold, cfg.t_prep)
        P_inact, g_pre = self._prop_peak(P0, cfg.v_depo, cfg.t_pulse - cfg.t_prep)
        g_pre = g_pre.clamp(min=1e-12)

        ratios = []
        for t_rec_end in x_sim.tolist():
            t_rec_dur = max(float(t_rec_end) - cfg.t_pulse, 0.0)
            t_test_dur = max(self.t_total - float(t_rec_end), 1e-3)
            P_rec = (
                self._prop(P_inact, cfg.v_hold, t_rec_dur) if t_rec_dur > 0 else P_inact
            )
            _, g_test = self._prop_peak(P_rec, cfg.v_depo, t_test_dur)
            ratios.append(g_test / g_pre)  # (P,)

        pred = torch.stack(ratios, dim=0)  # (N_T, P)
        target = torch.tensor(y_data, dtype=self.dtype, device=self.device)
        return ((pred - target.unsqueeze(1)) ** 2).mean(dim=0)

    # ── combined weighted loss ────────────────────────────────────────────────

    def total_loss(
        self,
        exp_data: dict,
        weights: dict,
        csi_cfg: CSInactivationConfig = None,
        rec_cfg: RecoveryConfig = None,
        reg_lambda: float = 0.0,
    ) -> torch.Tensor:
        """Weighted sum of protocol MSEs plus optional L2 regularization. Returns (P,)."""
        active = {k: v for k, v in exp_data.items() if v is not None}
        loss = torch.zeros(self.P, dtype=self.dtype, device=self.device)

        if "activation" in active:
            x, y, _ = active["activation"]
            loss = loss + weights.get("activation", 1.0) * self.activation_loss(
                np.asarray(x), np.asarray(y)
            )

        if "inactivation" in active:
            x, y, _ = active["inactivation"]
            loss = loss + weights.get("inactivation", 1.0) * self.inactivation_loss(
                np.asarray(x), np.asarray(y)
            )

        if "cs_inactivation" in active:
            x, y, _ = active["cs_inactivation"]
            _csi = csi_cfg or CSInactivationConfig()
            x_sim = _csi.t_initial + np.asarray(x) / 1000.0
            loss = loss + weights.get(
                "cs_inactivation", 2.0
            ) * self.cs_inactivation_loss(x_sim, np.asarray(y))

        if "recovery" in active:
            x, y, _ = active["recovery"]
            _rec = rec_cfg or RecoveryConfig()
            x_sim = _rec.t_pulse + np.asarray(x) / 1000.0
            loss = loss + weights.get("recovery", 2.0) * self.recovery_loss(
                x_sim, np.asarray(y)
            )

        # NOTE: regularization is applied in TorchDEOptimizer._eval_population
        # (which has the *free* params + prior), not here — this simulator only
        # sees the full expanded params.  reg_lambda is accepted for signature
        # compatibility but intentionally unused.
        return loss


# ─── DE/rand/1/bin optimizer ─────────────────────────────────────────────────


class TorchDEOptimizer:
    """
    Differential Evolution (DE/rand/1/bin) with batched population evaluation.

    The entire population is evaluated simultaneously via BatchedProtocolSimulator,
    so each generation requires only one batched matrix-exponential pass per
    voltage step rather than P independent passes.

    Parameters are mapped through a logit transform:
        param = lb + (ub - lb) * sigmoid(raw)
    so optimisation is unconstrained in raw space.
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
        g_k_max: float = G_K_MAX,
        t_total: float = 3.0,
        n_peak_steps: int = 50,
        device: torch.device = None,
        dtype: torch.dtype = None,
        reg_lambda: float = 0.0,
    ):
        self.exp = experimental_data
        self.w = weights or dict(self._DEFAULT_WEIGHTS)
        self._msm_def = msm_def
        self._act_cfg = act_cfg
        self._inact_cfg = inact_cfg
        self._csi_cfg = csi_cfg
        self._rec_cfg = rec_cfg
        self._g_k_max = g_k_max
        self._t_total = t_total
        self._n_peak_steps = n_peak_steps
        self.reg_lambda = reg_lambda
        self.device = device or get_device()
        self.dtype = dtype or preferred_dtype(self.device)
        # Prior (free initial guess) for the scale-invariant regularizer.
        if msm_def is not None:
            self._reg_prior = torch.tensor(
                msm_def.free_initial_guess, dtype=self.dtype, device=self.device
            )
        else:
            self._reg_prior = None

    def _eval_population(
        self, raw: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, log_mask=None
    ) -> torch.Tensor:
        """Evaluate (P,) fitness values for all members in the raw population."""
        params_free = _from_raw_batch(raw, lb, ub, log_mask)  # (P, n_free)
        params_full = _expand_batch(params_free, self._msm_def)  # (P, n_full)
        sim = BatchedProtocolSimulator(
            params_full,
            msm_def=self._msm_def,
            act_cfg=self._act_cfg,
            inact_cfg=self._inact_cfg,
            csi_cfg=self._csi_cfg,
            rec_cfg=self._rec_cfg,
            g_k_max=self._g_k_max,
            t_total=self._t_total,
            n_peak_steps=self._n_peak_steps,
        )
        loss = sim.total_loss(self.exp, self.w, self._csi_cfg, self._rec_cfg)
        if self.reg_lambda > 0.0 and self._reg_prior is not None:
            # Scale-invariant pull toward the prior, per population member (P,).
            loss = loss + self.reg_lambda * prior_penalty(
                params_free, self._reg_prior, lb, ub, log_mask
            )
        return loss

    def optimize(
        self,
        bounds: list,
        pop_size: int = 50,
        maxiter: int = 200,
        F: float = 0.8,
        CR: float = 0.9,
        progress_callback: Optional[Callable] = None,
        strategy: str = "adaptive",
        log_mask: Optional[list] = None,
        return_population: bool = False,
        p_best: float = 0.2,
    ):
        """
        Run Differential Evolution and return the best free-parameter vector.

        Parameters
        ----------
        bounds    : [(lb, ub), …]  one per free parameter
        pop_size  : population size P
        maxiter   : number of generations
        F         : differential weight (used only by the fixed 'rand1' strategy)
        CR        : crossover probability (used only by 'rand1')
        strategy  : 'adaptive' — JADE-style DE/current-to-pbest/1 with
                    self-adapting F/CR (default, more robust);
                    'rand1'    — classic fixed-F/CR DE/rand/1/bin.
        log_mask  : per-free-parameter log-scale flags (e.g.
                    ``msm_def.free_log_mask``).  None → inferred from bounds.
        return_population : if True, also return the whole final population
                    (free params) sorted best-first, for multi-start polishing.
        p_best    : fraction of the population treated as "pbest" élite in the
                    adaptive strategy.

        Returns
        -------
        best_params_free : np.ndarray (n_free,)
        or, if return_population: (best_params_free, population_free_sorted (P, n_free), fitness_sorted (P,))
        """
        device, dtype = self.device, self.dtype
        n = len(bounds)
        P = pop_size

        lb = torch.tensor([b[0] for b in bounds], dtype=dtype, device=device)
        ub = torch.tensor([b[1] for b in bounds], dtype=dtype, device=device)
        mask = build_log_mask(bounds, log_mask, device=device)

        # Initialise population.  For log-scaled parameters sample log-uniformly
        # so orders of magnitude are covered evenly; linear params stay uniform.
        u = torch.rand(P, n, dtype=dtype, device=device)
        lin_init = lb + (ub - lb) * u
        log_init = torch.exp(torch.log(lb.clamp_min(1e-30)) + (torch.log(ub) - torch.log(lb.clamp_min(1e-30))) * u)
        params_init = torch.where(mask, log_init, lin_init)
        raw = _to_raw_batch(params_init, lb, ub, mask)  # (P, n)

        with torch.no_grad():
            fitness = self._eval_population(raw, lb, ub, mask)  # (P,)
        fitness = torch.nan_to_num(fitness, nan=1e6, posinf=1e6)

        adaptive = strategy == "adaptive"
        mu_F, mu_CR = 0.5, 0.5  # JADE adaptation means
        n_pbest = max(2, int(round(p_best * P)))

        for gen in range(maxiter):
            if adaptive:
                # Per-member control parameters (JADE): F ~ Cauchy(mu_F,0.1),
                # CR ~ Normal(mu_CR,0.1).
                F_i = (
                    torch.distributions.Cauchy(mu_F, 0.1)
                    .sample((P,))
                    .to(device=device, dtype=dtype)
                    .clamp(0.05, 1.0)
                )
                CR_i = (
                    (mu_CR + 0.1 * torch.randn(P, device=device, dtype=dtype))
                    .clamp(0.0, 1.0)
                )
                # DE/current-to-pbest/1: pull toward a random élite member.
                elite_order = torch.argsort(fitness)[:n_pbest]
                pbest_pick = elite_order[torch.randint(n_pbest, (P,), device=device)]
                r1 = torch.randint(P, (P,), device=device)
                r2 = torch.randint(P, (P,), device=device)
                Fc = F_i.unsqueeze(1)
                mutant = (
                    raw + Fc * (raw[pbest_pick] - raw) + Fc * (raw[r1] - raw[r2])
                )
                cross = torch.rand(P, n, dtype=dtype, device=device) < CR_i.unsqueeze(1)
            else:
                # Classic DE/rand/1 with fixed F/CR.
                r1 = torch.randperm(P, device=device)
                r2 = torch.randperm(P, device=device)
                r3 = torch.randperm(P, device=device)
                mutant = raw[r1] + F * (raw[r2] - raw[r3])
                cross = torch.rand(P, n, dtype=dtype, device=device) < CR

            # Guarantee at least one mutant gene per trial.
            j_rand = torch.randint(n, (P,), device=device)
            cross[torch.arange(P, device=device), j_rand] = True
            trial = torch.where(cross, mutant, raw)  # (P, n)

            with torch.no_grad():
                trial_fitness = self._eval_population(trial, lb, ub, mask)  # (P,)
            trial_fitness = torch.nan_to_num(trial_fitness, nan=1e6, posinf=1e6)

            # Selection: keep trial if it improves fitness.
            better = trial_fitness < fitness
            raw = torch.where(better.unsqueeze(1), trial, raw)
            fitness = torch.where(better, trial_fitness, fitness)

            if adaptive and bool(better.any()):
                # Update adaptation means from the successful control params.
                succ_F = F_i[better]
                succ_CR = CR_i[better]
                # Lehmer mean for F, arithmetic mean for CR (JADE).
                lehmer = float((succ_F ** 2).sum() / succ_F.sum().clamp_min(1e-12))
                mu_F = 0.9 * mu_F + 0.1 * lehmer
                mu_CR = 0.9 * mu_CR + 0.1 * float(succ_CR.mean())

            if progress_callback and (gen % 5 == 0 or gen == maxiter - 1):
                progress_callback(gen + 1, float(fitness.min()), None)

        order = torch.argsort(fitness)
        best_free = _from_raw_batch(raw[order[:1]], lb, ub, mask).squeeze(0)
        if not return_population:
            return best_free.cpu().numpy()
        pop_sorted = _from_raw_batch(raw[order], lb, ub, mask)  # (P, n)
        return (
            best_free.cpu().numpy(),
            pop_sorted.cpu().numpy(),
            fitness[order].cpu().numpy(),
        )


# ─── full pipeline: DE → Adam → L-BFGS ──────────────────────────────────────


class TorchPipelineOptimizer:
    """
    Orchestrates the three-phase optimisation pipeline:

    1. TorchDEOptimizer   — batched global search
    2. Adam               — gradient-based warm-up from best DE solution
    3. L-BFGS             — tight local convergence

    The progress_callback receives absolute iteration numbers across all phases
    so that iteration ≤ de_maxiter is DE, ≤ de_maxiter+n_adam is Adam, and
    beyond that is L-BFGS.
    """

    def __init__(
        self,
        experimental_data: dict,
        weights: dict = None,
        msm_def=None,
        act_cfg: ActivationConfig = None,
        inact_cfg: InactivationConfig = None,
        csi_cfg: CSInactivationConfig = None,
        rec_cfg: RecoveryConfig = None,
        g_k_max: float = G_K_MAX,
        t_total: float = 3.0,
        n_peak_steps: int = 50,
        device: torch.device = None,
        dtype: torch.dtype = None,
        reg_lambda: float = 0.0,
    ):
        device = device or get_device()
        dtype = dtype or preferred_dtype(device)

        _common = dict(
            experimental_data=experimental_data,
            weights=weights,
            msm_def=msm_def,
            act_cfg=act_cfg,
            inact_cfg=inact_cfg,
            csi_cfg=csi_cfg,
            rec_cfg=rec_cfg,
            g_k_max=g_k_max,
            t_total=t_total,
            n_peak_steps=n_peak_steps,
            device=device,
            dtype=dtype,
            reg_lambda=reg_lambda,
        )

        self._de = TorchDEOptimizer(**_common)

        # Local phases reuse the existing differentiable cost function
        _local_kw = dict(
            experimental_data=experimental_data,
            weights=weights,
            msm_def=msm_def,
            act_cfg=act_cfg,
            inact_cfg=inact_cfg,
            csi_cfg=csi_cfg,
            rec_cfg=rec_cfg,
            g_k_max=g_k_max,
            t_total=t_total,
            n_peak_steps=n_peak_steps,
            device=device,
            dtype=dtype,
            reg_lambda=reg_lambda,
        )
        self._torch_cost = TorchCostFunction(**_local_kw)
        self._local = TorchParameterOptimizer(self._torch_cost)
        self._msm_def = msm_def

    def optimize(
        self,
        bounds: list,
        pop_size: int = 50,
        de_maxiter: int = 200,
        F: float = 0.8,
        CR: float = 0.9,
        n_adam: int = 500,
        adam_lr: float = 0.05,
        n_lbfgs: int = 200,
        progress_callback: Optional[Callable] = None,
        skip_de: bool = False,
        initial_params: Optional[np.ndarray] = None,
        strategy: str = "adaptive",
        n_starts: int = 5,
    ) -> OptimizeResult:
        """
        Run the full pipeline.

        Parameters
        ----------
        bounds         : [(lb, ub), …] for free parameters only
        pop_size       : DE population size
        de_maxiter     : DE generations
        F              : DE mutation scale ('rand1' strategy only)
        CR             : DE crossover probability ('rand1' strategy only)
        n_adam         : Adam steps
        adam_lr        : Adam learning rate
        n_lbfgs        : L-BFGS outer steps
        skip_de        : if True, skip DE and start local optimisation from
                         *initial_params* (or the MSM initial guess if None)
        initial_params : starting point when skip_de=True (free params, numpy array)
        strategy       : DE strategy, 'adaptive' (JADE) or 'rand1' (classic).
        n_starts       : number of top DE members to polish with Adam→L-BFGS;
                         the lowest-cost polished result wins.  n_starts=1 keeps
                         the old best-only behaviour.

        Returns
        -------
        OptimizeResult with .x (best free params), .fun, .nit
        """
        de_gen = [0]
        # Log-scale mask from the MSM (positive-lower-bound free params); the DE
        # and local optimisers infer it from the bounds when no MSM is present.
        log_mask = self._msm_def.free_log_mask if self._msm_def is not None else None

        # Candidate starting points for the local phase (best-first).
        starts = []

        if skip_de:
            if initial_params is None:
                raise ValueError("initial_params must be provided when skip_de=True")
            starts = [np.asarray(initial_params, dtype=float)]
        else:

            def _de_cb(iteration: int, cost: float, convergence):
                de_gen[0] = iteration
                if progress_callback:
                    progress_callback(iteration, cost, convergence)

            # ── Phase 1: DE ──────────────────────────────────────────────────
            best_free, pop_sorted, _fit = self._de.optimize(
                bounds=bounds,
                pop_size=pop_size,
                maxiter=de_maxiter,
                F=F,
                CR=CR,
                strategy=strategy,
                log_mask=log_mask,
                return_population=True,
                progress_callback=_de_cb,
            )
            k = max(1, min(int(n_starts), pop_sorted.shape[0]))
            starts = [pop_sorted[i] for i in range(k)]

        # ── Phases 2+3: Adam → L-BFGS, polished from each candidate start ────
        def _local_cb(iteration: int, cost: float, convergence):
            if progress_callback:
                progress_callback(de_gen[0] + iteration, cost, convergence)

        best_result = None
        for si, x0 in enumerate(starts):
            # Only the first (best) start streams progress, to keep the log clean.
            cb = _local_cb if si == 0 else None
            res = self._local.optimize(
                initial_guess=x0,
                bounds=bounds,
                n_adam=n_adam,
                adam_lr=adam_lr,
                n_lbfgs=n_lbfgs,
                progress_callback=cb,
                log_mask=log_mask,
            )
            if best_result is None or res.fun < best_result.fun:
                best_result = res

        return best_result

    def cost_breakdown(self, params_free_np: np.ndarray) -> dict:
        """Per-protocol cost breakdown for the given free-parameter vector."""
        return self._local.cost_breakdown(params_free_np)
