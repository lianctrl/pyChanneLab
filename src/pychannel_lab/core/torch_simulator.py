"""
PyTorch matrix-exponential simulator for ion channel MSMs.

For each piecewise-constant voltage segment the exact solution is:
    P(t + dt) = expm(Q(V) · dt) @ P(t)

No numerical integration error; fully differentiable w.r.t. parameters.
"""

import math
from typing import Optional

import numpy as np
import torch

from core.config import (
    EXP_FACTOR,
    INITIAL_CONDITIONS,
    G_K_MAX,
    ActivationConfig,
    InactivationConfig,
    CSInactivationConfig,
    RecoveryConfig,
)
from core.protocols import (
    ActivationProtocol,
    InactivationProtocol,
    CSInactivationProtocol,
    RecoveryProtocol,
)


def get_device() -> torch.device:
    """Return best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preferred_dtype(device: torch.device) -> torch.dtype:
    """MPS only supports float32; use float64 everywhere else."""
    if device.type == "mps":
        return torch.float32
    return torch.float64


# Topology of the hardcoded 11-state model.
# Order must match the `rates` vector in build_Q_11state().
_TRANSITIONS_11 = [
    (0, 1),
    (1, 0),  # C0 ↔ C1
    (1, 2),
    (2, 1),  # C1 ↔ C2
    (2, 3),
    (3, 2),  # C2 ↔ C3
    (3, 4),
    (4, 3),  # C3 ↔ C4
    (5, 6),
    (6, 5),  # I0 ↔ I1
    (6, 7),
    (7, 6),  # I1 ↔ I2
    (7, 8),
    (8, 7),  # I2 ↔ I3
    (8, 9),
    (9, 8),  # I3 ↔ I4
    (0, 5),
    (5, 0),  # C0 ↔ I0
    (1, 6),
    (6, 1),  # C1 ↔ I1
    (2, 7),
    (7, 2),  # C2 ↔ I2
    (3, 8),
    (8, 3),  # C3 ↔ I3
    (4, 9),
    (9, 4),  # C4 ↔ I4
    (4, 10),
    (10, 4),  # C4 ↔ O
]

_BASIS_11: Optional[torch.Tensor] = None


def _get_basis_11() -> torch.Tensor:
    """Return the precomputed (28, 11, 11) constant basis tensor."""
    global _BASIS_11
    if _BASIS_11 is None:
        n = 11
        T = len(_TRANSITIONS_11)
        basis = torch.zeros(T, n, n, dtype=torch.float64)
        for t, (fr, to) in enumerate(_TRANSITIONS_11):
            basis[t, to, fr] = 1.0  # off-diagonal: flux i→j
            basis[t, fr, fr] = -1.0  # diagonal: loss from i
        _BASIS_11 = basis
    return _BASIS_11


def build_Q_11state(params: torch.Tensor, V: float) -> torch.Tensor:
    """
    Build the generator matrix Q for the hardcoded 11-state Kv model.

    State order: C0=0 … C4=4, I0=5 … I4=9, O=10
    Q[j, i] = rate(i→j) for i≠j;  Q[i, i] = −(sum of outgoing rates from i)
    Fully differentiable w.r.t. params.
    """
    device, dtype = params.device, params.dtype
    EF = EXP_FACTOR  # Python float — no gradient needed
    Vt = float(V)

    a0, a1, b0, b1, kco0, kco1, koc0, koc1, kCI, kIC, f = (params[i] for i in range(11))

    alpha = a0 * torch.exp(a1 * Vt * EF)
    beta = b0 * torch.exp(-b1 * Vt * EF)
    k_CO = kco0 * torch.exp(kco1 * Vt * EF)
    k_OC = koc0 * torch.exp(-koc1 * Vt * EF)

    # Must match _TRANSITIONS_11 ordering
    rates = torch.stack(
        [
            4 * alpha,
            beta,  # C0↔C1
            3 * alpha,
            2 * beta,  # C1↔C2
            2 * alpha,
            3 * beta,  # C2↔C3
            alpha,
            4 * beta,  # C3↔C4
            4 * alpha / f,
            beta * f,  # I0↔I1
            3 * alpha / f,
            2 * beta * f,  # I1↔I2
            2 * alpha / f,
            3 * beta * f,  # I2↔I3
            alpha / f,
            4 * beta * f,  # I3↔I4
            kCI * f**4,
            kIC / f**4,  # C0↔I0
            kCI * f**3,
            kIC / f**3,  # C1↔I1
            kCI * f**2,
            kIC / f**2,  # C2↔I2
            kCI * f,
            kIC / f,  # C3↔I3
            kCI,
            kIC,  # C4↔I4
            k_CO,
            k_OC,  # C4↔O
        ]
    )  # shape: (28,)

    basis = _get_basis_11().to(device=device, dtype=dtype)
    return torch.einsum("t,tij->ij", rates, basis)  # (11, 11)


def _precompute_dynamic_basis(msm_def, dtype: torch.dtype) -> tuple[torch.Tensor, dict]:
    """
    Build constant basis matrices for a user-defined MSMDefinition.

    Returns
    -------
    basis : Tensor shape (n_transitions, n_states, n_states)
    state_idx : dict mapping state name → column index
    """
    n = msm_def.n_states
    state_idx = {s: i for i, s in enumerate(msm_def.state_names)}
    T = len(msm_def.transitions)
    basis = torch.zeros(T, n, n, dtype=dtype)
    for t, tr in enumerate(msm_def.transitions):
        fr = state_idx[tr.from_state]
        to = state_idx[tr.to_state]
        basis[t, to, fr] = 1.0
        basis[t, fr, fr] = -1.0
    return basis, state_idx


def build_Q_dynamic(
    params: torch.Tensor,
    V: float,
    msm_def,
    basis: torch.Tensor,
) -> torch.Tensor:
    """
    Build Q for a user-defined MSMDefinition.

    Rate expressions are evaluated with torch operations in scope, so autograd
    flows through the exponentials and products in user-written formulas.
    """
    device, dtype = params.device, params.dtype
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
    for name, val in zip(msm_def.param_names, params):
        ns[name] = val

    rate_list = []
    for tr in msm_def.transitions:
        r = eval(tr.rate_expr, ns)
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(float(r), device=device, dtype=dtype)
        rate_list.append(r)

    rates = torch.stack(rate_list)  # vectorize the list of rates
    return torch.einsum("t,tij->ij", rates, basis.to(device=device))


class TorchProtocolSimulator:
    """
    Matrix-exponential simulator for all four voltage-clamp protocols.

    Equilibration phases (long, constant V) use a single matrix_exp call —
    exact with no step-size error.  Measurement phases use n_peak_steps
    substeps so that the peak open probability can be tracked.

    Parameters
    ----------
    params : torch.Tensor  — kinetic parameters, requires_grad for autograd.
    msm_def : MSMDefinition, optional — custom topology; None → 11-state model.
    n_peak_steps : int — temporal resolution inside measurement phases.
    t_total : float — total simulation time (s), needed for inactivation / recovery.
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
        initial_state=None,
        t_total: float = 3.0,
        n_peak_steps: int = 50,
    ):
        self.params = params
        self.msm_def = msm_def
        self.g_k_max = g_k_max
        self.t_total = t_total
        self.n_peak_steps = n_peak_steps
        self.device = params.device
        self.dtype = params.dtype

        if msm_def is not None:
            self._open_idx = msm_def.open_state_indices
            s0 = (
                msm_def.default_initial_conditions
                if initial_state is None
                else np.asarray(initial_state, dtype=float)
            )
            self._dyn_basis, _ = _precompute_dynamic_basis(msm_def, self.dtype)
        else:
            self._open_idx = [10]
            s0 = (
                INITIAL_CONDITIONS
                if initial_state is None
                else np.asarray(initial_state, dtype=float)
            )
            self._dyn_basis = None

        self.s0 = torch.tensor(s0, dtype=self.dtype, device=self.device)

        self.act_proto = ActivationProtocol(act_cfg)
        self.inact_proto = InactivationProtocol(inact_cfg)
        self.csi_proto = CSInactivationProtocol(csi_cfg)
        self.rec_proto = RecoveryProtocol(rec_cfg)

    def _Q(self, V: float) -> torch.Tensor:
        if self.msm_def is None:
            return build_Q_11state(self.params, V)
        return build_Q_dynamic(self.params, V, self.msm_def, self._dyn_basis)

    def _expm(self, Q_dt: torch.Tensor) -> torch.Tensor:
        """Matrix exponential with MPS fallback (MPS lacks aten::linalg_matrix_exp)."""
        if Q_dt.device.type == "mps":
            # Compute on CPU; .cpu()/.to() are differentiable so autograd is preserved
            return torch.linalg.matrix_exp(Q_dt.cpu()).to(Q_dt.device)
        return torch.linalg.matrix_exp(Q_dt)

    def _prop(self, P: torch.Tensor, V: float, dt: float) -> torch.Tensor:
        """Exact propagation through a constant-voltage segment."""
        return self._expm(self._Q(V) * dt) @ P

    def _prop_peak(
        self, P: torch.Tensor, V: float, total_dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate through total_dt at voltage V using n_peak_steps substeps.

        Returns (P_final, peak_open_probability).
        Gradient flows through the peak value via torch.max.
        """
        n = self.n_peak_steps
        dt_sub = total_dt / n
        M = self._expm(self._Q(V) * dt_sub)

        states = [P]
        for _ in range(n):
            P = M @ P
            states.append(P)

        all_states = torch.stack(states, dim=0)  # (n+1, n_states)
        open_probs = all_states[:, self._open_idx].sum(dim=1)  # (n+1,)
        peak = open_probs.max()
        return P, peak

    def _open_prob(self, P: torch.Tensor) -> torch.Tensor:
        return P[self._open_idx].sum()

    def run_activation(self, test_voltages=None) -> torch.Tensor:
        proto = self.act_proto
        if test_voltages is None:
            test_voltages = proto.get_test_voltages()

        cfg = proto.cfg
        # Single exact propagation through the common holding phase
        P_hold = self._prop(self.s0, cfg.v_hold, cfg.t_hold)

        conductances = []
        for V in test_voltages:
            _, peak = self._prop_peak(P_hold, float(V), cfg.t_test)
            conductances.append(self.g_k_max * peak)

        cond = torch.stack(conductances)
        mx = cond.max()
        return cond / mx if mx > 0 else cond

    def run_inactivation(self, test_voltages=None) -> torch.Tensor:
        proto = self.inact_proto
        if test_voltages is None:
            test_voltages = proto.get_test_voltages()

        cfg = proto.cfg
        P_hold = self._prop(self.s0, cfg.v_hold, cfg.t_hold)
        t_test_dur = max(self.t_total - cfg.t_hold - cfg.t_cond, 1e-3)

        currents = []
        for V_cond in test_voltages:
            P_cond = self._prop(P_hold, float(V_cond), cfg.t_cond)
            baseline = self._open_prob(P_cond)
            _, peak = self._prop_peak(P_cond, cfg.v_depo, t_test_dur)
            g = self.g_k_max * (peak - baseline)
            currents.append(g * (cfg.v_depo - cfg.v_hold))

        curr = torch.stack(currents)
        mx = curr.max()
        return curr / mx if mx > 0 else curr

    def run_cs_inactivation(self, test_times=None) -> torch.Tensor:
        proto = self.csi_proto
        if test_times is None:
            test_times = proto.get_test_times()

        cfg = proto.cfg
        t_initial = cfg.t_initial

        # Baseline: peak open probability during initial hold
        _, baseline_peak = self._prop_peak(self.s0, cfg.v_hold, t_initial)
        baseline_peak = baseline_peak.clamp(min=1e-12)
        P_after_initial = self._prop(self.s0, cfg.v_hold, t_initial)

        currents = []
        for t_pulse in test_times:
            t_prep_dur = float(t_pulse) - t_initial
            t_depo_dur = max(cfg.t_test_end - float(t_pulse), 1e-3)
            P_prep = self._prop(P_after_initial, cfg.v_prep, t_prep_dur)
            _, peak_after = self._prop_peak(P_prep, cfg.v_depo, t_depo_dur)
            g = self.g_k_max * peak_after / baseline_peak
            currents.append(g * (cfg.v_depo - cfg.v_prep))

        curr = torch.stack(currents)
        mx = curr.max()
        return curr / mx if mx > 0 else curr

    def run_recovery(self, test_times=None) -> torch.Tensor:
        proto = self.rec_proto
        if test_times is None:
            test_times = proto.get_test_times()

        cfg = proto.cfg
        t_prep = cfg.t_prep
        t_pulse = cfg.t_pulse

        # Equilibrate then apply inactivating pulse, tracking peak
        P0 = self._prop(self.s0, cfg.v_hold, t_prep)
        P_inact, g_pre = self._prop_peak(P0, cfg.v_depo, t_pulse - t_prep)
        g_pre = g_pre.clamp(min=1e-12)

        ratios = []
        for t_rec_end in test_times:
            t_rec_dur = max(float(t_rec_end) - t_pulse, 0.0)
            t_test_dur = max(self.t_total - float(t_rec_end), 1e-3)

            if t_rec_dur > 0:
                P_rec = self._prop(P_inact, cfg.v_hold, t_rec_dur)
            else:
                P_rec = P_inact

            _, g_test = self._prop_peak(P_rec, cfg.v_depo, t_test_dur)
            ratios.append(g_test / g_pre)

        return torch.stack(ratios)
