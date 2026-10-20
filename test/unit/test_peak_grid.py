"""
Unit tests — peak open probability on the geometric (per-octave) time grid.

Kv4.3 peaks a few ms after the test step while the measurement window lasts
up to seconds.  A uniform grid over the whole window put the first sample
20-60 ms after the step, so the "peak" was really a late, sweep-dependent
current.  propagate_peak samples per octave instead.

Checks:
1. The peak matches a brute-force 10 µs reference over windows from 50 ms to
   1.5 s (the old uniform grid fails this).
2. The final state is exact: expm(Q·T) @ P.
3. Batched and unbatched calls agree.
4. TorchProtocolSimulator activation matches the scipy simulator (dt = 10 µs).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.config import ActivationConfig  # noqa: E402
from core.msm_builder import make_11state_preset  # noqa: E402
from core.simulator import ProtocolSimulator  # noqa: E402
from core.torch_simulator import (  # noqa: E402
    TorchProtocolSimulator,
    propagate_peak,
    stationary_distribution,
)

MSM = make_11state_preset()
# WT fit (s23_both): peak at +60 mV ≈ 5 ms after the step.
WT = dict(alpha_0=169.4, alpha_1=1.758, beta_0=22.13, beta_1=0.001272, k_CO_0=71.80,
          k_CO_1=0.06178, k_OC_0=12.35, k_OC_1=1.185e-4, k_CI=773.5, k_IC=0.5302, f=0.2319)
PARAMS = torch.tensor([WT[n] for n in MSM.param_names], dtype=torch.float64)
OPEN = MSM.open_state_indices


def _sim(**kw):
    return TorchProtocolSimulator(PARAMS, msm_def=MSM, **kw)


def _step(sim, v_from=-90.0, v_to=60.0):
    return stationary_distribution(sim._Q(v_from)), sim._Q(v_to)


def _reference_peak(Q, P, dt=1e-5, t_max=0.05):
    M = torch.linalg.matrix_exp(Q * dt)
    best = 0.0
    for _ in range(int(round(t_max / dt))):
        P = M @ P
        best = max(best, float(P[OPEN].sum()))
    return best


@pytest.mark.parametrize("total_dt", [0.05, 0.5, 1.5])
def test_peak_matches_fine_reference(total_dt):
    sim = _sim()
    P, Q = _step(sim)
    ref = _reference_peak(Q, P)
    _, peak = propagate_peak(torch.linalg.matrix_exp, Q, P, total_dt, OPEN, 16)
    assert float(peak) == pytest.approx(ref, rel=2e-3)


def test_uniform_grid_would_fail():
    """Guard that the reference actually discriminates: 25 uniform samples over
    1.5 s (the old scheme) miss the peak by far more than the tolerance."""
    sim = _sim()
    P, Q = _step(sim)
    M = torch.linalg.matrix_exp(Q * (1.5 / 25))
    old = max(float((torch.linalg.matrix_exp(Q * 0) @ P)[OPEN].sum()),
              float((M @ P)[OPEN].sum()))
    assert old < 0.9 * _reference_peak(Q, P)


def test_final_state_exact():
    sim = _sim()
    P, Q = _step(sim)
    for T in (0.0123, 0.05, 1.5):
        P_end, _ = propagate_peak(torch.linalg.matrix_exp, Q, P, T, OPEN, 16)
        exact = torch.linalg.matrix_exp(Q * T) @ P
        torch.testing.assert_close(P_end, exact, rtol=1e-9, atol=1e-12)


def test_batched_matches_unbatched():
    sim = _sim()
    P, Q = _step(sim)
    P2, Q2 = _step(sim, -70.0, 20.0)
    Pb, peak_b = propagate_peak(
        torch.linalg.matrix_exp, torch.stack([Q, Q2]), torch.stack([P, P2]), 0.3, OPEN, 16
    )
    for i, (q, p) in enumerate([(Q, P), (Q2, P2)]):
        Pi, peak_i = propagate_peak(torch.linalg.matrix_exp, q, p, 0.3, OPEN, 16)
        torch.testing.assert_close(Pb[i], Pi)
        torch.testing.assert_close(peak_b[i], peak_i)


def test_activation_matches_scipy():
    cfg = ActivationConfig()
    v = np.arange(-90.0, 91.0, 10.0)
    torch_curve = _sim(act_cfg=cfg).run_activation(v).detach().numpy()
    scipy_curve = ProtocolSimulator(
        PARAMS.numpy(), msm_def=MSM, act_cfg=cfg, t_total=cfg.t_hold + cfg.t_test + 0.01
    ).run_activation(v)
    np.testing.assert_allclose(torch_curve, scipy_curve, atol=1e-3)
