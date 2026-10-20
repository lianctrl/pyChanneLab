"""
Unit tests — sweeps start from the steady state at the holding potential.

In the experiments the cell sits at v_hold for seconds between sweeps, so each
simulated sweep must start from the stationary distribution of Q(v_hold), not
from an arbitrary initial vector relaxed for a short t_hold / t_initial.

Checks:
1. stationary_distribution solves Q p = 0, Σp = 1, p ≥ 0, equals long-time
   propagation, is batched, differentiable, and matches its NumPy twin.
2. With steady_state_start=True the protocol output no longer depends on the
   length of the initial hold (it did before — the artefact being fixed).
3. The batched DE simulator and the scipy simulator honour the flag too.
"""

import dataclasses

import numpy as np
import pytest
import torch

from core.config import CSInactivationConfig
from core.msm_builder import make_11state_preset
from core.simulator import ProtocolSimulator, stationary_distribution as stationary_np
from core.torch_de import BatchedProtocolSimulator
from core.torch_simulator import TorchProtocolSimulator, stationary_distribution

DT = torch.float64
MSM = make_11state_preset()
PARAMS = torch.tensor(MSM.free_initial_guess, dtype=DT)
DURATIONS = np.array([0.010, 0.130, 0.250, 0.550])  # s, CSI prepulse lengths


def _csi_cfg(t_initial: float, steady: bool) -> CSInactivationConfig:
    return CSInactivationConfig(
        t_initial=t_initial,
        t_test_end=t_initial + DURATIONS.max() + 0.5,
        steady_state_start=steady,
    )


def _torch_sim(params=PARAMS, **kw) -> TorchProtocolSimulator:
    return TorchProtocolSimulator(params, msm_def=MSM, n_peak_steps=40, **kw)


class TestStationaryDistribution:
    def test_balance_normalisation_positivity(self):
        Q = _torch_sim()._Q(-90.0)
        p = stationary_distribution(Q)
        assert torch.allclose(Q @ p, torch.zeros_like(p), atol=1e-10)
        assert float(p.sum()) == pytest.approx(1.0, abs=1e-12)
        assert bool((p >= 0).all())

    def test_equals_long_time_propagation(self):
        sim = _torch_sim()
        p_long = sim._prop(sim.s0, -90.0, 200.0)  # ≫ slowest relaxation time
        assert torch.allclose(stationary_distribution(sim._Q(-90.0)), p_long, atol=1e-8)

    def test_batched_matches_unbatched(self):
        sets = torch.stack([PARAMS, PARAMS * 1.5, PARAMS * 0.7])
        Qs = torch.stack([_torch_sim(p)._Q(-90.0) for p in sets])
        pb = stationary_distribution(Qs)
        for i in range(3):
            assert torch.allclose(pb[i], stationary_distribution(Qs[i]), atol=1e-12)

    def test_numpy_twin_matches(self):
        Q = _torch_sim()._Q(-70.0)
        p_np = stationary_np(Q.detach().numpy())
        assert np.allclose(p_np, stationary_distribution(Q).numpy(), atol=1e-12)

    def test_differentiable(self):
        p = PARAMS.clone().requires_grad_(True)
        stationary_distribution(_torch_sim(p)._Q(-90.0))[0].backward()
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


class TestHoldLengthIndependence:
    """Steady-state start makes CSI independent of t_initial; s0 start did not."""

    def _run(self, t_initial: float, steady: bool) -> np.ndarray:
        sim = _torch_sim(csi_cfg=_csi_cfg(t_initial, steady))
        return sim.run_cs_inactivation(t_initial + DURATIONS).detach().numpy()

    def test_steady_start_independent_of_hold(self):
        assert np.allclose(self._run(0.1, True), self._run(2.0, True), atol=1e-10)

    def test_s0_start_depends_on_hold(self):
        # Documents the artefact being fixed: a 0.1 s hold from s0 is not
        # equilibrated, so the curve changes when the hold is lengthened.
        diff = np.abs(self._run(0.1, False) - self._run(2.0, False)).max()
        assert diff > 1e-3

    def test_short_hold_s0_converges_to_steady(self):
        # A long-enough hold from s0 reaches the same answer as steady start.
        assert np.allclose(self._run(20.0, False), self._run(0.1, True), atol=1e-6)


class TestOtherSimulators:
    def test_batched_de_loss_matches_single(self):
        cfg = _csi_cfg(0.1, True)
        x_sim = 0.1 + DURATIONS
        y = _torch_sim(csi_cfg=cfg).run_cs_inactivation(x_sim).detach().numpy()
        # Perturbed targets so the loss is non-trivial.
        y_target = y * 0.9
        single = float(np.mean((y - y_target) ** 2))
        batched = BatchedProtocolSimulator(
            PARAMS.unsqueeze(0), msm_def=MSM, csi_cfg=cfg, n_peak_steps=40
        ).cs_inactivation_loss(x_sim, y_target)
        assert float(batched[0]) == pytest.approx(single, rel=1e-8)

    def test_scipy_simulator_independent_of_hold(self):
        def run(t_initial):
            sim = ProtocolSimulator(
                MSM.free_initial_guess, msm_def=MSM,
                csi_cfg=_csi_cfg(t_initial, True), t_total=2.0, dt=1e-4,
            )
            return sim.run_cs_inactivation(t_initial + DURATIONS)

        # The dt-grid simulator can place a pulse edge ±1 sample apart for
        # different t_initial (float arange vs `t <= t_pulse`); that error is
        # O(dt) (~1e-5 at dt=1e-4), unrelated to the start state.
        assert np.allclose(run(0.1), run(0.6), atol=1e-4)

    def test_flag_is_a_dataclass_field(self):
        # helpers.generate_run_script exports configs via dataclasses.fields,
        # so the flag must be a real field to round-trip into the run script.
        names = {f.name for f in dataclasses.fields(CSInactivationConfig)}
        assert "steady_state_start" in names
