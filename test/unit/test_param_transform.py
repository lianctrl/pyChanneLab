"""
Unit tests for the bounded <-> raw parameter transforms (param_transform.py).

Guarantees:
  * round-trip  from_raw(to_raw(p)) ≈ p  for both log and linear parameters,
  * every raw value maps back inside the bounds,
  * the log branch gives log-uniform spacing (equal raw steps -> equal ratios),
  * mixed masks and negative-lower-bound (linear-only) cases behave.
"""

import numpy as np
import pytest
import torch

from core.param_transform import (
    to_raw,
    from_raw,
    build_log_mask,
    prior_penalty,
    prior_penalty_np,
)


def _t(x):
    return torch.tensor(x, dtype=torch.float64)


class TestRoundTrip:
    def test_roundtrip_log_and_linear(self):
        lb = _t([1e-3, 1e-4, 0.01, 1.0])
        ub = _t([2000.0, 5.0, 1.0, 10.0])
        mask = _t([True, True, True, False]).bool()
        # spread of representative values within bounds
        p = _t([1.5, 3e-3, 0.44, 4.2])
        raw = to_raw(p, lb, ub, mask)
        p_rt = from_raw(raw, lb, ub, mask)
        assert torch.allclose(p, p_rt, rtol=1e-6, atol=1e-9)

    def test_roundtrip_batched(self):
        lb = _t([1e-3, 1e-4])
        ub = _t([2000.0, 5.0])
        mask = _t([True, True]).bool()
        p = _t([[0.01, 0.02], [500.0, 4.0], [1.0, 1e-3]])
        raw = to_raw(p, lb, ub, mask)
        assert torch.allclose(from_raw(raw, lb, ub, mask), p, rtol=1e-6, atol=1e-9)

    def test_no_mask_is_linear(self):
        lb, ub = _t([0.0, -10.0]), _t([1.0, 10.0])
        p = _t([0.3, -4.0])
        raw = to_raw(p, lb, ub, None)
        assert torch.allclose(from_raw(raw, lb, ub, None), p, rtol=1e-6, atol=1e-9)


class TestBoundsHold:
    def test_extreme_raw_stays_in_bounds(self):
        lb = _t([1e-3, 1e-4, 0.01])
        ub = _t([2000.0, 5.0, 1.0])
        mask = _t([True, True, True]).bool()
        raw = _t([[-1e3, 1e3, 0.0], [50.0, -50.0, 12.0]])
        p = from_raw(raw, lb, ub, mask)
        assert torch.all(p >= lb - 1e-9)
        assert torch.all(p <= ub + 1e-9)

    def test_log_branch_is_log_uniform(self):
        # Equal steps in raw (via sigmoid arg) should give equal multiplicative
        # steps in parameter space for a log-scaled parameter.
        lb, ub = _t([1e-3]), _t([1e3])
        mask = _t([True]).bool()
        raws = _t([[-2.0], [0.0], [2.0]])
        vals = from_raw(raws, lb, ub, mask).squeeze(1)
        r1 = float(vals[1] / vals[0])
        r2 = float(vals[2] / vals[1])
        assert abs(r1 - r2) / r1 < 0.05  # ratios roughly equal


class TestPriorPenalty:
    def test_zero_at_prior(self):
        lb, ub = _t([1e-3, 0.0]), _t([2000.0, 5.0])
        mask = _t([True, False]).bool()
        prior = _t([200.0, 2.0])
        assert float(prior_penalty(prior, prior, lb, ub, mask)) == pytest.approx(0.0)
        assert prior_penalty_np([200.0, 2.0], [200.0, 2.0], [1e-3, 0.0], [2000.0, 5.0],
                                [True, False]) == pytest.approx(0.0)

    def test_log_fold_formula(self):
        # single log param: doubling -> (ln2)^2
        lb, ub = _t([1e-3]), _t([2000.0])
        mask = _t([True]).bool()
        val = float(prior_penalty(_t([400.0]), _t([200.0]), lb, ub, mask))
        assert val == pytest.approx(np.log(2.0) ** 2, rel=1e-9)

    def test_linear_normalised_formula(self):
        # single linear param: deviation normalised by bound width
        lb, ub = _t([0.0]), _t([4.0])
        mask = _t([False]).bool()
        val = float(prior_penalty(_t([3.0]), _t([1.0]), lb, ub, mask))
        assert val == pytest.approx(((3.0 - 1.0) / 4.0) ** 2, rel=1e-9)

    def test_log_scale_invariance(self):
        # Same fold-change gives the same penalty regardless of magnitude.
        lb, ub = _t([1e-6]), _t([1e6])
        mask = _t([True]).bool()
        p1 = float(prior_penalty(_t([20.0]), _t([10.0]), lb, ub, mask))
        p2 = float(prior_penalty(_t([2000.0]), _t([1000.0]), lb, ub, mask))
        assert p1 == pytest.approx(p2, rel=1e-9)

    def test_batched_shape(self):
        lb, ub = _t([1e-3, 1e-3]), _t([100.0, 100.0])
        mask = _t([True, True]).bool()
        prior = _t([10.0, 5.0])
        params = _t([[10.0, 5.0], [20.0, 5.0], [10.0, 10.0]])
        out = prior_penalty(params, prior, lb, ub, mask)
        assert out.shape == (3,)
        assert float(out[0]) == pytest.approx(0.0)  # at prior


class TestBuildLogMask:
    def test_explicit_flags_respect_positive_lb(self):
        bounds = [(1e-3, 2000.0), (0.0, 5.0), (0.01, 1.0)]
        # ask for all-log, but the 0-lb one must be forced linear
        mask = build_log_mask(bounds, [True, True, True])
        assert mask.tolist() == [True, False, True]

    def test_inferred_when_no_flags(self):
        # >2 decades and positive -> log; small span or non-positive -> linear
        bounds = [(1e-3, 2000.0), (0.0, 5.0), (1.0, 10.0), (-3.0, 3.0)]
        mask = build_log_mask(bounds, None)
        assert mask.tolist() == [True, False, False, False]
