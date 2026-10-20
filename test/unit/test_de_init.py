"""
Unit tests — DE population centred on a given parameter vector.

Used to refit a related dataset (e.g. a mutant) starting from a previous fit.
The fitness is replaced by the raw-space distance to the centre, so with
maxiter=0 the best member is the centre iff it is in the initial population.

Checks:
1. Member 0 is exactly the centre.
2. Every member lies inside the bounds.
3. The same seed gives the same population; a different seed does not.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.msm_builder import make_11state_preset  # noqa: E402
from core.torch_de import TorchDEOptimizer, _to_raw_batch  # noqa: E402
from core.param_transform import build_log_mask  # noqa: E402

MSM = make_11state_preset()
BOUNDS = list(MSM.free_bounds)
LB = np.array([b[0] for b in BOUNDS])
UB = np.array([b[1] for b in BOUNDS])
# A centre well inside the box, plus one parameter sitting near a bound.
CENTER = np.sqrt(LB * UB)
CENTER[0] = UB[0] * 0.99


def _population(seed, center=CENTER):
    de = TorchDEOptimizer(
        {"activation": None, "inactivation": None, "cs_inactivation": None, "recovery": None},
        msm_def=MSM, device=torch.device("cpu"), dtype=torch.float64,
    )
    lb = torch.tensor(LB)
    ub = torch.tensor(UB)
    mask = build_log_mask(BOUNDS, MSM.free_log_mask)
    raw_c = _to_raw_batch(torch.tensor(center).unsqueeze(0), lb, ub, mask)
    de._eval_population = lambda raw, *a: ((raw - raw_c) ** 2).sum(dim=1)
    torch.manual_seed(seed)
    return de.optimize(
        BOUNDS, pop_size=20, maxiter=0, log_mask=MSM.free_log_mask,
        return_population=True, init_center=center, init_spread=0.5,
    )


def test_member_zero_is_center():
    best, _, fit = _population(0)
    np.testing.assert_allclose(best, CENTER, rtol=1e-9)
    assert fit[0] == pytest.approx(0.0, abs=1e-12)


def test_population_within_bounds():
    _, pop, _ = _population(0)
    assert (pop >= LB * (1 - 1e-9)).all() and (pop <= UB * (1 + 1e-9)).all()
    # Perturbed members actually differ from the centre.
    assert np.abs(np.log(pop[1:] / CENTER)).max() > 0.1


def test_seed_reproducible():
    _, a, _ = _population(7)
    _, b, _ = _population(7)
    _, c, _ = _population(8)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
