"""
Bounded parameter <-> unconstrained-raw transforms shared by every optimiser.

Every parameter is mapped to an unconstrained ``raw`` variable so gradient
steps (Adam / L-BFGS) and DE mutations never violate the box bounds.  Two
mappings are supported per parameter, selected by a boolean ``log_mask``:

  linear :  p = lb + (ub - lb) * sigmoid(raw)
  log    :  p = exp( log(lb) + (log(ub) - log(lb)) * sigmoid(raw) )

The log mapping gives *uniform resolution across decades*, which is what rate
constants spanning several orders of magnitude need — with a linear sigmoid the
optimiser has fine resolution only near the middle of the bound range and
vanishing gradients near the small end.

Both directions accept 1-D ``(n,)`` or 2-D ``(P, n)`` parameter tensors; ``lb``,
``ub`` and ``log_mask`` are ``(n,)`` and broadcast over the population axis.
``log_mask=None`` reproduces the old pure-linear behaviour exactly.
"""

import torch

_EPS = 1e-6  # keeps logit() away from its asymptotes


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    """log() with a tiny floor so linear-branch (unused) values never go -inf.

    Where a parameter is *not* log-scaled its lower bound may legitimately be 0;
    torch.where would still evaluate the log branch, and an inf/nan there can
    poison the gradient of the *selected* branch.  Flooring avoids that.
    """
    return torch.log(x.clamp_min(1e-30))


def to_raw(
    params: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    log_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Map parameters in ``[lb, ub]`` to unconstrained raw space."""
    span = ub - lb
    u_lin = (params - lb) / span
    if log_mask is None:
        u = u_lin.clamp(_EPS, 1.0 - _EPS)
        return torch.logit(u)
    log_lb = _safe_log(lb)
    log_span = _safe_log(ub) - log_lb
    u_log = (_safe_log(params) - log_lb) / log_span
    u = torch.where(log_mask, u_log, u_lin).clamp(_EPS, 1.0 - _EPS)
    return torch.logit(u)


def from_raw(
    raw: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    log_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Map unconstrained raw space back to parameters in ``[lb, ub]``."""
    s = torch.sigmoid(raw)
    p_lin = lb + (ub - lb) * s
    if log_mask is None:
        return p_lin
    log_lb = _safe_log(lb)
    log_span = _safe_log(ub) - log_lb
    p_log = torch.exp(log_lb + log_span * s)
    return torch.where(log_mask, p_log, p_lin)


def prior_penalty(
    params: torch.Tensor,
    prior: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    log_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Scale-invariant squared deviation of ``params`` from a ``prior``.

    For log-scaled parameters the deviation is the log fold-change
    ``log(θ) − log(θ_prior)`` (a 2× change costs ``(ln2)²`` regardless of the
    parameter's magnitude); for linear parameters it is normalised by the bound
    width, ``(θ − θ_prior)/(ub − lb)``.  Returns ``Σ_i d_i²`` — a scalar for a
    1-D ``(n,)`` input, or ``(P,)`` for a batched ``(P, n)`` input.  The penalty
    is 0 at the prior, so as a cost term it acts as a soft pull toward it.
    """
    span = ub - lb
    d_lin = (params - prior) / span
    if log_mask is None:
        d = d_lin
    else:
        d_log = _safe_log(params) - _safe_log(prior)
        d = torch.where(log_mask, d_log, d_lin)
    return (d ** 2).sum(dim=-1)


def prior_penalty_np(params, prior, lb, ub, log_mask=None) -> float:
    """NumPy twin of :func:`prior_penalty` (for the scipy cost path). 1-D only."""
    import numpy as np

    params = np.asarray(params, dtype=float)
    prior = np.asarray(prior, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    d_lin = (params - prior) / (ub - lb)
    if log_mask is None:
        d = d_lin
    else:
        mask = np.asarray(log_mask, dtype=bool)
        d_log = np.log(np.maximum(params, 1e-30)) - np.log(np.maximum(prior, 1e-30))
        d = np.where(mask, d_log, d_lin)
    return float(np.sum(d ** 2))


def build_log_mask(
    bounds,
    log_flags=None,
    *,
    device=None,
    dtype=torch.bool,
) -> torch.Tensor:
    """Build a ``(n,)`` boolean mask of which parameters are log-scaled.

    Parameters
    ----------
    bounds    : list of (lb, ub) tuples.
    log_flags : explicit per-parameter flags (e.g. ``msm_def.free_log_mask``).
                If None, log-scale is *inferred* for parameters whose bounds are
                strictly positive and span more than two decades — this keeps
                callers that have no MSMDefinition (e.g. the toy-function tests)
                working sensibly.
    """
    n = len(bounds)
    if log_flags is None:
        flags = [
            (lb > 0.0 and ub / lb > 100.0) for (lb, ub) in bounds
        ]
    else:
        flags = [bool(f) and bounds[i][0] > 0.0 for i, f in enumerate(log_flags)]
    return torch.tensor(flags, dtype=dtype, device=device)
