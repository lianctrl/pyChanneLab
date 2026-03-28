"""
Phenomenological curve fitting and information criteria for ion-channel protocols.

Four model functions are provided:
  exp_decay   — a·exp(-x/τ)+C          (e.g. inactivation onset)
  exp_rise    — C+a·(1-exp(-x/τ))      (e.g. recovery, CSI)
  sigmoid     — B+A/(1+exp(k·(x-v₀))) (Boltzmann, decreasing; e.g. h∞/V)
  sigmoid_inv — B+A/(1+exp(-k·(x-v₀)))(Boltzmann, increasing; e.g. G/V)
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple

# ─── model functions ──────────────────────────────────────────────────────────


def _exp_decay(x, a, tau, C):
    """y = a * exp(-x/tau) + C"""
    return a * np.exp(-x / np.where(tau == 0, 1e-12, tau)) + C


def _exp_rise(x, a, tau, C):
    """y = C + a * (1 - exp(-x/tau))"""
    return C + a * (1.0 - np.exp(-x / np.where(tau == 0, 1e-12, tau)))


def _sigmoid(x, A, B, k, v0):
    """y = B + A / (1 + exp(k*(x-v0)))  — decreasing Boltzmann"""
    z = np.clip(k * (x - v0), -500.0, 500.0)
    return B + A / (1.0 + np.exp(z))


def _sigmoid_inv(x, A, B, k, v0):
    """y = B + A / (1 + exp(-k*(x-v0)))  — increasing Boltzmann"""
    z = np.clip(-k * (x - v0), -500.0, 500.0)
    return B + A / (1.0 + np.exp(z))


# ─── registry ─────────────────────────────────────────────────────────────────

# {key: (function, formula_string, [param_names])}
CURVE_FUNCTIONS: Dict[str, Tuple] = {
    "exp_decay": (_exp_decay, "a·exp(-x/τ)+C", ["a", "τ", "C"]),
    "exp_rise": (_exp_rise, "C+a·(1-exp(-x/τ))", ["a", "τ", "C"]),
    "sigmoid": (_sigmoid, "B+A/(1+exp(k·(x-v₀)))", ["A", "B", "k", "v₀"]),
    "sigmoid_inv": (_sigmoid_inv, "B+A/(1+exp(-k·(x-v₀)))", ["A", "B", "k", "v₀"]),
}

CURVE_LABELS: Dict[str, str] = {
    "exp_decay": "Exp decay: a·exp(-x/τ)+C",
    "exp_rise": "Exp rise: C+a·(1-exp(-x/τ))",
    "sigmoid": "Sigmoid (↓): B+A/(1+exp(k·(x-v₀)))",
    "sigmoid_inv": "Inv sigmoid (↑): B+A/(1+exp(-k·(x-v₀)))",
}

# Sensible defaults per protocol
PROTOCOL_CURVE_DEFAULTS: Dict[str, str] = {
    "activation": "sigmoid_inv",  # G/V rises with voltage
    "inactivation": "sigmoid",  # h∞/V decreases with voltage
    "cs_inactivation": "exp_rise",  # rises with prepulse duration
    "recovery": "exp_rise",  # recovers (rises) with interval
}

# Python-source names (for generated scripts)
CURVE_FUNC_NAMES: Dict[str, str] = {
    "exp_decay": "_exp_decay",
    "exp_rise": "_exp_rise",
    "sigmoid": "_sigmoid",
    "sigmoid_inv": "_sigmoid_inv",
}


# ─── fitting ──────────────────────────────────────────────────────────────────


def _initial_guess(x: np.ndarray, y: np.ndarray, curve_type: str) -> List[float]:
    y_range = float(np.ptp(y)) or 1.0
    y_min = float(np.min(y))
    x_mid = float(np.median(x))
    x_range = float(np.ptp(x)) or 1.0

    if curve_type in ("exp_decay", "exp_rise"):
        return [y_range, x_range / 3.0, y_min]
    else:  # sigmoid / sigmoid_inv
        return [y_range, y_min, 1.0 / max(x_range, 1e-9), x_mid]


def _make_constrained(fn, param_names: List[str], fixed: dict):
    """
    Return a wrapper function with *fixed* parameters baked in.
    *fixed* maps param_name → fixed_value.
    The wrapper accepts only the free parameters as positional args.
    """
    free_names = [n for n in param_names if n not in fixed]

    def wrapper(x, *free_args):
        all_args = []
        fi = 0
        for name in param_names:
            if name in fixed:
                all_args.append(fixed[name])
            else:
                all_args.append(free_args[fi])
                fi += 1
        return fn(x, *all_args)

    return wrapper, free_names


def fit_curve(
    x: np.ndarray,
    y: np.ndarray,
    curve_type: str,
    fix_baseline: bool = False,
    fix_amplitude: bool = False,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Fit *curve_type* to data (x, y).

    Parameters
    ----------
    fix_baseline  : if True, fix the additive offset (C / B) to 0.
    fix_amplitude : if True, fix the multiplicative amplitude (a / A) to 1.

    Returns
    -------
    popt    : best-fit parameters (full length, fixed params at their fixed value)
    perr    : 1-σ uncertainties (0 for fixed params, nan where not computable)
    success : True if curve_fit converged
    """
    fn, _, param_names = CURVE_FUNCTIONS[curve_type]
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    n_params = len(param_names)
    nan_arr = np.full(n_params, np.nan)

    # Determine which parameters are fixed
    fixed: dict = {}
    if fix_amplitude:
        # First param is amplitude: 'a' for exp functions, 'A' for sigmoids
        fixed[param_names[0]] = 1.0
    if fix_baseline:
        if "C" in param_names:
            fixed["C"] = 0.0
        elif "B" in param_names:
            fixed["B"] = 0.0

    free_names = [n for n in param_names if n not in fixed]
    n_free = len(free_names)

    if len(x) < max(n_free, 1):
        return nan_arr.copy(), nan_arr.copy(), False

    # Build fitting function (possibly constrained)
    if fixed:
        fit_fn, _ = _make_constrained(fn, param_names, fixed)
        p0_all = _initial_guess(x, y, curve_type)
        free_indices = [i for i, n in enumerate(param_names) if n not in fixed]
        p0_free = [p0_all[i] for i in free_indices]
    else:
        fit_fn = fn
        p0_free = _initial_guess(x, y, curve_type)
        free_indices = list(range(n_params))

    try:
        popt_free, pcov_free = curve_fit(fit_fn, x, y, p0=p0_free, maxfev=20000)
        diag = np.diag(pcov_free)
        perr_free = np.where(diag >= 0, np.sqrt(diag), np.nan)
    except Exception:
        return nan_arr.copy(), nan_arr.copy(), False

    # Reconstruct full-length popt / perr (fixed params at their fixed values)
    popt = np.array([fixed.get(n, np.nan) for n in param_names], dtype=float)
    perr = np.zeros(n_params, dtype=float)
    for fi, idx in enumerate(free_indices):
        popt[idx] = popt_free[fi]
        perr[idx] = perr_free[fi]

    return popt, perr, True


def eval_curve(x: np.ndarray, popt: np.ndarray, curve_type: str) -> np.ndarray:
    """Evaluate *curve_type* at positions *x* given parameters *popt*."""
    fn, _, _ = CURVE_FUNCTIONS[curve_type]
    return fn(np.asarray(x, float), *popt)


# ─── AIC / BIC ────────────────────────────────────────────────────────────────


def compute_aic_bic(
    fitted_params: np.ndarray,
    exp_data: dict,
    sim_data: dict,
) -> Dict[str, float]:
    """
    Compute AIC and BIC for a fitted MSM using the Gaussian log-likelihood
    approximation (MLE estimate σ² = RSS/n):

        AIC = n·ln(RSS/n) + 2·k
        BIC = n·ln(RSS/n) + k·ln(n)

    Parameters
    ----------
    fitted_params : array of length k (optimised model parameters)
    exp_data      : {protocol_key: (x_exp, y_exp, y_err) or None}
    sim_data      : {protocol_key: (x_sim, y_sim)}
    """
    k = int(len(fitted_params))
    n_total = 0
    rss_total = 0.0

    for pk, exp in exp_data.items():
        if exp is None or pk not in sim_data:
            continue
        x_exp, y_exp, _ = exp
        x_sim, y_sim = sim_data[pk]
        y_pred = np.interp(
            np.asarray(x_exp, float),
            np.asarray(x_sim, float),
            np.asarray(y_sim, float),
        )
        rss_total += float(np.sum((np.asarray(y_exp, float) - y_pred) ** 2))
        n_total += int(len(y_exp))

    out: Dict[str, float] = {
        "AIC": float("nan"),
        "BIC": float("nan"),
        "n_points": float(n_total),
        "k_params": float(k),
        "RSS": rss_total,
    }

    if n_total > 0 and rss_total > 0:
        log_rss_n = np.log(rss_total / n_total)
        out["AIC"] = float(n_total * log_rss_n + 2.0 * k)
        out["BIC"] = float(n_total * log_rss_n + k * np.log(n_total))

    return out
