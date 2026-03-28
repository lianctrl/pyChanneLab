import sys
from pathlib import Path

# Ensure core/ and helpers are importable when running via `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).parent))

from copy import deepcopy

import streamlit as st

from core.config import TIME_PARAMS, G_K_MAX, TEMPERATURE
from core.config import (
    ActivationConfig,
    InactivationConfig,
    CSInactivationConfig,
    RecoveryConfig,
)
from core.msm_builder import PRESETS

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(page_title="pyChanneLab", page_icon="🔬", layout="wide")
st.title("🔬 pyChanneLab — Ion Channel MSM Fitting")
st.caption("Build your own Markov State Model, configure protocols, upload data, fit.")


# ── Session-state initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    if "msm_def" not in st.session_state:
        st.session_state.msm_def = PRESETS["11-state Kv channel (C0-C4, I0-I4, O)"]()

    _DEFAULTS = {
        "initial_conditions": None,  # None → derived from msm_def
        "g_k_max": G_K_MAX,
        "temperature": TEMPERATURE,
        "t_total": TIME_PARAMS["tend"],
        "dt": TIME_PARAMS["dt"],
        "act_cfg": ActivationConfig(),
        "inact_cfg": InactivationConfig(),
        "csi_cfg": CSInactivationConfig(),
        "rec_cfg": RecoveryConfig(),
        "exp_data": {},
        "opt_log": [],
        "opt_result": None,
        "fitted_params": None,
        "opt_costs_initial": None,
        "opt_costs_final": None,
        "curve_fit_types": {},  # {pk: curve_type_key}
        "curve_fix_baseline": {},  # {pk: bool}
        "curve_fix_amplitude": {},  # {pk: bool}
        "curve_fit_params_exp": {},  # {pk: (popt, perr, ok, cft, fix_base, fix_amp)}
        "curve_fit_params_sim": {},  # {pk: (popt, perr, ok, cft)}
        "_preview_sim": None,
        "_preview_sim_params": None,
        "opt_weights": {
            "activation": 1.0,
            "inactivation": 1.0,
            "cs_inactivation": 2.0,
            "recovery": 2.0,
        },
    }
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = deepcopy(v)


_init_state()

# ── Tab imports (after sys.path is set) ────────────────────────────────────────
from tabs.builder import render as render_builder
from tabs.protocols import render as render_protocols
from tabs.data import render as render_data
from tabs.preview import render as render_preview
from tabs.optimise import render as render_optimise
from tabs.results import render as render_results

# ── Tab layout ─────────────────────────────────────────────────────────────────
tab_builder, tab_proto, tab_data, tab_preview, tab_opt, tab_results = st.tabs(
    [
        "🏗️ MSM Builder",
        "🧪 Protocols",
        "📁 Data",
        "🔍 Preview",
        "🚀 Optimise",
        "📊 Results",
    ]
)

with tab_builder:
    render_builder()

with tab_proto:
    render_protocols()

with tab_data:
    render_data()

with tab_preview:
    render_preview()

with tab_opt:
    render_optimise()

with tab_results:
    render_results()
