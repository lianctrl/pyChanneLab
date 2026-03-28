Quick Start
===========

Launching the GUI
-----------------

After installation (see :doc:`installation`):

.. code-block:: bash

   pychannelab

or without installation:

.. code-block:: bash

   cd src/pychannel_lab
   streamlit run app.py

The GUI opens in your browser at ``http://localhost:8501``.

GUI workflow
~~~~~~~~~~~~

1. **🏗️ MSM Builder** — Load a preset model (e.g. the default 11-state Kv
   channel) or define your own Markov topology by entering states,
   transitions, and parameters.

2. **🧪 Protocols** — Adjust the timing and voltage levels for each of the
   four voltage-clamp protocols (activation G/V, inactivation h∞/V,
   closed-state inactivation, recovery from inactivation).

3. **📁 Data** — Upload one CSV per protocol.  Each file must have a header
   row followed by columns ``x``, ``y`` (and optionally ``y_err``).

4. **🔍 Preview** — Sanity-check the initial parameter guess against your
   experimental data before running the optimiser.

5. **🚀 Optimise** — Configure the DE → Adam → L-BFGS pipeline and click
   *Run Optimisation*.  Progress is shown in real time.  Use the
   *Export run script* button to download a self-contained Python script
   for HPC use (see below).

6. **📊 Results** — Inspect fitted parameters, AIC/BIC information criteria,
   and comparison figures.  Download the results as JSON.

Running the optimisation script on HPC
---------------------------------------

Export the run script from the **Optimise** tab.  The downloaded
``pychannelab_run_<timestamp>.py`` file embeds all current settings (model,
protocols, data, weights, and optimiser hyperparameters) and can be
executed on any machine that has the package installed:

.. code-block:: bash

   # On your workstation / HPC login node
   cp pychannelab_run_20250328_120000.py  <hpc-dir>/src/pychannel_lab/
   ssh hpc

   # On the HPC node
   cd <hpc-dir>
   python src/pychannel_lab/pychannelab_run_20250328_120000.py

The script will:

1. Run DE → Adam → L-BFGS (PyTorch, uses GPU if available).
2. Simulate the fitted model.
3. Compute AIC / BIC.
4. Fit phenomenological curves to experimental and simulated data.
5. Save comparison plots (``comparison.html``, ``comparison.png``) and a
   Markdown report (``report.md``) to ``pychannelab_output/``.

Using the Python API directly
------------------------------

.. code-block:: python

   import sys, numpy as np
   sys.path.insert(0, "src/pychannel_lab")

   from core.msm_builder import MSMDefinition, StateSpec, TransitionSpec, ParamSpec
   from core.config      import ActivationConfig
   from core.simulator   import ProtocolSimulator

   # Define a minimal 2-state model
   msm = MSMDefinition(
       states=[StateSpec("C", "closed"), StateSpec("O", "open")],
       transitions=[
           TransitionSpec("C", "O", "k_CO"),
           TransitionSpec("O", "C", "k_OC"),
       ],
       parameters=[
           ParamSpec("k_CO", 4.0, 0.01, 100.0),
           ParamSpec("k_OC", 1.0, 0.01, 100.0),
       ],
   )

   act_cfg = ActivationConfig(v_min=-60, v_max=60, v_step=20, t_hold=0.1, t_test=0.3)
   sim = ProtocolSimulator(
       np.array([4.0, 1.0]), msm_def=msm, act_cfg=act_cfg, t_total=0.41
   )
   voltages = sim.act_proto.get_test_voltages()
   g_norm   = sim.run_activation()
   print(list(zip(voltages, g_norm)))
