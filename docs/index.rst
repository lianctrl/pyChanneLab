pyChanneLab Documentation
=========================

**pyChanneLab** is a Python toolkit for fitting Markov State Models (MSMs)
to voltage-clamp ion-channel data.  It provides:

* A **Streamlit GUI** for interactive model definition, data upload, and
  fitting.
* A **PyTorch pipeline** (Differential Evolution → Adam → L-BFGS) for
  GPU-accelerated parameter optimisation.
* A **standalone script generator** that exports self-contained Python
  scripts for running optimisation on HPC clusters.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
