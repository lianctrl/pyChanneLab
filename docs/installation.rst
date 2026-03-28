Installation
============

Prerequisites
-------------

* Python ≥ 3.12
* `uv <https://docs.astral.sh/uv/>`_ (recommended) **or** conda/mamba

With uv (recommended)
---------------------

.. code-block:: bash

   # Clone the repository
   git clone <repo-url>
   cd pyChanneLab

   # Create a virtual environment and install all dependencies
   uv sync

   # Launch the GUI
   uv run streamlit run src/pychannel_lab/app.py
   # or, after sync, use the installed entry-point:
   uv run pychannelab

With conda / mamba
------------------

.. code-block:: bash

   conda create -n pychannelab python=3.12
   conda activate pychannelab
   pip install -e ".[docs]"

   # Launch the GUI
   pychannelab

With plain pip (editable install)
----------------------------------

.. code-block:: bash

   pip install -e "."
   pychannelab

Optional: GPU support
---------------------

PyTorch is installed by default as a CPU build.  For CUDA acceleration on
an HPC cluster, replace the torch dependency before syncing:

.. code-block:: bash

   # Example for CUDA 12.1
   uv pip install torch --index-url https://download.pytorch.org/whl/cu121

Running the tests
-----------------

.. code-block:: bash

   uv run pytest test/
