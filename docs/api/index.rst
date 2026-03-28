API Reference
=============

All public modules live under ``core/``.  Import them as:

.. code-block:: python

   from core.config      import ActivationConfig, ...
   from core.msm_builder import MSMDefinition, ...
   from core.simulator   import ProtocolSimulator
   from core.optimizer   import CostFunction, ParameterOptimizer

.. toctree::
   :maxdepth: 1

   config
   msm_builder
   protocols
   simulator
   optimizer
   curve_fitter
   data_loader
   torch_simulator
   torch_optimizer
   torch_de
