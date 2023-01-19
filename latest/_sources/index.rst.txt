.. Salesforce CausalAI Library documentation master file, created by
   sphinx-quickstart on Mon Nov 28 11:39:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Salesforce CausalAI Library's documentation!
=======================================================

Salesforce CausalAI is an open-source Python library for causal analysis using observational data. It supports causal discovery and causal inference for tabular and time series data, of both discrete and continuous types. This library includes algorithms that handle linear and non-linear causal relationship between variables, and uses multi-processing for speed-up. We also include a data generator capable of generating synthetic data with specified structural equation model for both the aforementioned data formats and types, that helps users control the ground-truth causal process while investigating various algorithms. Finally, we provide a user interface (UI) that allows users to perform causal analysis on data without coding. The goal of this library is to provide a fast and flexible solution for a variety of problems in the domain of causality.

Installation
============

Prior to installing the library, create a conda environment with Python 3.9 or a later version. This can be done by executing ``conda create -n causal_ai_env python=3.9``. Activate this environment by executing ``conda activate causal_ai_env``. To install Salesforce CausalAI, git clone the library, go to the root directory of the repository, and execute ``pip install .``. 

Before importing and calling the library, or launching the UI, remember to first activate the conda environemnt.

User Inferface (UI)
===================

We provide an online UI for users to directly upload their data and run causal discovery and causal inference algorithms without the need to write any code. An introduction to the UI can be found :doc:`here <ui_tutorial>`.

In order to launch the UI, go to the root directory of the library and execute ``./launch_ui.sh``, and open the url specified in the terminal in a browser. In order to terminate the UI, press Ctrl+c, and then execute ``./exit_ui.sh``.

Contents
========

1. :doc:`Prior Knowledge <models.common.prior_knowledge>`

2. Data Layer
	- :doc:`Base Data Class <data.base>`
	- :doc:`Time Series Data Class <data.time_series>`
	- :doc:`Tabular Data Class <data.tabular>`
	- :doc:`Data Generator <data.data_generator>`
	- Transform
		- :doc:`Base Transform Class <data.transforms.base>`
		- :doc:`Tabular Transform Class <data.transforms.tabular>`
		- :doc:`Time Series Transform Class <data.transforms.time_series>`

3. Causal Discovery
	- Time Series
		- :doc:`Base Class <models.time_series.base>`
		- :doc:`PC Algorithm <models.time_series.pc>`
		- :doc:`Granger Causality <models.time_series.granger>`
		- :doc:`VARLINGAM <models.time_series.var_lingam>`
	- Tabular
		- :doc:`Base Class <models.tabular.base>`
		- :doc:`PC Algorithm <models.tabular.pc>`

4. Causal Inference
	- :doc:`Time Series Causal Inference <models.time_series.causal_inference>`
	- :doc:`Tabular Causal Inference <models.tabular.causal_inference>`

5. Other
	- :doc:`Misc <misc.misc>` (plotting and evaluation)
	- CI Tests for PC Algorithm
		- :doc:`Discrete CI Tests <models.common.CI_tests.discrete_ci_tests>`
		- :doc:`KCI Tests <models.common.CI_tests.kci>`
		- :doc:`Partial Correlation <models.common.CI_tests.partial_correlation>`
		- :doc:`Kernels <models.common.CI_tests.kernels>`


Tutorials
=========

1. Prior Knowledge

.. toctree::
   :maxdepth: 1
   :glob:

   tutorials/Prior Knowledge.ipynb

2. Data

.. toctree::
   :maxdepth: 1
   :glob:

   tutorials/Data objects.ipynb

.. toctree::
   :maxdepth: 1
   :glob:

   tutorials/Data Generator.ipynb


3. Causal Discovery for Time Series

.. toctree::
   :maxdepth: 1
   :glob:

   tutorials/PC_Algorithm_TimeSeries.ipynb

.. toctree::
   :maxdepth: 1
   :glob:

   tutorials/GrangerAlgorithm_TimeSeries.ipynb

.. toctree::
   :maxdepth: 1
   :glob:

   tutorials/VARLINGAM_Algorithm_TimeSeries.ipynb


4. Causal Discovery for Tabular Data

.. toctree::
   :maxdepth: 1
   :glob:

   tutorials/PC_Algorithm_Tabular.ipynb


5. Causal Inference

.. toctree::
   :maxdepth: 1
   :glob:

   tutorials/Causal Inference Time Series Data.ipynb

.. toctree::
   :maxdepth: 1
   :glob:

   tutorials/Causal Inference Tabular Data.ipynb

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
