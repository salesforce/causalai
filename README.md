# Salesforce CausalAI Library

## Table of Contents
1. [Introduction](#introduction)
1. [Installation](#installation)
1. [User Inferface](#user-inferface)
1. [Documentation](#documentation)
1. [Technical Report and Citing Salesforce CausalAI](#technical-report-and-citing-salesforce-causalai)

## Introduction

Salesforce CausalAI is an open-source Python library for causal analysis using observational data. It supports causal discovery and causal inference for tabular and time series data, of both discrete and continuous types. This library includes algorithms that handle linear and non-linear causal relationship between variables, and uses multi-processing for speed-up. We also include a data generator capable of generating synthetic data with specified structural equation model for both the aforementioned data formats and types, that helps users control the ground-truth causal process while investigating various algorithms. Finally, we provide a user interface (UI) that allows users to perform causal analysis on data without coding. The goal of this library is to provide a fast and flexible solution for a variety of problems in the domain of causality.

Some of the key features of CausalAI are:

- Data: Causal analysis on tabular and time series data, of both discrete and continuous types.
- Missing Values: Support for handling missing/NaN values in data.
- Data Generator: A synthetic data generator that uses a specified structural equation model (SEM) for generating tabular and time series data. This can be used for evaluating and comparing different causal discovery algorithms since the ground truth values are known.
- Distributed Computing: Use of multi-processing using the Ray \citep{moritz2018ray} library, that can be optionally turned on by the user when dealing with large datasets or number of variables for faster compute.
- Targeted Causal Discovery: In certain cases, we support targeted causal discovery, in which the user is only interested in discovering the causal parents of a specific variable of interest instead of the entire causal graph. This option reduces computational overhead.
- Visualization: Visualize tabular and time series causal graphs.
- Prior Knowledge: Incorporate any user provided partial prior knowledge about the causal graph in the causal discovery process.
- Code-free UI: Provide a code-free user interface in which users may directly upload their data and perform their desired choice of causal analysis algorithm at the click of a button.


### Causal Discovery

Here we clarify which Causal discovery algorithm is supported by CausalAI depending on your data:
- Tabular data
      - Continuous data: PC algorithm.
      - Discrete data: PC algorithm.
- Time Series
      - Continuous data: PC algorithm, Granger Causality, VARLINGAM.
      - Discrete data: PC algorithm.

PC algorithm supports both linear and non-linear causal relationships. Granger and VARLINGAM support linear relationships. Note that we currently do not support hidden confounders in the implemented causal discovery algorithms.

### Causal Inference

We support causal inference-- average treatment effect (ATE) and conditional ATE (CATE)-- for tabular and time series data of continuous and discrete types. Depending on whether the relationship between variables is linear or non-linear, the user may specify a linear or non-linear prediction model respectively in the inference module.

## Installation

Prior to installing the library, create a conda environment with Python 3.9 or a later version. This can be done by executing ``conda create -n causal_ai_env python=3.9``. Activate this environment by executing ``conda activate causal_ai_env``. To install Salesforce CausalAI, git clone the library, go to the root directory of the repository, and execute ``pip install .``. 

Before importing and calling the library, or launching the UI, remember to first activate the conda environemnt.

## User Inferface

We provide an online UI for users to directly upload their data and run causal discovery and causal inference algorithms without the need to write any code. An introduction to the UI can be found :doc:`here <ui_tutorial>`.

In order to launch the UI, go to the root directory of the library and execute ``./launch_ui.sh``, and open the url specified in the terminal in a browser. In order to terminate the UI, press Ctrl+c, and then execute ``./exit_ui.sh``.

## Documentation

For Jupyter notebooks with exmaples, see
[`tutorials`](https://github.com/MetaMind/causalai/tree/main/tutorials). Detailed API documentation with tutorials can be found [here](https://opensource.salesforce.com/causalai). The
[technical report](??) describes the implementation details of the algorithms along with their assumptions and also covers important aspects of the API. Further, it also presents experimental results that demosntrate the speed and performance our library compared with some of the existing libraries.

## Technical Report and Citing Salesforce CausalAI
You can find more details in our technical report: ??

If you're using Salesforce CausalAI in your research or applications, please cite using this BibTeX:
```
@article{?,
      title={Salesforce CausalAI Library: A Fast and Scalable framework for Causal Analysis of Time Series and Tabular Data},
      author={?},
      year={2022},
      eprint={?},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```