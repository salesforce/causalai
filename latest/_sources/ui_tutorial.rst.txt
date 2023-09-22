
How to use Salesforce CausalAI Library's UI
===========================================

What is the purpose of this UI?
-------------------------------

The purpose of this user inferface is to allow users to be able to perform causal discovery and causal inference on a dataset without the need for coding. This UI is mainly meant for a quick exploratoration of various algorithms on either synthetic data or user uploaded data.

Data:
-----

The UI supports generating synthetic tabular and time series data using a randomly generated causal graph, which are downloadable. The user may also upload their own data. For causal discovery, the user may only upload a data file. But for causal inference, the user must upload both a data file and the causal graph associated with the data. Our UI limits the uploadable data file size to 9MB and a maximum of 40 variables. If the user needs to perform causal analysis on a larger dataset, then they will need to use our library using their own computational resource.

We provide a README and sample data files for users to understand the format which is acceptable by our UI, in case they want to upload a data file. CSV and JSON formats are supported. We similarly provide sample causal graph files (JSON format supported). The link to these files can be found on the main UI page. Look for "Need an example? Download sample files here".

Algorithm:
----------

We support all the causal discovery and causal inference algorithms in our library , along with prior knowledge in the UI. However, there are some important details and exceptions to this. 

- Prior knowledge is not supported for the VARLINGAM algorithm. 
- For time series data, existing links is not supported in prior knowledge. 
- For the PC algorithm, we use the default setting of a maximum condition set size of 4 for conditional independence tests between any two variables. 
- For the tabular PC algorithm, we found that the edge orientation part of the algorithm may perform poorly in some cases. This is not a problem for time series data since the temporal information helps in orienting edges. Therefore, for tabular PC, we provide two download graph buttons on the causal discovery page-- `Download directed graph`, and `Download undirected graph`. The former downloads the final directed graph that the PC algorithm outputs. The latter, on the other hand, downloads the skeleton graph prior to the edge orientation step of the PC algorithm. This is because the edge orientation step may eliminate some connections. In such cases, the undirected (skeleton) graph may be more useful. For all other algorithms, the causal discovery page only shows the `Download directed graph` button.
- For causal inference on discrete data, we only allow the use of multi-layer perceptron (MLP) as the prediction model, because linear regression performs poorly in this case.