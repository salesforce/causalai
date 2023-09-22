
from causalai.models.time_series.pc import PC as PC_timeseries
from causalai.models.time_series.granger import Granger
from causalai.models.time_series.var_lingam import VARLINGAM
from causalai.models.tabular.pc import PC as PC_tabular
from causalai.models.tabular.ges import GES
from causalai.models.tabular.lingam import LINGAM

from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.models.common.CI_tests.kci import KCI
from causalai.models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests
from causalai.data.data_generator import DataGenerator, GenerateRandomTimeseriesSEM, GenerateRandomTabularSEM

from causalai.models.tabular.causal_inference import CausalInference as CausalInference_tabular
from causalai.models.time_series.causal_inference import CausalInference as CausalInference_timeseries

from causalai.data.time_series import TimeSeriesData
from causalai.data.tabular import TabularData
from causalai.data.transforms.time_series import StandardizeTransform
from causalai.models.common.prior_knowledge import PriorKnowledge
from causalai.misc.misc import get_precision_recall

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import time
import matplotlib.pyplot as plt
import warnings
import json
import math
import numpy
from types import SimpleNamespace


# Generate synthetic time-series data
def get_timeseries_data(num_vars, max_num_parents=4, max_lag=4, T=1000, seed=0, discrete=False, intervention=None):
    # sem = {
    #         'a': [], 
    #         'b': [(('a', -1), coef, fn), (('f', -1), coef, fn)], 
    #         'c': [(('b', -2), coef, fn), (('f', -2), coef, fn)],
    #         'd': [(('b', -4), coef, fn), (('b', -1), coef, fn), (('g', -1), coef, fn)],
    #         'e': [(('f', -1), coef, fn)], 
    #         'f': [],
    #         'g': [],
    #         }
    var_names_all = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    assert int(num_vars)<=len(var_names_all), f'num_vars can be at most {len(var_names_all)} but found {int(num_vars)}'
    var_names = [var_names_all[i] for i in range(int(num_vars))]
    sem = GenerateRandomTimeseriesSEM(var_names=var_names, max_num_parents=max_num_parents, max_lag=int(max_lag), seed=int(seed))

    data_array, var_names, graph_gt = DataGenerator(sem, T=int(T), seed=int(seed), discrete=discrete, intervention=intervention)
    data_array_list = data_array.tolist()
    return (data_array_list, var_names, graph_gt)

# Generate synthetic time-series data
def get_tabular_data(num_vars, max_num_parents=4, T=1000, seed=0, discrete=False, intervention=None):
    # sem = {
    #         'a': [], 
    #         'b': [('a', coef, fn), ('f', coef, fn)], 
    #         'c': [('b', coef, fn), ('f', coef, fn)],
    #         'd': [('b', coef, fn), ('g', coef, fn)],
    #         'e': [('f', coef, fn)], 
    #         'f': [],
    #         'g': [],
    #         }
    var_names_all = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    assert int(num_vars)<=len(var_names_all), f'num_vars can be at most {len(var_names_all)} but found {int(num_vars)}'
    var_names = [var_names_all[i] for i in range(int(num_vars))]
    sem = GenerateRandomTabularSEM(var_names=var_names, max_num_parents=max_num_parents, seed=int(seed))

    data_array, var_names, graph_gt = DataGenerator(sem, T=int(T), seed=int(seed), intervention=intervention, discrete=discrete)
    data_array_list = data_array.tolist()
    return (data_array_list, var_names, graph_gt)
    # return {jsonify(data_array), jsonify(var_names), jsonify(graph_gt)}


def get_data(data_type, num_vars, max_num_parents, num_samples, random_seed, isDiscrete, max_lag=None, intervention=None):
    if data_type=='Time Series':
        data_array, var_names, causal_graph = get_timeseries_data(num_vars, max_num_parents=max_num_parents, max_lag=max_lag,\
                                                T=num_samples, seed=random_seed, discrete=isDiscrete, intervention=intervention)
    elif data_type=='Tabular':
        data_array, var_names, causal_graph = get_tabular_data(num_vars, max_num_parents=max_num_parents, T=num_samples,\
                                                seed=random_seed, discrete=isDiscrete, intervention=intervention)
    else:
        raise ValueError(f'data_type must be in [Time Series, Tabular], but found {data_type}')
    return data_array, var_names, causal_graph

# Initialize CausalAI TimeSeries data object
def get_causalai_timeseries_data_obj(data_array, var_names):
    # 0 mean, unit variance transformation
    StandardizeTransform_ = StandardizeTransform()
    StandardizeTransform_.fit(data_array)
    data_trans = StandardizeTransform_.transform(data_array)
    data_obj = TimeSeriesData(data_trans, var_names=var_names)
    return data_obj


from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route("/generate_data", methods=["POST"])
def generate_data_api():
    data_type = request.form['data_type']
    if(request.form['isDiscrete'] == 'false'):
        isDiscrete = False
    else:
        isDiscrete = True
    num_vars = request.form['num_vars']
    num_samples = request.form['num_samples']
    max_lag = request.form['max_lag']
    random_seed = request.form['random_seed']

    max_num_parents = 4

    data_array, var_names, causal_graph = get_data(data_type=data_type, num_vars=num_vars, max_num_parents=max_num_parents, max_lag=max_lag,\
                                    num_samples=num_samples, random_seed=random_seed, isDiscrete=isDiscrete)
    '''
    data_array: ndarray of shape (num_samples, num_vars)
    var_names: list of length num_vars. E.g. ['a', 'b', 'c']
    '''
    return jsonify({'data_array': data_array, 'var_names':var_names, 'causal_graph':causal_graph}) # plot causal_graph


@app.route("/perform_causal_discovery", methods=["POST"])
def perform_causal_discovery_api():
    if(request.form['isDiscrete'] == 'false'):
        isDiscrete = False
    else:
        isDiscrete = True
    data_array_json = request.form['data_array']
    var_names_json = request.form['var_names']
    causal_graph_json = request.form['causal_graph']
    prior_knowledge_json = request.form['prior_knowledge']

    data_type = request.form['data_type']
    data_array = json.loads(data_array_json)
    var_names = json.loads(var_names_json)
    causal_graph = json.loads(causal_graph_json)
    prior_knowledge = json.loads(prior_knowledge_json)

    existing_links = prior_knowledge.get('existing_links', {})
    forbidden_links = prior_knowledge.get('forbidden_links', {})
    root_variables = prior_knowledge.get('root_nodes', [])
    leaf_variables = prior_knowledge.get('leaf_nodes', [])

    prior_knowledge = PriorKnowledge(forbidden_links=forbidden_links,
                                     existing_links=existing_links,
                                     root_variables=root_variables,
                                     leaf_variables=leaf_variables)

    if(data_type == 'Time Series'):
        max_lag = int(request.form['max_lag'])

    algorithm = request.form['algorithm']
    ci_test_input = request.form['ci_test']
    pvalue_thres = float(request.form['pvalue'])

    if not isDiscrete:
        StandardizeTransform_ = StandardizeTransform()
        StandardizeTransform_.fit(data_array)
        data_trans = np.array(StandardizeTransform_.transform(data_array))
    else:
        data_trans = np.array(data_array)

    if data_type=='Time Series':
        data_obj = TimeSeriesData(data_trans, var_names=var_names)
    elif data_type=='Tabular':
        data_obj = TabularData(data_trans, var_names=var_names)
    else:
        raise ValueError(f'data_type must be in [Time Series, Tabular], but found {data_type}')


    if data_type=='Time Series':
        assert max_lag is not None and type(max_lag)==int and max_lag>=0
        # assert algorithm in ['PC', 'Granger', 'VARLINGAM']
        if algorithm=='PC':
            assert ci_test_input is not None
            if ci_test_input=='Partial Correlation':
                CI_test = PartialCorrelation()
            elif ci_test_input=='Pearson (discrete variables)':
                CI_test = DiscreteCI_tests(method="pearson")
            else:
                raise ValueError(f'ci_test must be in [Partial Correlation, Pearson (discrete variables)], but found {ci_test}')
            model = PC_timeseries(
                                data=data_obj,
                                prior_knowledge=prior_knowledge,
                                CI_test=CI_test,
                                use_multiprocessing=True
                                )
            result = model.run(pvalue_thres=pvalue_thres, max_lag=max_lag, max_condition_set_size=4)
        elif algorithm=='Granger':
            tic = time.time()
            model = Granger(
                                data=data_obj,
                                prior_knowledge=prior_knowledge,
                                max_iter=1000,
                                use_multiprocessing=False)
            result = model.run(pvalue_thres=pvalue_thres, max_lag=max_lag)
            toc = time.time()
            print(f'Granger, time taken: {toc-tic:.2f}s')
        elif algorithm=='VARLINGAM':
            model = VARLINGAM(data=data_obj, prior_knowledge=prior_knowledge, use_multiprocessing=False)
            result = model.run(pvalue_thres=pvalue_thres, max_lag=max_lag)
        else:
            raise ValueError(f'algorithm {algorithm} not supported.')

    elif data_type=='Tabular':
        if algorithm=='PC':
            assert ci_test_input is not None
            if ci_test_input=='Partial Correlation':
                CI_test = PartialCorrelation()
            elif ci_test_input=='Pearson (discrete variables)':
                CI_test = DiscreteCI_tests(method="pearson")
            else:
                raise ValueError(f'ci_test must be in [Partial Correlation, Pearson (discrete variables)], but found {ci_test}')
            model = PC_tabular(
                                data=data_obj,
                                prior_knowledge=prior_knowledge,
                                CI_test=CI_test,
                                use_multiprocessing=True
                                )
            result = model.run(pvalue_thres=pvalue_thres, max_condition_set_size=4)
        elif algorithm=='GES':
            model = GES(data=data_obj, prior_knowledge=prior_knowledge, use_multiprocessing=False)
            result = model.run(pvalue_thres=pvalue_thres)
        elif algorithm=='LINGAM':
            model = LINGAM(data=data_obj, prior_knowledge=prior_knowledge, use_multiprocessing=False)
            result = model.run(pvalue_thres=pvalue_thres)
        else:
            raise ValueError(f'algorithm {algorithm} not supported.')

    else:
        raise ValueError(f'data_type must be in [Time Series, Tabular], but found {data_type}')


    graph_est={n:[] for n in result.keys()}
    graph_est_undirected = {n:[] for n in result.keys()}
    if(data_type == 'Time Series'):
        for key in result.keys():
            parent = result[key]['parents']
            parents = []
            for i in range(len(parent)):
                parents.append(list(parent[i]))
            graph_est[key].extend(parents)
    elif(data_type == 'Tabular'):
        for key in result.keys():
            parents = result[key]['parents']
            graph_est[key].extend(parents)
            if algorithm=='PC':
                graph_est_undirected = model.skeleton
    precision, recall, f1_score = [None]*3

    if len(causal_graph) != 0 : # if ground truth causal_graph iss available, compute precision/recall
        precision, recall, f1_score = get_precision_recall(graph_est, causal_graph)

    '''
    graph_est:  Sample value for data_type=Time Series,
                            sem = {
                                    'a': [], 
                                    'b': [('a', -1), ('f', -1)], 
                                    'c': [('b', -2), ('f', -2)],
                                    'd': [('b', -4), ('b', -1), ('g', -1)],
                                    'e': [('f', -1)], 
                                    'f': [],
                                    'g': [],
                                    }
                Sample value for data_type=Tabular,
                            sem = {
                                    'a': [], 
                                    'b': ['a', 'f'], 
                                    'c': ['b', 'f'],
                                    'd': ['b', 'g'],
                                    'e': ['f'], 
                                    'f': [],
                                    'g': [],
                                    }
    precision: scalar or None
    recall: scalar or None
    f1_score: scalar or None
    '''
    return jsonify({'graph_est':graph_est, 'graph_est_undirected':graph_est_undirected,\
                    'precision':precision, 'recall':recall, 'f1_score':f1_score}) # plot graph_est, and print precision, recall, f1_score if they are not None


def define_treatments(name, t,c):
    treatment = dict(var_name=name,
                    treatment_value=t,
                    control_value=c)
    return treatment

@app.route("/ate", methods=["POST"])
def perform_causal_inference_ate_api():
    data_type = request.form['data_type']

    data_array = request.form['data_array']
    var_names = request.form['var_names']
    causal_graph = request.form['causal_graph']

    target_var = request.form['target_var']
    prediction_model = request.form['prediction_model']
    treatments = request.form['treatments']

    if(request.form['isDiscrete'] == 'false'):
        isDiscrete = False
    else:
        isDiscrete = True
    random_seed = request.form['random_seed']
    if(request.form['isDataGenerated'] == 'false'):
        isDataGenerated = False
    else:
        isDataGenerated = True

    num_vars = int(request.form['num_vars'])
    num_samples = int(request.form['num_samples'])


    var_names = json.loads(var_names)
    treatments = json.loads(treatments)
    data_array = json.loads(data_array)
    causal_graph = json.loads(causal_graph)

    max_num_parents = 4
    if prediction_model=='Linear Regression':
        prediction_model = LinearRegression
    elif prediction_model=='MLP Regression':
        prediction_model = MLPRegressor
    else:
        raise ValueError(f'Only [Linear Regression, MLP Regression] are supported for prediction_model, but got {prediction_model}.')

    true_ate = None
    if isDataGenerated:
        if(data_type == 'Time Series'):
            max_lag = int(request.form['max_lag'])
        else:
            max_lag = None
        treatment_vals_dict = {v:t*np.ones((int(num_samples),)) for (v,t,_) in treatments}
        control_vals_dict = {v:c*np.ones((int(num_samples),)) for (v,_,c) in treatments}
        treatment_data_list, _, _ = get_data(data_type, num_vars, max_num_parents, num_samples, random_seed, isDiscrete, max_lag, intervention=treatment_vals_dict)
        control_data_list, _, _ = get_data(data_type, num_vars, max_num_parents, num_samples, random_seed, isDiscrete, max_lag, intervention=control_vals_dict)
        
        target_var_idx = var_names.index(target_var)
        treatment_data = numpy.array(treatment_data_list)
        control_data = numpy.array(control_data_list)
        true_ate = (treatment_data[:,target_var_idx] - control_data[:,target_var_idx]).mean()

    intervention_lod = []
    for intervention in treatments:
        intervention_lod.append(define_treatments(intervention[0],intervention[1],intervention[2]))


    if data_type=='Time Series':
        causal_graph = format_ts_graph(causal_graph)
        CausalInference_ = CausalInference_timeseries(numpy.array(data_array), var_names, causal_graph, prediction_model, discrete=isDiscrete)
    else:
        CausalInference_ = CausalInference_tabular(numpy.array(data_array), var_names, causal_graph, prediction_model, discrete=isDiscrete)
    est_ate, _,_ = CausalInference_.ate(target_var, intervention_lod)

    '''
    est_ate: scalar
    true_ate: scalar or None
    '''

    if((est_ate)!=None and math.isnan(est_ate)):
        est_ate='NaN'

    if((true_ate)!=None and math.isnan(true_ate)):
        true_ate='NaN'

    if(est_ate is None):
        est_ate='-'

    if(true_ate is None):
        true_ate='-'


    return jsonify({'est_ate':est_ate, 'true_ate':true_ate})


@app.route("/cate", methods=["POST"])
def perform_causal_inference_cate_api():
    data_type = request.form['data_type']

    data_array = request.form['data_array']
    var_names = request.form['var_names']
    causal_graph = request.form['causal_graph']

    target_var = request.form['target_var']
    prediction_model = request.form['prediction_model']
    treatments = request.form['treatments']

    conditions = request.form['conditions']
    condition_prediction_model = request.form['condition_prediction_model']

    if(request.form['isDiscrete'] == 'false'):
        isDiscrete = False
    else:
        isDiscrete = True
    random_seed = request.form['random_seed']
    if(request.form['isDataGenerated'] == 'false'):
        isDataGenerated = False
    else:
        isDataGenerated = True

    num_vars = int(request.form['num_vars'])
    num_samples = int(request.form['num_samples'])
    

    data_array = json.loads(data_array)
    var_names = json.loads(var_names)
    causal_graph = json.loads(causal_graph)
    treatments = json.loads(treatments)
    conditions = json.loads(conditions)

    data_array = np.array(data_array)

    max_num_parents = 4

    if prediction_model=='Linear Regression':
        prediction_model = LinearRegression
    elif prediction_model=='MLP Regression':
        prediction_model = MLPRegressor
    else:
        raise ValueError(f'Only [Linear Regression, MLP Regression] are supported for prediction_model, but got {prediction_model}.')

    if condition_prediction_model=='Linear Regression':
        condition_prediction_model = LinearRegression
    elif condition_prediction_model=='MLP Regression':
        condition_prediction_model = MLPRegressor
    else:
        raise ValueError(f'Only [Linear Regression, MLP Regression] are supported for prediction_model, but got {condition_prediction_model}.')


    intervention_lod = []
    for intervention in treatments:
        intervention_lod.append(define_treatments(intervention[0],intervention[1],intervention[2]))

    if data_type=='Time Series':
        causal_graph = format_ts_graph(causal_graph)
        CausalInference_ = CausalInference_timeseries(np.array(data_array), var_names, causal_graph, prediction_model, discrete=isDiscrete)
    else:
        CausalInference_ = CausalInference_tabular(np.array(data_array), var_names, causal_graph, prediction_model, discrete=isDiscrete)

    conditions_lod = []
    for condition in conditions:
        conditions_lod.append({'var_name': condition[0], 'condition_value': condition[1]})
    est_cate = CausalInference_.cate(target_var, intervention_lod, conditions_lod, condition_prediction_model)

    is_treatment_relevant = CausalInference_.is_treatment_relevant


    true_cate = None
    if isDataGenerated:
        if is_treatment_relevant is False:
            true_cate = float(0.)
        else:
            assert len(conditions)==1, 'When using generated data, only single condition variables are supported'
            if(data_type == 'Time Series'):
                max_lag = int(request.form['max_lag'])
            else:
                max_lag = None

            treatment_vals_dict = {v:t*np.ones((num_samples,)) for (v,t,_) in treatments}
            control_vals_dict = {v:c*np.ones((num_samples,)) for (v,_,c) in treatments}
            treatment_data, _, _ = get_data(data_type, num_vars, max_num_parents, num_samples, random_seed, isDiscrete, max_lag, intervention=treatment_vals_dict)
            control_data, _, _ = get_data(data_type, num_vars, max_num_parents, num_samples, random_seed, isDiscrete, max_lag, intervention=control_vals_dict)
            target_var_idx = var_names.index(target_var)

            condition_var_name = conditions[0][0]
            control_var_idx = var_names.index(str(condition_var_name))
            condition_state = conditions[0][1]

            treatment_data = np.array(treatment_data)
            control_data = np.array(control_data)
            
            diff = np.abs(data_array[:,control_var_idx] - condition_state)
            idx = np.argmin(diff)
            # assert diff[idx]<0.1, f'No observational data exists for the conditional variable close to {condition_state}'
            true_cate = (treatment_data[idx,target_var_idx] - control_data[idx,target_var_idx])
            true_cate = float(true_cate)

    
    '''
    est_cate: scalar
    true_cate: scalar or None
    '''
    if((est_cate)!=None and math.isnan(est_cate)):
        est_cate='NaN'

    if((true_cate)!=None and math.isnan(true_cate)):
        true_cate='NaN'

    if(est_cate is None):
        est_cate='-'

    if(true_cate is None):
        true_cate='-'

    return jsonify({'est_cate':est_cate, 'true_cate':(true_cate)})

@app.route("/counterfactual", methods=["POST"])
def perform_causal_inference_counterfactual_api():
    data_type = request.form['data_type']

    data_array = request.form['data_array']
    var_names = request.form['var_names']
    causal_graph = request.form['causal_graph']

    target_var = request.form['target_var']
    prediction_model = request.form['prediction_model']
    treatments = request.form['treatments']

    conditions = request.form['conditions']
    condition_prediction_model = request.form['condition_prediction_model']

    if(request.form['isDiscrete'] == 'false'):
        isDiscrete = False
    else:
        isDiscrete = True
    random_seed = request.form['random_seed']
    if(request.form['isDataGenerated'] == 'false'):
        isDataGenerated = False
    else:
        isDataGenerated = True

    num_vars = int(request.form['num_vars'])
    num_samples = int(request.form['num_samples'])
    

    data_array = json.loads(data_array)
    var_names = json.loads(var_names)
    causal_graph = json.loads(causal_graph)
    treatments = json.loads(treatments)
    conditions = json.loads(conditions)

    data_array = np.array(data_array)

    max_num_parents = 4

    if prediction_model=='Linear Regression':
        prediction_model = LinearRegression
    elif prediction_model=='MLP Regression':
        prediction_model = MLPRegressor
    else:
        raise ValueError(f'Only [Linear Regression, MLP Regression] are supported for prediction_model, but got {prediction_model}.')

    if condition_prediction_model=='Linear Regression':
        condition_prediction_model = LinearRegression
    elif condition_prediction_model=='MLP Regression':
        condition_prediction_model = MLPRegressor
    else:
        raise ValueError(f'Only [Linear Regression, MLP Regression] are supported for prediction_model, but got {condition_prediction_model}.')

    # compute the 1D sample array
    sample = []
    condition_names = [conditions[i][0] for i in range(len(conditions))]
    condition_vals = [conditions[i][1] for i in range(len(conditions))]

    treatment_names = [treatments[i][0] for i in range(len(treatments))]
    treatment_vals = [treatments[i][1] for i in range(len(treatments))]

    for var in var_names:
        if var in treatment_names:
            idx = treatment_names.index(var)
            sample.append(treatment_vals[idx])
        elif var in condition_names:
            idx = condition_names.index(var)
            sample.append(condition_vals[idx])
        else:
            assert var==target_var
            sample.append(0.)
    sample = np.array(sample).reshape(-1)

    intervention_dict = {}
    for intervention in treatments:
        intervention_dict[intervention[0]] = intervention[1]

    if data_type=='Time Series':
        causal_graph = format_ts_graph(causal_graph)
        CausalInference_ = CausalInference_timeseries(np.array(data_array), var_names, causal_graph, prediction_model, discrete=isDiscrete)
    else:
        CausalInference_ = CausalInference_tabular(np.array(data_array), var_names, causal_graph, prediction_model, discrete=isDiscrete)


    est_counterfactual = CausalInference_.counterfactual(sample, target_var, intervention_dict, condition_prediction_model)

    is_treatment_relevant = CausalInference_.is_treatment_relevant

    true_counterfactual = None
    if isDataGenerated:
        target_var_idx = var_names.index(target_var)
        if is_treatment_relevant is False:
            true_counterfactual = float(sample[target_var_idx])
        else:
            if(data_type == 'Time Series'):
                max_lag = int(request.form['max_lag'])
            else:
                max_lag = None

            treatment_vals_dict = {v:t*np.ones((num_samples,)) for (v,t,_) in treatments}
            treatment_data, _, _ = get_data(data_type, num_vars, max_num_parents, num_samples, random_seed, isDiscrete, max_lag, intervention=treatment_vals_dict)
            treatment_data = np.array(treatment_data)

            diff = np.abs(treatment_data - sample.reshape(1,-1))
            idx = np.argmin(diff)
            # assert diff[idx]<0.1, f'No observational data exists for the conditional variable close to {condition_state}'
            true_counterfactual = (treatment_data[idx, target_var_idx])
            true_counterfactual = float(true_counterfactual)

    
    '''
    est_counterfactual: scalar
    true_counterfactual: scalar or None
    '''
    if((est_counterfactual)!=None and math.isnan(est_counterfactual)):
        est_counterfactual='NaN'

    if((true_counterfactual)!=None and math.isnan(true_counterfactual)):
        true_counterfactual='NaN'

    if(est_counterfactual is None):
        est_counterfactual='-'

    if(true_counterfactual is None):
        true_counterfactual='-'

    return jsonify({'est_counterfactual':est_counterfactual, 'true_counterfactual':(true_counterfactual)})

# @app.route("/undirected_edges", methods=["POST"])
# def get_undirected_edges():
#     graph = request.form['causal_graph']
#     graph = json.loads(graph)

#     undirected_edge_list = set()
#     for key in graph.keys():
#         candidate_parents = graph[key]
#         for p in candidate_parents:
#             if key in graph[p]:             
#                 if not ((key, p) in undirected_edge_list or (p, key) in undirected_edge_list):
#                     undirected_edge_list.add((key, p))

#     undirected_edge_list = list(undirected_edge_list) 
    
#     return jsonify({'undirected_edges': undirected_edge_list})
@app.route("/undirected_edges", methods=["POST"])
def get_undirected_edges():
    '''
    return undirected edge list in a given graph
    '''
    data_type = request.form['data_type']
    graph = request.form['causal_graph']
    graph = json.loads(graph)

    if data_type=='Tabular':
        undirected_edge_list = set()
        for key in graph.keys():
            parents = graph[key]
            for p in parents:
                if key in graph[p]:             
                    if not ((key, p) in undirected_edge_list or (p, key) in undirected_edge_list):
                        undirected_edge_list.add((key, p))

        undirected_edge_list = list(undirected_edge_list)
    else:
        # extract instantaneous connections only
        graph_tab = {}
        for key in graph.keys():
            graph_tab[key] = []
            parents = graph[key]
            for p in parents:
                if p[1]==0:
                    graph_tab[key].append(p[0])
        undirected_edge_list = set()
        for key in graph_tab.keys():
            parents = graph_tab[key]
            for p in parents:
                if key in graph_tab[p]:             
                    if not ((key, p) in undirected_edge_list or (p, key) in undirected_edge_list):
                        undirected_edge_list.add((key, p))
    
    return jsonify({'undirected_edges': undirected_edge_list})

def format_ts_graph(graph):
    '''
    in time series graph dictionary, each value is a list of tuples. But if user provides list of list, fix the format.
    '''
    graph_new = {}
    for key in graph.keys():
        graph_new[key] = []
        parents = graph[key]
        for p in parents:
            graph_new[key].append(tuple(p))
    return  graph_new


@app.route("/find_discrete_data_max_state", methods=["POST"])
def find_discrete_data_max_state():
    '''
    for discrete data, when doing causal inference, treatment/condition vars can only take discrete vals as inputs and their
    state is upper bounded by what's given in data. So find that upper bound so the UI can control the possible states
    '''
    data_array = request.form['data_array']
    var_names = request.form['var_names']

    data_array = json.loads(data_array)
    var_names = json.loads(var_names)
    data_array = np.array(data_array)

    max_state_val = {}
    for i in range(data_array.shape[1]):
        v = np.max(data_array[:,i])
        max_state_val[var_names[i]]=int(v)
    return jsonify({'max_state_val': max_state_val})

@app.route("/has_undirected_edges", methods=["POST"])
def has_undirected_edges():
    graph = request.form['graph']
    graph = json.loads(graph)
    data_type = 'time_series' if any([type(p[0]) in [list, tuple] for c,p in graph.items() if len(p)>0]) else 'tabular'
    msg = ''
    for child in graph.keys():
        parents = graph[child]
        for p in parents:
            child_t = child_l = child
            if data_type=='time_series':
                p_id = p
                p = p[0] # only keep variable name in case p is of the form (a, -2)
                child_t = (child,0)
                child_l = [child,0]
                
            if p not in graph: # if the parent node is not one of the keys, then this graph format is invalid
                msg = f'All variables in the causal graph must be part of the graph dictionary keys. '\
                      f'Found {p} which is not a key in the graph dictionary.'
                return jsonify({'bool': msg!='', 'msg': msg})
            if data_type=='time_series':
                if (child_t in graph[p] or child_l in graph[p]) and p_id[1]==0:
                    msg = f'Found an undirected edge between variables {p} and {child}. Please direct all the edges '\
                        f'in the graph before uploading the file.'
            else:
                if child in graph[p]:
                    msg = f'Found an undirected edge between variables {p} and {child}. Please direct all the edges '\
                        f'in the graph before uploading the file.'
                    return jsonify({'bool': msg!='', 'msg': msg})
    return jsonify({'bool': msg!='', 'msg': msg})

@app.route("/check_causal_graph_format", methods=["POST"])
def check_causal_graph_format():
    '''
    Examples:
    
        Tabular graph:

        graph = {'A': ['B', 'C'],
             'B': ['C', 'D'], 'C': []}

        Time series graph:

        graph = {'A': [('B',-1), ('C', -5)],
             'B': [('C', -1), ('D',-3)]}
    '''
    graph = request.form['graph']
    graph = json.loads(graph)
    
    if type(graph)!=dict: 
        msg = 'graph variable must be a dictionary'
        return jsonify({'bool': False, 'msg': msg})
    graph_type = None
    for child in graph.keys():
        if type(graph[child])!=list: 
            msg = f'graph values must be lists, but found {type(graph[child])} {graph[child]}'
            return jsonify({'bool': False, 'msg': msg})
        for e in graph[child]:
            if type(e) in [tuple, list]:
                graph_type_ = 'time_series'
                if len(e)!=2:
                    msg = f'Time series causal graphs must have edges in the form of a 2-tuple: (var_name, lag),'\
                            f' but found {e}'
                    return jsonify({'bool': False, 'msg': msg})
                if type(e[1])!=int or e[1]>0:
                    msg = f'Time series causal graphs must have edges in the form of a 2-tuple: (var_name, lag),'\
                            f' where lag is a non-positive integer, but found {e}'
                    return jsonify({'bool': False, 'msg': msg})
            elif type(e) in [int, str]:
                graph_type_ = 'tabular'
            else:
                graph_type_ = None
                msg = f'The values of all graph keys must be a list of either of type'\
                    f' tuple/list for time series data, or of type str or int for'\
                    f' tabular data but found {type(graph[child])}.'
                return jsonify({'bool': False, 'msg': msg})
            if graph_type is None:
                graph_type = graph_type_
            else:
                if graph_type!=graph_type_:
                    msg = f'The values of all keys of the'\
                        f'variable graph must be of the same type.'
                    return jsonify({'bool': False, 'msg': msg})
    return jsonify({'bool': True, 'msg': ''})

@app.route("/is_latest_link_valid", methods=["POST"])
def is_latest_link_valid():
    data_type = request.form['data_type']
    prior_knowledge = request.form['prior_knowledge']
    prior_knowledge = json.loads(prior_knowledge)

    existing_links = prior_knowledge.get('existing_links', {})
    forbidden_links = prior_knowledge.get('forbidden_links', {})
    root_variables = prior_knowledge.get('root_nodes', [])
    leaf_variables = prior_knowledge.get('leaf_nodes', [])
    # parents specified in existing_links cannot exist if the key variable is specified in root_variables
    # parents specified in existing_links must not conflict with forbidden_links
    err = ''
    all_children = []
    all_parents = []
    for v in existing_links.keys():
        if len(existing_links[v])!=0:
            all_children.append(v)
            all_parents.extend(list(existing_links[v]))
        if v in forbidden_links.keys():
            print(f'checking if node {v} has intersecting parents in both forbidden_links and existing_links')
            flag, intersection = _is_intersection(forbidden_links[v], existing_links[v])
            if flag:
                err= f'The variable {v} is specified as a child of node(s) {intersection}'\
                    f' in the argument existing_links,'\
                    f' but {v} is also specified as a forbidden child of node(s) {intersection} in forbidden_links.'
    for v in root_variables:
        if v in all_children:
            err = f'The variable {v} is specified as a child in the argument existing_links,'\
                       f' but {v} is also specified as a root_variable which cannot have parents.'

    for v in leaf_variables:
        if v in all_parents:
            err = f'The variable {v} is specified as a parent in the argument existing_links,'\
                       f' but {v} is also specified as a leaf_variable which cannot have children.'
            
    if data_type=='Tabular':
        var_names = list(set((all_children) + (all_parents)))
        isCyclic = TabularGraph(existing_links, var_names).isCyclic()
        if isCyclic:
            err = 'Adding the latest existing_link is creating a cycle! Cycles are not allowed.'
    return  jsonify({'bool': err=='', 'string':err })

### helper code below ###


class TabularGraph:
    def __init__(self, G, var_names):
        self.causal_graph = G
        self.var_names = var_names
        self.num_nodes = len(var_names)
        self.construct_full_graph_dict()
        
        self.adj_graph, _ = self.get_adjacency_graph(G)

    def construct_full_graph_dict(self):
        '''
        Verify that all nodes in causal_graph are listed in var_names, and if 
        any node is missing in causal_graph.keys(), add it with an empty list 
        of parents as the corresponding value.
        '''
        all_nodes = []
        for child, parents in self.causal_graph.items():
            if child not in all_nodes:
                all_nodes.append(child)

            for parent in parents:
                if parent not in all_nodes:
                    all_nodes.append(parent)
        all_nodes = set(all_nodes)
        assert len(all_nodes) - len(set(self.var_names))==0,\
            f'Oops, there are nodes in the causal_graph ({(all_nodes) - (set(self.var_names))}) which are '\
            f'missing in var_names! var_names must contain all the nodes.'
                    
        for node in self.var_names:
            if node not in self.causal_graph.keys():
                self.causal_graph[node] = []

    def _isCyclic(self, v, visited, visited_during_recusion):
        visited[v] = True
        visited_during_recusion[v] = True

        for child in self.adj_graph[self.var_names[v]]:
            child = self.var_names.index(child)
            if visited[child] == False:
                if self._isCyclic(child, visited, visited_during_recusion) == True:
                    return True
            elif visited_during_recusion[child] == True:
                return True

        visited_during_recusion[v] = False
        return False

    def isCyclic(self):
        '''
        Check if the causal graph has cycles among the non-lagged connections
        '''
        visited = [False] * (self.num_nodes + 1)
        visited_during_recusion = [False] * (self.num_nodes + 1)
        for node in range(self.num_nodes):
            if visited[node] == False:
                if self._isCyclic(node,visited,visited_during_recusion) == True:
                    return True
        return False
    def get_adjacency_graph(self, graph):
        '''
        Given graph where keys are children and values are parents, convert to adjacency graph where
        keys are parents and values are children.
        '''
        ad_graph = dict()
        all_nodes = []
        for child, parents in graph.items():
            if child not in all_nodes:
                all_nodes.append(child)

            for parent in parents:
                if parent not in all_nodes:
                    all_nodes.append(parent)

                if parent in ad_graph:
                    ad_graph[parent].append(child)
                else:
                    ad_graph[parent] = [child]
        for node in all_nodes:
            if node not in ad_graph.keys():
                ad_graph[node] = []
        return ad_graph, all_nodes

def _is_intersection(lst1, lst2):
    intersection = list(set(lst1) & set(lst2))
    flag = len(intersection) > 0
    return flag, intersection



