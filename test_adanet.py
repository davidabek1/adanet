""" Created by David Abekasis March 2018
AdaNet: Adaptive Structural Learning of Artificial Neural Networks 

This helper module is used to implement and test the AdaNet_CVX.py API

- First part is loading 10 datasets for comparisons
- Second part is defining which classifiers will be used as baselines to test AdaNet (MLP-FFNN, LR)
- Third part is for scoring results and includes several functions to iterate and produce relevant scores

"""

from AdaNet_CVX import AdaNetCVX
import twospirals as ts_dataset
import AdaNet_CIFAR_10_feature_extraction as AdaFE

import csv
import itertools
import collections
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

#Datasets are stored in a python dictionary
datasets = {}
# verbose with pyplot graph of Loss convergence on every set of parameters
verbose = True
verbose_graph = True
# number of folds in each repetition, of which in each repetition a test set will be set aside
# the rest will be used for validation and training
n_fold_splits = 3
# Fetch test dataset of two spirals 

from sklearn.datasets import fetch_mldata
from sklearn.datasets.base import Bunch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, label_binarize
from sklearn.utils import shuffle
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

"""
#######################################
FIRST PART - Load datasets
#######################################

"""
data = ts_dataset.twospirals(44100,540*1,0,1)
datasets["twospirals"] = {
    "X": data[:,:-1],
    "y": (2*(data[:,-1])-1)
}
# Fetch german_data dataset from mldata.org
data = fetch_mldata("german-ida")
# training data has now zero mean and standard deviation one
datasets["german-ida"] = {
    "X": data.data,
    "y": (data.target)
}
# Fetch diabetes scale from mldata.org
data = fetch_mldata("diabetes_scale")
datasets["diabetes_scale"] = {
    "X": data.data,
    "y": label_binarize(data.target, classes=[-1, 1], neg_label=-1.0, pos_label=1.0).reshape(-1,).astype(float)
}
# Fetch wisconsin breast cancer from sklearn datasets
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
datasets["wisconsin"] = {
    "X": data.data,
    "y": label_binarize(data.target, classes=[0, 1], neg_label=-1.0, pos_label=1.0).reshape(-1,).astype(float)
}
# Fetch tic tac toe from UCI Maching Learning Repository
def load_tic_tac_toe(return_X_y=False):
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'tic-tac-toe', 'tic-tac-toe.data')
    with open(file_path) as f:
        data_file = csv.reader(f, delimiter=';')
        # first row of header, that will turn into feature_names
        # temp = next(data_file)
        # number of features = ??
        # feature_names = np.array(temp)
    data = []
    target = []
    with open(file_path) as f:
        data_file = csv.reader(f, delimiter=',')
        i = 0
        firstline = False
        for d in data_file:
            if firstline:
                # skip first line of header, as we already read it for features
                firstline = False
            else:
                d = np.array(d)
                #x=player x has taken, o=player o has taken, b=blank
                d[d == 'x'] = 1.0
                d[d == 'o'] = 2.0
                d[d == 'b'] = 3.0
                
                #positive (i.e., wins for "x")
                d[d == 'positive'] = 1.0
                d[d == 'negative'] = -1.0
                
                d = np.array(d, dtype='float')
                # load into data all column but the last, which is the target
                data.append(d[:-1])
                # load last column as target values
                target.append(d[-1])
                i += 1
    data = np.array(data).astype(float)
    target = np.array(target).astype(float)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target)
                 # last column is target value

data = load_tic_tac_toe()
datasets["tic_tac_toe"] = {
    "X": data.data,
    "y": data.target
}

# Fetch images of dog and horse from CIFAR10, 
# based on the helper python file AdaNet_CIFAR_10_feature_extraction.py
dog_horse = AdaFE.CF10_pairs('dog','horse')
X_train, y_train, X_test, y_test = AdaFE.train_test_dataset(dog_horse)
datasets['dog_horse'] = {
    'X': np.vstack((X_train,X_test)),
    'y': np.vstack((y_train,y_test))

}
# Fetch images of dog and horse from CIFAR10, 
# based on the helper python file AdaNet_CIFAR_10_feature_extraction.py
deer_horse = AdaFE.CF10_pairs('deer','horse')
X_train, y_train, X_test, y_test = AdaFE.train_test_dataset(deer_horse)
datasets['deer_horse'] = {
    'X': np.vstack((X_train,X_test)),
    'y': np.vstack((y_train,y_test))

}
# Fetch images of dog and horse from CIFAR10, 
# based on the helper python file AdaNet_CIFAR_10_feature_extraction.py
deer_truck = AdaFE.CF10_pairs('deer','truck')
X_train, y_train, X_test, y_test = AdaFE.train_test_dataset(deer_truck)
datasets['deer_truck'] = {
    'X': np.vstack((X_train,X_test)),
    'y': np.vstack((y_train,y_test))

}
# Fetch images of dog and horse from CIFAR10, 
# based on the helper python file AdaNet_CIFAR_10_feature_extraction.py
automobile_truck = AdaFE.CF10_pairs('automobile','truck')
X_train, y_train, X_test, y_test = AdaFE.train_test_dataset(automobile_truck)
datasets['automobile_truck'] = {
    'X': np.vstack((X_train,X_test)),
    'y': np.vstack((y_train,y_test))

}
# Fetch images of cat and dog from CIFAR10, 
# based on the helper python file AdaNet_CIFAR_10_feature_extraction.py
cat_dog = AdaFE.CF10_pairs('cat','dog')
X_train, y_train, X_test, y_test = AdaFE.train_test_dataset(cat_dog)
datasets['cat_dog'] = {
    'X': np.vstack((X_train,X_test)),
    'y': np.vstack((y_train,y_test))

}

"""
##############################################################################
SECOND PART - Define experiments configurations, and helper functions
##############################################################################

"""
def write_line(filename, dict, is_first=False):
    dict = collections.OrderedDict(sorted(dict.items()))
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=dict.keys())
        if is_first:
            writer.writeheader()
        writer.writerow(dict)


# return a list of experimet with all the relevant parameters
def get_index_product(params):
    i = 0
    params_index = {}
    for k, v in params.items():
        params_index[k] = i
        i += 1
    params_list = [None] * len(params_index.values())
    for name, loc in params_index.items():
        params_list[loc] = params[name]

    params_product = list(itertools.product(*params_list))
    params_product_dicts = []
    for params_value in params_product:
        params_dict = {}
        for param_name, param_index in params_index.items():
            params_dict[param_name] = params_value[param_index]
        params_product_dicts.append(params_dict)

    return params_product_dicts

params_adanets = {
    'maxLayers': [3], #[1,2,3],
    'maxNodes': [2000], #, 150, 512, 1024, 2048],
    'capitalLambda': [1.0, 1.045],  #, 1.045, 0.1, 1, 10, 100
    'Ck': [1],  # 0.1, 10, 100 
    'Ck_bias': [0.1],
    'lowerLambda': [1e-3],  #  ,1e-4,1e-5,1e-6
    'beta': [1e-3],
    'bolAugment': [True],  #False,
    'bolAugmentLayers': [True],  #False,
    'T': [50],  #300
    'optMethod': ['Nelder-Mead'],  #'Nelder-Mead' 'BFGS'
    'optIsGrad': [None]  # True  None
}
params_MLP = {
    'max_iter': [10], # similar to T, number of epochs
    'hidden_layer_sizes': [(10,),(10,10),(10,10,10)], #(2000,),(2000,2000),(2000,2000,2000)
    'alpha': [1e-3], # similar to lambda, a L2 regularization term
    'learning_rate_init': [0.001]
}
params_LR = {
    'tol': [0.001],
    'C': [1.0, 10.0]  #0.1, 1.0, 10.0 , 100.0
}
#generate all combination of experiments with diffrent parameters
experiments_params_adanets = get_index_product(params_adanets)
experiments_params_MLP = get_index_product(params_MLP)
experiments_params_LR = get_index_product(params_LR)

is_first_write_val_adanet = True
is_first_write_val_MLP = True
is_first_write_val_LR = True
is_first_test_write = True

# dataset_experiment = {
#     'wine_quality': load_wine_quality()
# }

def update_adanet_stats(stats, adanet_best_mean_accuracy, adanet_accuracy, adanet_fit_time):
    stats['adanet_fit_time'] = np.mean(adanet_fit_time)
    stats['adanet_mean'] = np.mean(adanet_accuracy)   
    stats['adanet_std'] = np.std(adanet_accuracy)            
    stats['numN'] = adanet_clf.adaParams['numNodes']
    stats['numL'] = adanet_clf.adaParams['numLayers']
    stats['lossLast'] = adanet_clf.adaParams['lossStore'][-1][0]
    stats['lossFirst'] = adanet_clf.adaParams['lossStore'][0][0]
    stats['lossFirstChange'] = np.mean(np.diff(np.hstack(adanet_clf.adaParams['lossStore'][:3])))
    stats['lossLastChange'] = np.mean(np.diff(np.hstack(adanet_clf.adaParams['lossStore'][-3:])))
    stats['actual_epochs'] = len(adanet_clf.adaParams['lossStore'])                
    stats.update(params_adanet)
    if adanet_best_mean_accuracy < stats['adanet_mean']:
        adanet_best_mean_accuracy=stats['adanet_mean']
    return stats, adanet_best_mean_accuracy

def update_model_stats(stats, model_best_mean_accuracy, model_accuracy, model_fit_time, model_params):
    stats[model+'_fit_time'] = np.mean(model_fit_time)
    stats[model+'_mean'] = np.mean(model_accuracy)   
    stats[model+'_std'] = np.std(model_accuracy)                           
    stats.update(model_params)
    if model_best_mean_accuracy < stats[model+'_mean']:
        model_best_mean_accuracy=stats[model+'_mean']
    return stats, model_best_mean_accuracy   

def update_test_stats(test_scores, model):
    test_scores = {
        model+"_dataset": dataset_name,
        model+"_accuracy": np.mean(test_accuracy[model]),
        model+"_accuracy_std": np.std(test_accuracy[model]),
        model+"_f1score": np.mean(test_f1score[model]),
        model+"_recall": np.mean(test_recall[model]),
        model+"_fit_time": np.mean(test_fit_time[model]),
        model+"_pred_time": np.mean(test_predict_time[model]),
        model+"_auc": np.mean(test_auc[model])
    }                        
    return test_scores

"""
##############################################################################
THIRD PART - RUN experiments (ADANET, MLP, LR)
##############################################################################

"""
# run experiment for each of the parameters
for dataset_name, dataset in datasets.items():
    X, y = shuffle(dataset['X'], dataset['y'], random_state=46)
    # X = X.astype(np.float32)
    # y = y.reshape(-1,1)

    stats = {}
    test_scores = {}
    test_scores['dataset'] = dataset_name
    stats['dataset'] = dataset_name
    if verbose:
        print (dataset_name)

    test_accuracy = {}
#    test_std = {}
    test_f1score = {}
    test_recall = {}
    test_fit_time = {}
    test_predict_time = {}
    test_auc = {}
    model = 'adanet'
    test_fit_time[model]=[]
    test_predict_time[model]=[]
    test_accuracy[model]=[]
    test_f1score[model]=[]
    test_recall[model]=[]
    test_auc[model]=[]

    adanet_best_experiment_mean_accuracy = []
    MLP_best_experiment_mean_accuracy = []
    LR_best_experiment_mean_accuracy = []   

################################
################################
#     Adanet scores
################################
################################    
    # using cv to look for best hyperparameters
    for params_adanet in experiments_params_adanets:

        k_fold = RepeatedKFold(n_splits=n_fold_splits, n_repeats=3, random_state=46) #10)

        #split_data = train_test_split(X, y, test_size=0.1, random_state=46)
        #X_train, X_test, y_train, y_test = split_data

        #evaluate each model and average in the end
        adanet_accuracy = []
        adanet_fit_time = []
        adanet_best_mean_accuracy = 0  # lower bound value to find highest accuracy
        
        is_first_fold = True

        for i, (train_indices, test_indices) in enumerate(k_fold.split(X)):
            # set aside true test set, that will not be trained or validated
            # and update the validation mean accuracy
            if i % n_fold_splits == 0:
                if is_first_fold:
                    is_first_fold = False
                else:
                    #is_first_fold = True
                    stats, adanet_best_mean_accuracy = update_adanet_stats(stats, adanet_best_mean_accuracy, adanet_accuracy, adanet_fit_time)
                    if verbose: 
                        print (stats)
                    write_line('adanet_val_results.csv', stats, is_first_write_val_adanet)
                    is_first_write_val_adanet = False
                    adanet_accuracy = []
                    adanet_fit_time = []
            else:
            # the test set here are actually the validation set
                X_train, y_train = X[train_indices], y[train_indices]
                X_test, y_test = X[test_indices], y[test_indices]

                adanet_clf = AdaNetCVX(**params_adanet)

                start_time = time.time()
                adanet_clf.fit(X_train, y_train)
                adanet_fit_time.append(time.time() - start_time)

                adanet_accuracy.append(accuracy_score(y_test,adanet_clf.predict(X_test)))

        # update values of last validation set
        stats, adanet_best_mean_accuracy = update_adanet_stats(stats, adanet_best_mean_accuracy, adanet_accuracy, adanet_fit_time)
        if verbose: 
            print (stats)
        write_line('adanet_val_results.csv', stats, is_first_write_val_adanet)

        if verbose_graph:
            plt.plot(adanet_clf.adaParams['lossStore'])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(str(params_adanet))
            plt.show()

        adanet_best_experiment_mean_accuracy.append(adanet_best_mean_accuracy)

    #print(adanet_best_experiment_mean_accuracy)
    best_index = (np.array(adanet_best_experiment_mean_accuracy)).argmax()
    #print(best_index, experiments_params_adanets[best_index])
    best_params_adanet = experiments_params_adanets[best_index]

    # start test session with best params
    for i, (train_indices, test_indices) in enumerate(k_fold.split(X)):
        # use the set aside true test set, to train and test accuracy
        if i % n_fold_splits == 0:
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            adanet_clf = AdaNetCVX(**best_params_adanet)

            start_time = time.time()
            adanet_clf.fit(X_train, y_train)
            test_fit_time[model].append(time.time() - start_time)

            start_time = time.time()
            pred = adanet_clf.predict(X_test)
            test_predict_time[model].append(time.time() - start_time)

            test_accuracy[model].append(accuracy_score(y_test,pred))
            test_f1score[model].append(f1_score(y_test,pred))
            test_recall[model].append(recall_score(y_test,pred))
            test_auc[model].append(roc_auc_score(y_test,adanet_clf.predict_proba(X_test)))

        else:
        # original train and validation sets to discard for this phase
            pass

    test_scores.update({'adanet_best_params': best_params_adanet, 
                    'adanet_numL': adanet_clf.adaParams['numLayers'], 
                    'adanet_numN': adanet_clf.adaParams['numNodes'] })
    test_scores.update(update_test_stats(test_scores, model))

################################
################################
#     MLP scores
################################
################################


    model = 'MLP'
    test_fit_time[model]=[]
    test_predict_time[model]=[]
    test_accuracy[model]=[]
    test_f1score[model]=[]
    test_recall[model]=[]
    test_auc[model]=[]
    stats = {}
    stats['dataset'] = dataset_name

    # using cv to look for best hyperparameters
    for params_MLP in experiments_params_MLP:

        k_fold = RepeatedKFold(n_splits=n_fold_splits, n_repeats=3, random_state=46) #10)

        #split_data = train_test_split(X, y, test_size=0.1, random_state=46)
        #X_train, X_test, y_train, y_test = split_data

        #evaluate each model and average in the end
        model_accuracy = []
        model_fit_time = []
        model_best_mean_accuracy = 0  # lower bound value to find highest accuracy
        
        is_first_fold = True

        for i, (train_indices, test_indices) in enumerate(k_fold.split(X)):
            # set aside true test set, that will not be trained or validated
            # and update the validation mean accuracy
            if i % n_fold_splits == 0:
                if is_first_fold:
                    is_first_fold = False
                else:
                    #is_first_fold = True
                    stats, model_best_mean_accuracy = update_model_stats(stats, model_best_mean_accuracy, model_accuracy, model_fit_time, params_MLP)
                    if verbose: 
                        print (stats)
                    write_line('MLP_val_results.csv', stats, is_first_write_val_MLP)
                    is_first_write_val_MLP = False
                    model_accuracy = []
                    model_fit_time = []
            else:
            # the test set here are actually the validation set
                X_train, y_train = X[train_indices], y[train_indices]
                X_test, y_test = X[test_indices], y[test_indices]

                clf = MLPClassifier(**params_MLP)

                start_time = time.time()
                clf.fit(X_train, y_train)
                model_fit_time.append(time.time() - start_time)

                model_accuracy.append(accuracy_score(y_test,clf.predict(X_test)))

        # update values of last validation set
        stats, model_best_mean_accuracy = update_model_stats(stats, model_best_mean_accuracy, model_accuracy, model_fit_time, params_MLP)
        if verbose: 
            print (stats)
        write_line('MLP_val_results.csv', stats, is_first_write_val_MLP)

        MLP_best_experiment_mean_accuracy.append(model_best_mean_accuracy)

    #print(adanet_best_experiment_mean_accuracy)
    best_index = (np.array(MLP_best_experiment_mean_accuracy)).argmax()
    #print(best_index, experiments_params_adanets[best_index])
    best_params_MLP = experiments_params_MLP[best_index]

    # start test session with best params
    for i, (train_indices, test_indices) in enumerate(k_fold.split(X)):
        # use the set aside true test set, to train and test accuracy
        if i % n_fold_splits == 0:
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            clf = MLPClassifier(**best_params_MLP)

            start_time = time.time()
            clf.fit(X_train, y_train)
            test_fit_time[model].append(time.time() - start_time)

            start_time = time.time()
            pred = clf.predict(X_test)
            test_predict_time[model].append(time.time() - start_time)

            test_accuracy[model].append(accuracy_score(y_test,pred))
            test_f1score[model].append(f1_score(y_test,pred))
            test_recall[model].append(recall_score(y_test,pred))
            test_auc[model].append(roc_auc_score(y_test,clf.predict_proba(X_test)[:,0]))

        else:
        # original train and validation sets to discard for this phase
            pass

    test_scores.update({'MLP_best_params': best_params_MLP })
    test_scores.update(update_test_stats(test_scores, model))

################################
################################
#     LR scores
################################
################################

    model = 'LR'
    test_fit_time[model]=[]
    test_predict_time[model]=[]
    test_accuracy[model]=[]
    test_f1score[model]=[]
    test_recall[model]=[]
    test_auc[model]=[]
    stats = {}
    stats['dataset'] = dataset_name

    # using cv to look for best hyperparameters
    for params_LR in experiments_params_LR:

        k_fold = RepeatedKFold(n_splits=n_fold_splits, n_repeats=3, random_state=46) #10)

        #split_data = train_test_split(X, y, test_size=0.1, random_state=46)
        #X_train, X_test, y_train, y_test = split_data

        #evaluate each model and average in the end
        model_accuracy = []
        model_fit_time = []
        model_best_mean_accuracy = 0  # lower bound value to find highest accuracy
        
        is_first_fold = True

        for i, (train_indices, test_indices) in enumerate(k_fold.split(X)):
            # set aside true test set, that will not be trained or validated
            # and update the validation mean accuracy
            if i % n_fold_splits == 0:
                if is_first_fold:
                    is_first_fold = False
                else:
                    #is_first_fold = True
                    stats, model_best_mean_accuracy = update_model_stats(stats, model_best_mean_accuracy, model_accuracy, model_fit_time, params_LR)
                    if verbose: 
                        print (stats)
                    write_line('LR_val_results.csv', stats, is_first_write_val_LR)
                    is_first_write_val_LR = False
                    model_accuracy = []
                    model_fit_time = []
            else:
            # the test set here are actually the validation set
                X_train, y_train = X[train_indices], y[train_indices]
                X_test, y_test = X[test_indices], y[test_indices]

                clf = LogisticRegression(**params_LR)

                start_time = time.time()
                clf.fit(X_train, y_train)
                model_fit_time.append(time.time() - start_time)

                model_accuracy.append(accuracy_score(y_test,clf.predict(X_test)))

        # update values of last validation set
        stats, model_best_mean_accuracy = update_model_stats(stats, model_best_mean_accuracy, model_accuracy, model_fit_time, params_LR)
        if verbose: 
            print (stats)
        write_line('LR_val_results.csv', stats, is_first_write_val_LR)

        LR_best_experiment_mean_accuracy.append(model_best_mean_accuracy)


    best_index = (np.array(LR_best_experiment_mean_accuracy)).argmax()

    best_params_LR = experiments_params_LR[best_index]

    # start test session with best params
    for i, (train_indices, test_indices) in enumerate(k_fold.split(X)):
        # use the set aside true test set, to train and test accuracy
        if i % n_fold_splits == 0:
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            clf = LogisticRegression(**best_params_LR)

            start_time = time.time()
            clf.fit(X_train, y_train)
            test_fit_time[model].append(time.time() - start_time)

            start_time = time.time()
            pred = clf.predict(X_test)
            test_predict_time[model].append(time.time() - start_time)

            test_accuracy[model].append(accuracy_score(y_test,pred))
            test_f1score[model].append(f1_score(y_test,pred))
            test_recall[model].append(recall_score(y_test,pred))
            test_auc[model].append(roc_auc_score(y_test,clf.predict_proba(X_test)[:,0]))

        else:
        # original train and validation sets to discard for this phase
            pass

    test_scores.update({'LR_best_params': best_params_LR })
    test_scores.update(update_test_stats(test_scores, model))

################################
################################
#     Finishing with comparing ttest
################################
################################
    # Adding t-test values, before writing all test restults into csv file
    tvalue, tprob = ttest_rel(test_accuracy['adanet'],test_accuracy['MLP'])
    test_scores.update({'AdaNet-MLP_ttest': tvalue, 'AdaNet-MLP_tprob': tprob})
    tvalue, tprob = ttest_rel(test_accuracy['adanet'],test_accuracy['LR'])
    test_scores.update({'AdaNet-LR_ttest': tvalue, 'AdaNet-LR_tprob': tprob})
    write_line('test_results.csv', test_scores, is_first_test_write)
    is_first_test_write = False
