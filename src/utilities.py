from __future__ import division
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

import pickle
import os


def get_data(path):
    '''
    loads the data specified on the path and puts it on a dataframe
    input: path (srt)
    output: dataframe
    '''

    x = np.load(path)['x']
    x = x[~np.isinf(x['is_fishing']) & ~np.isnan(x['is_fishing']) & ~np.isnan(
        x['timestamp']) & ~np.isnan(x['speed']) & ~np.isnan(x['course'])]
    df = pd.DataFrame(x)
    return df


def get_all_data(dir=None):
    '''
    it will loop over the files in dir and load them
    input: directory name (srt)
    output: dictionary of dataframes with key = name of file
    '''
    if dir is None:
        dir = os.path.join(os.path.dirname(__file__), '.', 'data/labeled')

    datasets = {}
    for filename in os.listdir(dir):
        if filename.endswith('.measures.labels.npz'):
            name = filename[:-len('.measures.labels.npz')]
            x = np.load(os.path.join(dir, filename))['x']
            # print(filename)
            x = x[~np.isinf(x['is_fishing']) & ~np.isnan(x['is_fishing']) & ~np.isnan(
                x['timestamp']) & ~np.isnan(x['speed']) & ~np.isnan(x['course'])]
            datasets[name] = pd.DataFrame(x)
    return datasets


def get_group(data_dict, gear_type):

    group_per_gear = {

        'longliners':
        ['alex_crowd_sourced_Drifting_longlines',
         'kristina_longliner_Drifting_longlines',
         'pybossa_project_3_Drifting_longlines'],

        'purse_seines':
        ['alex_crowd_sourced_Purse_seines',
         'kristina_ps_Purse_seines',
         'pybossa_project_3_Purse_seines'],

        'trawlers':
        ['kristina_trawl_Trawlers',
         'pybossa_project_3_Trawlers'],

        'others':
        ['alex_crowd_sourced_Unknown',
         'kristina_longliner_Fixed_gear',
         'kristina_longliner_Unknown',
         'kristina_ps_Unknown',
         'pybossa_project_3_Unknown',
         'pybossa_project_3_Pole_and_line',
         'pybossa_project_3_Trollers'
         'pybossa_project_3_Fixed_gear'],

        'false_positives':
        ['false_positives_Drifting_longlines',
         'false_positives_Fixed_gear',
         'false_positives_Purse_seines',
         'false_positives_Trawlers',
         'false_positives_Unknown']

    }

    df = pd.concat([data_dict[filename]
                   for filename in group_per_gear[gear_type]])
    df = df.reset_index()
    df = df.drop('index', axis=1)
    return df


def pickle_model(model, model_name):
    file_name = 'results/{}.pkl'.format(model_name)
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)
    pass


def roc_curve(probabilities, labels):

    prob_order = np.sort(probabilities)
    TPRs = []
    FPRs = []
    for pr in prob_order:
        th = pr
        classification = probabilities >= th  # boolean
        TP = sum(np.logical_and(classification, labels))
        TPRs.append(float(TP))
        labels_neg = -1*labels + 1
        FP = sum(np.logical_and(classification, labels_neg))
        FPRs.append(float(FP))
    P = sum(labels)
    N = len(labels) - P
    return TPRs/P, FPRs/N, prob_order


def do_grid_search(est, param_grid, X_train, y_train, X_cross, y_cross):

    best_score = 0.5
    for g in ParameterGrid(param_grid):
        print('running: ', str(g))  # more verbose
        est.set_params(**g)
        est.fit(X_train, y_train)
        cross_score = est.score(X_cross, y_cross)  # mean accuracy
        train_score = est.score(X_train, y_train)  # mean accuracy
        if cross_score > best_score:
            best_cross_score = cross_score
            best_train_score = train_score
            best_params = g

    return best_train_score, best_cross_score, best_params


def get_scores(fitted_classifier, X_test, y_test):

    model = fitted_classifier
    y_predict = model.predict(X_test)
    C = confusion_matrix(y_test, y_predict)
    accurary = (C[0][0]+C[1][1]) / len(y_test)
    precision = C[1][1] / (C[0][1] + C[1][1])
    recall = C[1][1] / (C[1][1] + C[1][0])
    F1 = 2 * (precision * recall) / (precision + recall)

    return accurary, recall, F1


def is_fishy(x):
    return x['is_fishing'] == 1


def fishy(x):
    return x[is_fishy(x)]


def nonfishy(x):
    return x[~is_fishy(x)]


def _subsample_even(x0, mmsi, n):
    """Return `n` subsamples from `x0`

    - all samples have given `mmsi`

    - samples are evenly divided between fishing and nonfishing
    """
    # Create a mask that is true whenever mmsi is one of the mmsi passed in
    mask = np.zeros([len(x0)], dtype=bool)
    for m in mmsi:
        mask |= (x0['mmsi'] == m)
    x = x0[mask]


    # Pick half the values from fishy rows and half from nonfishy rows.
    f = fishy(x)
    nf = nonfishy(x)
    if n//2 > len(f) or n//2 > len(nf):
        warnings.warn("insufficient items to sample, returning fewer")
    f_index = np.random.choice(f.index, min(n//2, len(f)), replace=False)
    nf_index = np.random.choice(nf.index, min(n//2, len(nf)), replace=False)

    xf = []
    for i in f_index:
        xf.append(f.xs(i))

    xf = pd.DataFrame(xf)
    xnf = []
    for i in nf_index:
        xnf.append(nf.xs(i))
    xnf = pd.DataFrame(xnf)
    # nf = nf.xs(nf_index[0])
    ss = pd.concat([xf, xnf])
    return ss


def _subsample_proportional(x0, mmsi, n):
    mask = np.zeros([len(x0)], dtype=bool)
    for m in mmsi:
        mask |= (x0['mmsi'] == m)
    x = x0[mask]

    if n > len(x):
        warnings.warn("Warning, inufficient items to sample, returning {}".format(len(x)))
        n = len(x)
    x_index = np.random.choice(x.index, n, replace=False)
    # ss = x.xs(x_index[0])
    ss = []
    for i in x_index:
        ss.append(x.xs(i))

    ss = pd.DataFrame(ss)
    return ss

    # mask = np.zeros([len(x0)], dtype=bool)
    # for m in mmsi:
    # 	mask |= (x0['mmsi'] == m)
    # x = x0[mask]

    # # Pick half the values from fishy rows and half from nonfishy rows.
    # f = fishy(x)
    # nf = nonfishy(x)
    # if n//2 > len(f) or n//2 > len(nf):
    # 	warnings.warn("insufficient items to sample, returning fewer")
    # f_index = np.random.choice(f.index, min(n//2, len(f)), replace=False)
    # nf_index = np.random.choice(nf.index, min(n//2, len(nf)), replace=False)

    # # f = f.xs(tuple([f_index[1]]))
    # # nf = nf.xs(tuple([nf_index[1]]))

    # ss = pd.concat([f, nf])
    # return ss


def sample_by_vessel(x, size=20000, even_split=None, seed=4321):

    np.random.seed(seed)

    if size > len(x):
        print("Warning, insufficient items to sample, returning all")
        size = len(x)

    mmsi = list(set(x['mmsi']))
    if even_split is None:
        even_split = x['is_fishing'].sum() > 1 and x['is_fishing'].sum() < len(x)
    if even_split:
        base_mmsi = mmsi
        # Exclude mmsi that don't have at least one fishing or nonfishing point
        mmsi = []
        for m in base_mmsi:
            subset = x[x['mmsi'] == m]
            fishing_count = subset['is_fishing'].sum()
            if fishing_count == 0 or fishing_count == len(subset):
                continue
            mmsi.append(m)
    np.random.shuffle(mmsi)
    nx = len(x)
    sums = np.cumsum([(x['mmsi'] == m).sum() for m in mmsi])
    n1, n2 = np.searchsorted(sums, [nx//2, 3*nx//4])
    if n2 == n1:
        n2 += 1

    train_subsample = _subsample_even if even_split else _subsample_proportional

    xtrain = train_subsample(x, mmsi[:n1], size//2)
    xcross = _subsample_proportional(x, mmsi[n1:n2], size//4)
    xtest = _subsample_proportional(x, mmsi[n2:], size//4)

    return xtrain, xcross, xtest
