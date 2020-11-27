#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# import tkinter
import pandas as pd
import os
import pickle
import numpy as np
from numpy import mean
from numpy import std
# import matplotlib.pyplot as plt
# import seaborn as sns
# import math 

from sklearn.utils import safe_sqr
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# #Feature Select
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV, RFE
from boruta import BorutaPy

# #Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.linear_model import LassoCV, Lasso

from sklearn.metrics import r2_score
from warnings import filterwarnings
filterwarnings('ignore')
from argparse import ArgumentParser, RawTextHelpFormatter

def vanilla(X, y):
    elas = ElasticNet()
    name = 'Vanilla de novo'
    data = X
    age = y
    X = X.values
    y = y.values
    benchmark_elas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    result_dict = {}
    cpgs_dict = {}
    for i in benchmark_elas:
        scores = []
        cv = KFold(n_splits = 10, shuffle = True, random_state = 88)
        for train_indices, test_indices in cv.split(X):
            X_train, X_test = X[train_indices, :], X[test_indices, :]
            y_train, y_test = y[train_indices], y[test_indices]
            model = ElasticNet(alpha = i)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = r2_score(y_test, y_pred)
            scores.append(acc)
        mean = np.mean(scores)

        coef = pd.DataFrame({'coef': model.coef_, 'cpg' : data.columns})
        cur = coef[coef.coef != 0]

        result_dict[i] = mean
        cpgs_dict[i] = list(cur.iloc[:,1])
        
    max_scoring_parameter = list(result_dict.keys())[list(result_dict.values()).index(max(result_dict.values()))]
    selected_cpgs = cpgs_dict[max_scoring_parameter]
    mean_score, std = reduced_dataset_training(data, age, selected_cpgs)
    return(name, mean_score, selected_cpgs, std)

def reduced_dataset_training(X, y, selected_cpgs):
    if len(selected_cpgs) ==0:
        return(0, 0)
    else:
        X = X[selected_cpgs].values
        y = y.values

        elas = ElasticNet()
        chosen_model = elas
        chosen_score = 0
        score_list = []

        outer_cv = KFold(n_splits = 10, shuffle = True, random_state = 88)

        for train_indices, test_indices in outer_cv.split(X):
            X_train, X_test = X[train_indices, :], X[test_indices, :]
            y_train, y_test = y[train_indices], y[test_indices]

            inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 88)
            model = elas
            param_grid = {"max_iter": [100, 500, 1000],
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    "l1_ratio": np.arange(0.0, 1.0, 0.1)}

            grid = GridSearchCV(model, param_grid, scoring='r2', cv=inner_cv, refit=True)
            result = grid.fit(X_train, y_train)
            best_model = result.best_estimator_
            y_pred = best_model.predict(X_test)

            acc = r2_score(y_test, y_pred)
            score_list.append(acc)
            if acc> chosen_score:
                chosen_score = acc
                chosen_model = best_model

        return(np.mean(score_list), np.std(score_list))


def boruta(X, y):
    elas = ElasticNet()
    name = 'Boruta de novo'
    rf = RandomForestRegressor()
    if len(X.columns) > 1500:
        selector = BorutaPy(estimator = rf, n_estimators = 8, verbose=10).fit(np.array(X), np.array(y))
        selected_cpgs = list(X.columns[selector.support_])
        mean_score, std = reduced_dataset_training(X, y, selected_cpgs)        
        return(name, mean_score, selected_cpgs, std)
    
    elif len(X.columns) <= 1500:
        selector = BorutaPy(estimator = rf, n_estimators = 'auto', verbose=10).fit(np.array(X), np.array(y))
        selected_cpgs = X.columns[selector.support_]
        mean_score, std = reduced_dataset_training(X, y, selected_cpgs)
        return(name, mean_score, selected_cpgs, std)


def preselected_with_boruta(X, y, selected_cpgs, name):
    elas = ElasticNet()
    origin = name
    X = X[selected_cpgs]
    if len(X.columns) <= 0:
        return(origin, 0, [], 0)
    else:
        name, mean_score, new_cpgs, std = boruta(X,y)
        return(origin, mean_score, new_cpgs, std)


def SFM(X, y):
    elas = ElasticNet()
    name = "SFM de novo"
    thresh_list = [0.01, 0.05, 0.1, 0.5]
    
    best_score= 0
    best_cpgs = []
    best_std = 0
    
    for i in thresh_list:
        print("Completing SFM with threshold: " +str(i))
        selector = SelectFromModel(elas, threshold=i).fit(X, y)
        feature_idx = selector.get_support(indices=True)
        selected_cpgs = X.columns[feature_idx]
        
        mean_score, std = reduced_dataset_training(X, y, selected_cpgs)
        if mean_score > best_score:
            best_cpgs = selected_cpgs
            best_score = mean_score
            best_std = best_std

    return(name, best_score, best_cpgs, best_std)

def RFE100(X, y, n_features_to_select = 100):
    name = 'RFE de novo to 100'
    elas = ElasticNet()
    estimator = elas
    n_features = X.shape[1]
    n_features_to_select = n_features_to_select
    support_ = np.ones(n_features, dtype=bool)
    ranking_ = np.ones(n_features, dtype=int)
    step = 0.01

    while np.sum(support_) > n_features_to_select:
        step = 0.01
        features = np.arange(n_features)[support_]
        estimator = clone(estimator)
        print("Fitting estimator with %d features." % np.sum(support_))

        estimator.fit(X.iloc[:,features], y)   
        step = int(max(1, step * np.sum(support_)))
        print("Eliminating "+str(step)+ " features")

        importances = estimator.coef_
        if importances.ndim == 1:
            importances = safe_sqr(importances)
        else:
            importances = safe_sqr(importances).sum(axis=0)

        ranks = np.argsort(importances)
        ranks = np.ravel(ranks)
        threshold = min(step, np.sum(support_) - n_features_to_select)
        support_[features[ranks][:threshold]] = False
        ranking_[np.logical_not(support_)] += 1


    features = np.arange(n_features)[support_]
    estimator_ = clone(estimator)
    final_model = estimator_.fit(X.iloc[:,features], y)

    end_support = support_
    end_ranking = ranking_
    n_features_ = support_.sum()
    feature_name = X.columns[end_support]
    
    selected_cpgs = list(feature_name)
    mean_score, std = reduced_dataset_training(X, y, selected_cpgs)
        
    return(name, mean_score, selected_cpgs, std)  

def RFE1500(X, y, n_features_to_select = 1500):
    name = 'RFE de novo to 1500'
    elas = ElasticNet()
    estimator = elas
    n_features = X.shape[1]
    n_features_to_select = n_features_to_select
    support_ = np.ones(n_features, dtype=bool)
    ranking_ = np.ones(n_features, dtype=int)
    step = 0.01

    while np.sum(support_) > n_features_to_select:
        step = 0.01
        features = np.arange(n_features)[support_]
        estimator = clone(estimator)
        print("Fitting estimator with %d features." % np.sum(support_))

        estimator.fit(X.iloc[:,features], y)   
        step = int(max(1, step * np.sum(support_)))
        print("Eliminating "+str(step)+ " features")

        importances = estimator.coef_
        if importances.ndim == 1:
            importances = safe_sqr(importances)
        else:
            importances = safe_sqr(importances).sum(axis=0)

        ranks = np.argsort(importances)
        ranks = np.ravel(ranks)
        threshold = min(step, np.sum(support_) - n_features_to_select)
        support_[features[ranks][:threshold]] = False
        ranking_[np.logical_not(support_)] += 1


    features = np.arange(n_features)[support_]
    estimator_ = clone(estimator)
    final_model = estimator_.fit(X.iloc[:,features], y)

    end_support = support_
    end_ranking = ranking_
    n_features_ = support_.sum()
    feature_name = X.columns[end_support]
    
    selected_cpgs = list(feature_name)
    return(selected_cpgs)  
    
def training_intersected_cpgs(data, age, score_dict):
    elas = ElasticNet()
    name = 'Intersection of all selected CpGs'
    all_cpgs = []
    for score in score_dict.keys():
        all_cpgs.append(score_dict[score][1])
    all_cpgs = list(pd.Series(all_cpgs).dropna())
    intersected_cpgs = list(set([item for sublist in all_cpgs for item in sublist]))
    mean_score, std = reduced_dataset_training(data, age, intersected_cpgs)
    return(name, mean_score, intersected_cpgs, std)

def parse_args():

    parser = ArgumentParser(prog = 'python3 age_prediction.py',
        formatter_class = RawTextHelpFormatter, description =
        '  Example syntax:'
        '    python3 age_prediction.py methylation.csv metadata.csv')

    parser.add_argument('methylation', help='path to methylation matrix CSV file')

#     #Optionals
#     parser.add_argument('-o', dest='out_dir', help='path to a where output files are saved')
    return(parser.parse_args())


def main():
    elas = ElasticNet()
    args = parse_args()
    methylation_data = pd.read_csv(args.methylation)
    data = methylation_data.drop(['Age'], axis=1)
    age = methylation_data['Age']
    if (data.to_numpy().max() > 1 or methylation_data.to_numpy().min() < 0):
        print('A value larger than 1 or smaller than 0 was detected, please normalize data before uploading.')
    
#     elif methylation_data.isnull().values.any() == True:
#         print('Missing Values/NaNs detected, imputing now.')
#         imputer = KNNImputer(n_neighbors=2)
#         data = pd.DataFrame(imputer.fit_transform(data, age),
#                                           columns=data.columns)
    else:

        score_dict = {}
        
 #_____________________________________________________________________________________________________________________________________

        vanilla_name, vanilla_score, vanilla_cpgs, vanilla_std = vanilla(data,age)
        score_dict[vanilla_score] = [vanilla_name, vanilla_cpgs, vanilla_std]
        #_____________________________________________________________________________________________________________________________________

        boruta_name, boruta_mean_score, boruta_selected_cpgs, boruta_std= boruta(data,age)
        score_dict[boruta_mean_score] = [boruta_name, boruta_selected_cpgs, boruta_std]

        #_____________________________________________________________________________________________________________________________________

        rfe_1500_cpgs = RFE1500(data,age)
        rfe_1000_cpgs = RFE1500(data[rfe_1500_cpgs],age, 1000) #Use the 1500 before to go to 1000

        selector = RFECV(estimator = elas, step=1, cv = 10, scoring = 'r2').fit(data[rfe_1000_cpgs], age)
        feature_idx = selector.get_support(indices=True)
        rfe_1000_to_rfecv = list(data[rfe_1000_cpgs].columns[feature_idx])
        rfe_1000_to_rfecv_score, rfe_1000_to_rfecv_std = reduced_dataset_training(data, age, rfe_1000_to_rfecv)
        score_dict[rfe_1000_to_rfecv_score] = ['RFE de novo to 1000 followed by RFECV', rfe_1000_to_rfecv, rfe_1000_to_rfecv_std]

        rfe_w_boruta_name, rfe_w_boruta_mean_score, rfe_w_boruta_selected_cpgs, rfe_w_boruta_std = preselected_with_boruta(
          data, age, rfe_1500_cpgs, 'RFE de novo to 1500 followed by Boruta')
        score_dict[rfe_w_boruta_mean_score] = [rfe_w_boruta_name, rfe_w_boruta_selected_cpgs, rfe_w_boruta_std]

        rfe_name, rfe_mean_score, rfe_selected_cpgs, rfe_std = RFE100(data[rfe_1000_cpgs],age)
        score_dict[rfe_mean_score] = [rfe_name, rfe_selected_cpgs, rfe_std]
        #_____________________________________________________________________________________________________________________________________

        sfm_name, sfm_mean_score, sfm_selected_cpgs, sfm_std = SFM(data,age)
        score_dict[sfm_mean_score] = [sfm_name, sfm_selected_cpgs, sfm_std]

        sfm_w_boruta_name, sfm_w_boruta_mean_score, sfm_w_boruta_selected_cpgs, sfm_w_boruta_std = preselected_with_boruta(
          data,age, sfm_selected_cpgs, 'SFM de novo followed by Boruta')
        score_dict[sfm_w_boruta_mean_score] = [sfm_w_boruta_name, sfm_w_boruta_selected_cpgs, sfm_w_boruta_std]

        rfe_1000_to_sfm_name, rfe_1000_to_sfm_mean_score,  rfe_1000_to_sfm_selected_cpgs,  rfe_1000_to_sfm_std = SFM(data[rfe_1000_cpgs],age)
        score_dict[rfe_1000_to_sfm_mean_score] = ['RFE de novo to 1000 followed by SFM', list(rfe_1000_to_sfm_selected_cpgs), rfe_1000_to_sfm_std]
        #_____________________________________________________________________________________________________________________________________

        inter_name, inter_mean_score, inter_selected_cpgs, inter_std = training_intersected_cpgs(data, age, score_dict)    
        score_dict[inter_mean_score] = [inter_name, inter_selected_cpgs, inter_std]

        inter_w_boruta_name, inter_w_boruta_mean_score, inter_w_boruta_selected_cpgs, inter_w_boruta_std = preselected_with_boruta(
          data,age, inter_selected_cpgs, 'Intersection followed by Boruta')
        score_dict[inter_w_boruta_mean_score] = [inter_w_boruta_name, inter_w_boruta_selected_cpgs, inter_w_boruta_std]

        #_____________________________________________________________________________________________________________________________________

        for score in score_dict.keys():
            print(score_dict[score][0]+ " selected "+ str(len(score_dict[score][1])) 
                +" CpGs. Mean R2 Score: " + str(score) +" "+"("+str(score_dict[score][2])+")")

        max_key = max(score_dict.keys())
        score_dict[max_key][0] = score_dict[max_key][0]+" (Best Model)"
        print(str(score_dict[max_key][0])+" had the best score of "+ str(max_key)+" and selected "
        + str(len(score_dict[score][1])) +" CpGs. CpG List saved. Clock Coefficients and Intercept of best performing model saved.")

        #Building a final clock

        param_grid = {"max_iter": [100, 500, 1000],
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                "l1_ratio": np.arange(0.0, 1.0, 0.1)}

        X = data[score_dict[max_key][1]]
        y = age

        if len(X.columns) <= 0:
            print('Feature selection was not successful and no viable CpGs were selected. No viable clock built.')

        else:
            grid = GridSearchCV(estimator=elas,
                              param_grid=param_grid,
                              scoring='r2',
                              cv=10,
                              n_jobs=-1)
            grid.fit(X, y)
            best_parameters = grid.best_params_

            final_clock_model = ElasticNet(alpha = best_parameters['alpha'], l1_ratio = best_parameters['l1_ratio'], max_iter = best_parameters['max_iter'])
            X = data[score_dict[max_key][1]]
            y = age
            final_clock_model.fit(X,y)
            pickle.dump(final_clock_model, open('final_clock_model.pkl', 'wb'))
            final_intercept = final_clock_model.intercept_
            final_coef = final_clock_model.coef_
            y_pred = final_clock_model.predict(data[score_dict[max_key][1]])

            print(str(score_dict[max_key][0])+" had the best score of "+ str(max_key)+" and selected "
            + str(len(score_dict[max_key][1])) +" CpGs. CpG List saved. Clock Coefficients and Intercept of best performing model saved.")

            cur_dict = {'CpGs' : score_dict[max_key][1], 'Coefficients': final_clock_model.coef_, 'Intercept': final_clock_model.intercept_, 
                    'ElasticNet Alpha' : best_parameters['alpha'], 'ElasticNet L1 Ratio' : best_parameters['l1_ratio'], 'ElasticNet Max Iterations' : best_parameters['max_iter'] }
            cur_df = pd.DataFrame(cur_dict)
            cur_df.to_csv('best_model_cpgs_coefficients_intercept.csv',index=False)

            age_graph = pd.DataFrame({'Chronological Age' : age, 'Predicted Age' : y_pred})
            age_labels = []
            for index, row in age_graph.iterrows():
                if row[0] <=20:
                    age_labels.append('Youth (1-20)')
                elif row[0] >20 and row[0] <= 40:
                    age_labels.append('Adult (20-40)')
                elif row[0] >40 and row[0] <= 60:
                    age_labels.append('MiddleAged (40-60)')
                elif row[0] >60 and row[0] <= 80:
                    age_labels.append('Old (60-80)')
                elif row[0] >80:
                    age_labels.append('Elderly (80+)')
                else:
                    age_labels.append(float('nan'))
            age_graph['Age Labels'] = age_labels
            age_graph.to_csv('age_graph.csv',index=False)

            labelled_best_cpgs = data[score_dict[max_key][1]]
            labelled_best_cpgs['Age'] = age
            labelled_best_cpgs['Age Labels'] = age_labels
            labelled_best_cpgs.to_csv('labelled_best_cpgs.csv', index=False)

            final_cpgs_df = pd.DataFrame()
            name_list = []
            for score in score_dict.keys():
                cur_cpg_list = pd.Series(score_dict[score][1])
                name_list.append(score_dict[score][0])
                final_cpgs_df = pd.concat([final_cpgs_df, cur_cpg_list], ignore_index=True, axis=1)
            final_cpgs_df.columns = name_list 
            final_cpgs_df.columns = [col+ ' (' + str(round(score,5))+')' for col, score in zip(final_cpgs_df.columns, score_dict.keys())]

            final_cpgs_df.to_csv('final_cpg_list.csv', index=False)

            print('Finished')


if __name__ == "__main__":
    main()
