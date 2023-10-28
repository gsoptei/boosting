import pandas as pd
import numpy as np
import xgboost as xgb
from settings import *
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
import graphviz
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, auc, accuracy_score, confusion_matrix 
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import uniform, randint
from collections import namedtuple
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

# SAVE MODEL AS JSON

def plot_tree_custom(xgb_model, filename, rankdir='UT'):
    """
    Plot the tree in high resolution
    :param xgb_model: xgboost trained model
    :param filename: the pdf file where this is saved
    :param rankdir: direction of the tree: default Top-Down (UT), accepts:'LR' for left-to-right tree
    :return:
    """
    # Works for Classification with early stopping
    # gvz = xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration, rankdir=rankdir)
    # for regression, without early stopping
    gvz = xgb.to_graphviz(xgb_model, rankdir=rankdir)
    _, file_extension = os.path.splitext(filename)
    format = file_extension.strip('.').lower()
    data = gvz.pipe(format=format)
    full_filename = filename
    with open(full_filename, 'wb') as f:
        f.write(data)

def plot_final_results(model):

    matplotlib.use('agg')
    fig, ax = plt.subplots(figsize=(10, 20))

    model = model
    # Works for Classification with early stopping
    # model.fit(vals[2], vals[4], early_stopping_rounds=5, eval_set=[(vals[3], vals[5])])
    # for regression, without early stopping
    model.fit(vals[0],vals[1])
    # xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])

    sorted_features = sorted(model.feature_importances_)
    labels_ordered = [x for _, x in sorted(zip(model.feature_importances_, LABELS))]

    # xgb.plot_importance(model, ax=ax).figure.savefig('xgb_test_features.png')
    # plt.barh(list(LABELS), model.feature_importances_)
    plt.barh(labels_ordered, sorted_features)
    plt.savefig('xgb_test_features.png')
    
    print('Feature order saved as xgb_test_features.png')
    plot_tree_custom(model, 'xgb_tree_test.jpeg')
    print('Tree saved as xgb_tree_test.jpg')


def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            hyperparams = results['params'][candidate]
            return hyperparams

def convert_to_dataframe(r_data):
    
    # get the first item
    first_item = next(iter(r_data))
    main_dataframe = r_data[first_item]

    return main_dataframe

def perform_subsetting(get_subset=True, func=convert_to_dataframe):
    
    # drop unwanted columns
    main_dataframe = func(DATA_FILE).drop(COLUMNS_TO_DROP, axis=1)
    # print(main_dataframe.columns)
    X = main_dataframe.drop('SPROD', axis=1).to_numpy()
    y = main_dataframe.SPROD.to_numpy()
    
    # split data into separate training and test set
    if get_subset:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    return X, y, X_train, X_test, y_train, y_test

def search_hyperparams(params=PARAMS):

    global vals
    vals = perform_subsetting()
    X, y = vals[0], vals[1]

    xgb_model = xgb.XGBRegressor()
    
    # change number of iterations!!!!
    search = RandomizedSearchCV(xgb_model, param_distributions=params,random_state=42, n_iter=10, cv=3, 
    verbose=1, n_jobs=1, return_train_score=True
    )

    search.fit(X, y)

    hyperparams = report_best_scores(search.cv_results_, 1)
    return hyperparams, vals
    

def create_model(func=search_hyperparams):

    params = func()[0]
    model = xgb.XGBRegressor(**params)
    plot_final_results(model)
    model.save_model('xgboost_test.json')
    print("Model saved as 'xgboost_test.json'")
    return model

final_model = create_model()


