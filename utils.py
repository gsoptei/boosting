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

# TODO #3: CALCULATE AND SAVE RESIDUALS

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

def plot_tree_custom(xgb_model, filename, rankdir='UT'):
    """
    Plot the tree in high resolution
    :param xgb_model: xgboost trained model
    :param filename: the pdf file where this is saved
    :param rankdir: direction of the tree: default Top-Down (UT), accepts:'LR' for left-to-right tree
    :return:
    """
    # for regression, without early stopping
    gvz = xgb.to_graphviz(xgb_model, rankdir=rankdir)
    _, file_extension = os.path.splitext(filename)
    format = file_extension.strip('.').lower()
    data = gvz.pipe(format=format)
    full_filename = filename
    with open(full_filename, 'wb') as f:
        f.write(data)

def plot_final_results(models):

    X, y = vals[0], vals[1]
        
    for k, model in enumerate(models):
        print(f"Plotting {model}")
        matplotlib.use('agg')
        fig, ax = plt.subplots(figsize=(10, 20))

        model_to_plot = xgb.XGBRegressor()
        model_to_plot.load_model(FILE_PATH + model)
        model_to_plot.fit(X, y)

        sorted_features = sorted(model_to_plot.feature_importances_)
        labels_ordered = [x for _, x in sorted(zip(model_to_plot.feature_importances_, LABELS))]

        # custom plotting for feature importance
        plt.barh(labels_ordered, sorted_features)
        plt.savefig('xgb_features_model_%d.png' %((k) + 1))

        # plot feature importance using the method from xgb
        xgb.plot_importance(model_to_plot, ax=ax, max_num_features=15).figure.savefig('xgb_feature_importance_%d.png' %((k) + 1))


        print('Feature importance saved as xgb_features_model_%d.png' %((k) + 1))
        # plot trees
        plot_tree_custom(model_to_plot, 'xgb_tree_model_%d.jpeg' %((k) + 1))
        print('Tree saved as xgb_tree_model_%d.jpg' %((k) + 1))



def report_best_scores(results, n_top=3):
    param_list = []
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
            param_list.append(hyperparams)
    return param_list


def convert_to_dataframe(r_data):
    
    # get the first item
    first_item = next(iter(r_data))
    main_dataframe = r_data[first_item]

    return main_dataframe

def perform_subsetting(get_subset=True, func=convert_to_dataframe):
    
    # drop unwanted columns
    main_dataframe = func(DATA_FILE).drop(COLUMNS_TO_DROP, axis=1)
    X = main_dataframe.drop('SPROD', axis=1).to_numpy()
    y = main_dataframe.SPROD.to_numpy()
    
    # split data into separate training and test set
    if get_subset:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    return X, y, X_train, X_test, y_train, y_test, main_dataframe


def search_hyperparams(params=PARAMS):

    global vals
    vals = perform_subsetting()
    X, y = vals[0], vals[1]
    hyperparams = []

    xgb_model = xgb.XGBRegressor()
    
    # perform cross validation and boosting
    for i in ITERATIONS:
        for j in CV:
            search = RandomizedSearchCV(xgb_model, param_distributions=params,random_state=42, n_iter=i, cv=j, 
            verbose=1, n_jobs=1, return_train_score=True
            )
            search.fit(X, y)
            # hyperparams = report_best_scores(search.cv_results_, 1)
            hyperparams.append(report_best_scores(search.cv_results_, 1)) 

    print(f"Number of parameters: {len(hyperparams)}, parameters: {hyperparams}")
    return hyperparams, vals

def create_model(func=search_hyperparams):

    params, vals = func()
    X, y = vals[0], vals[1]
    print(f"The number of hyperparameters is: {len(params)}")
    errors = {}
    
    for k, param in enumerate(params):
        print(f"Model will be created with the following hyperparameters: {param[0]}")
        model = xgb.XGBRegressor(**param[0])
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        model.save_model('xgboost_test_%d.json' %((k) + 1)) 
        MODELS.append('xgboost_test_%d.json' %((k) + 1))
        print("Model saved as 'xgboost_test_%d.json'" %((k) + 1))
        print("")
        print(np.sqrt(mse))

        # add model rmse to a dictionary
        errors['xgboost_test_%d.json' %((k) + 1)] = (np.sqrt(mse))

        print(f"MODEL CREATION FINISHED WITH PARAMETERS: {param}")
        print("")
    
    print(f"These are the models: {MODELS}")
    return MODELS, errors

def predict_sprod(models):

    global vals
    X, X_train, X_test, y_train, y_test, final_dataset = vals[0], vals[2], vals[3], vals[4], vals[5], vals[6]

    for model in models:
        model_predict = xgb.XGBRegressor()
        model_predict.load_model(FILE_PATH + model)
        model_predict.fit(X_train, y_train, 
                          eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True
        )
        y_pred = model_predict.predict(X)

        print(f'This is the prediction of SPROD: {y_pred} for the model named {model}')
        print(f'SPROD PREDICTION FINISHED FOR {model}')
        PREDICTIONS.append(y_pred)
        # vals[6]["SPROD_pred_%d" %(models.index(model) + 1)] = y_pred

        # append SPROD prediciton to dataframe
        final_dataset["SPROD_pred_%d" %(models.index(model) + 1)] = y_pred

    # vals[6].to_excel('output.xlsx', index=False)
    # save final dataframe to excel
    final_dataset.to_excel('output.xlsx', index=False)
    print(len(PREDICTIONS))
    return PREDICTIONS
