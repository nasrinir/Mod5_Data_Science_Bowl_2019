import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,TimeSeriesSplit
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, f1_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

def XGBOOST_param_selection(X_train, y_train, nfolds, n_jobs = None):
    '''
    Perform gridsearchCV to fit XGboost models 
    and to find the optimum hyperparameters
    
    Parameters
    ----------
        
        X_train     (pandas DataFrame) train dataframe
        
        y_train     (pandas DataFrame) train labels
        
        nfolds       (int)     number of folds for cross validation
                                        
        n_jobs       (int or None) Number of jobs to run in parallel. 
                      optional (default=None)
                      None means 1 unless in a joblib.parallel_backend context.
                      -1 means using all processors. See Glossary for more details.
        
    Returns
    -------
        best_model     trained model 
        
    '''
#     # Number of trees in random forest
#     n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 11)]
#     # Number of features to consider at every split
#     max_features = ['auto', 'sqrt']
#     # Maximum number of levels in tree
#     max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#     max_depth.append(None)
#     # Minimum number of samples required to split a node
#     min_samples_split = [2, 5, 10]
#     # Minimum number of samples required at each leaf node
#     min_samples_leaf = [1, 2, 4]
#     # Method of selecting samples for training each tree
#     bootstrap = [True, False]
#     # Create the random grid
    
    params = {
        'colsample_bytree': 0.8,                 
        'learning_rate': 0.08,
        'max_depth': 10,
        'subsample': 1,
        'objective':'multi:softprob',
        'num_class':4,
        'eval_metric':'mlogloss',
        'min_child_weight':3,
        'gamma':0.25,
        'n_estimators':500
    }


    skf = StratifiedKFold(n_splits=nfolds, 
                          shuffle=True, 
                          random_state=1985)
    
    grid_search = RandomizedSearchCV(estimator = xgb(),
                                     param_distributions = params,
                                     cv = skf, 
                                     #scoring = 'roc_auc',
                                     n_jobs = n_jobs).fit(X_train, y_train)
    print('The training roc_auc_score is:', round(grid_search.best_score_, 3))
    print('The best parameters are:', grid_search.best_params_)
    best_model = grid_search.best_estimator_
    return best_model



def random_forest_param_selection(X_train, y_train, nfolds, n_jobs = None):
    '''
    Perform gridsearchCV to fit random forest models 
    and to find the optimum hyperparameters
    
    Parameters
    ----------
        
        X_train     (pandas DataFrame) train dataframe
        
        y_train     (pandas DataFrame) train labels
        
        nfolds       (int)     number of folds for cross validation
                                        
        n_jobs       (int or None) Number of jobs to run in parallel. 
                      optional (default=None)
                      None means 1 unless in a joblib.parallel_backend context.
                      -1 means using all processors. See Glossary for more details.
        
    Returns
    -------
        best_model     trained model 
        
    '''
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 11)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    params = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}

    skf = StratifiedKFold(n_splits=nfolds, 
                          shuffle=True, 
                          random_state=1985)
    
    grid_search = RandomizedSearchCV(estimator = RandomForestClassifier(),
                                     param_distributions = params,
                                     cv = skf, 
                                     scoring = 'roc_auc',
                                     n_jobs = n_jobs).fit(X_train, y_train)
    print('The training roc_auc_score is:', round(grid_search.best_score_, 3))
    print('The best parameters are:', grid_search.best_params_)
    best_model = grid_search.best_estimator_
    return best_model

def Convert_LabelEncoder(X_train, X_test):
    '''
    encode categorical features using 
    a one-hot or ordinal encoding scheme.
    
    Parameters
    ----------
        
        X_train     (pandas DataFrame) train dataframe
        
        X_test      (pandas DataFrame)  test dataframe
        
        
    Returns
    -------
        X_train     (pandas DataFrame) train dataframe with one-hot encoding
        X_test      (pandas DataFrame) test dataframe with one-hot encoding
        
    '''
    #concat X_train and X_test
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(X_train[col].astype(str).values) + list(X_test[col].astype(str).values))
            X_train[col] = le.transform(list(X_train[col].astype(str).values))
            X_test[col] = le.transform(list(X_test[col].astype(str).values))
    return X_train, X_test