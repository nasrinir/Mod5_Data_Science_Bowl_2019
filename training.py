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
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import cohen_kappa_score, make_scorer, confusion_matrix

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
    # A parameter grid for XGBoost
    pars = {
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
    xgb_model = xgb.train(pars,
                      xgb_train,
                      num_boost_round=1000,
                      evals=[(xgb_train, 'train'), (xgb_eval, 'val')],
                      verbose_eval=False,
                      early_stopping_rounds=20
                     )


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
                                     scoring = make_scorer(cohen_kappa_score),
                                     n_jobs = n_jobs).fit(X_train, y_train)
    print('The training roc_auc_score is:', round(grid_search.best_score_, 3))
    print('The best parameters are:', grid_search.best_params_)
    best_model = grid_search.best_estimator_
    return best_model

def Convert_LabelEncoder(X_train, X_test, encoding_col_names):
    '''
    encode categorical features using 
    a one-hot or ordinal encoding scheme.
    
    Parameters
    ----------
        
        X_train     (pandas DataFrame) train dataframe
        
        X_test      (pandas DataFrame)  test dataframe
        
        encoding_col_names   (List) the names of columns needed to be encoded
        
        
    Returns
    -------
        X_train     (pandas DataFrame) train dataframe with one-hot encoding
        X_test      (pandas DataFrame) test dataframe with one-hot encoding
        
    '''
    X_train['train'] = 1
    X_test['train'] = 0
    final=pd.concat([X_train,X_test])
    for col in encoding_col_names:
        lb=LabelEncoder()
        lb.fit(final[col].astype(str).values)
        final[col]=lb.transform(final[col].astype(str).values)
    final_train = final.loc[final['train'] == 1]
    final_train.drop('train', axis = 1, inplace = True)
    final_test = final.loc[final['train'] == 0]
    final_test.drop('train', axis = 1, inplace = True)
    return final_train, final_test

# this function is the quadratic weighted kappa (the metric used for the competition submission)
def qwk(act,pred,n=4,hist_range=(0,3)):
    
    # Calculate the percent each class was tagged each label
    O = confusion_matrix(act,pred)
    # normalize to sum 1
    O = np.divide(O,np.sum(O))
    
    # create a new matrix of zeroes that match the size of the confusion matrix
    # this matriz looks as a weight matrix that give more weight to the corrects
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # makes a weird matrix that is bigger in the corners top-right and botton-left (= 1)
            W[i][j] = ((i-j)**2)/((n-1)**2)
            
    # make two histograms of the categories real X prediction
    act_hist = np.histogram(act,bins=n,range=hist_range)[0]
    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]
    
    # multiply the two histograms using outer product
    E = np.outer(act_hist,prd_hist)
    E = np.divide(E,np.sum(E)) # normalize to sum 1
    
    # apply the weights to the confusion matrix
    num = np.sum(np.multiply(W,O))
    # apply the weights to the histograms
    den = np.sum(np.multiply(W,E))
    
    return 1-np.divide(num,den)