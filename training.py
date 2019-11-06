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
from scipy import stats
from time import time
import datetime
import catboost

def cat_boosting(X, y, X_test, all_features):
    # oof is an zeroed array of the same size of the input dataset
    oof = np.zeros(len(X))
    NFOLDS = 5
    # here the KFold class is used to split the dataset in 5 diferents training and validation sets
    # this technique is used to assure that the model isn't overfitting and can performs aswell in
    # unseen data. More the number of splits/folds, less the test will be impacted by randomness
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)
    training_start_time = time()
    models = []
    for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
        # each iteration of folds.split returns an array of indexes of the new training data and validation data
        start_time = time()
        print(f'Training on fold {fold+1}')
        # creates the model
        clf = make_classifier()
        # fits the model using .loc at the full dataset to select the splits indexes and features used
        clf.fit(X.loc[trn_idx, all_features], y.loc[trn_idx], eval_set=(X.loc[test_idx, all_features], y.loc[test_idx]),
                use_best_model=True, verbose=500)

        # then, the predictions of each split is inserted into the oof array
        oof[test_idx] = clf.predict(X.loc[test_idx, all_features]).reshape(len(test_idx))
        models.append(clf)
        print('Fold {} finished in {}'.format(
            fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
        print('____________________________________________________________________________________________\n')
        # break

        print('-' * 30)
        # and here, the complete oof is tested against the real data using que metric (quadratic weighted kappa)
        print('OOF QWK:', qwk(y, oof))
        print('-' * 30)
    # make predictions on test set once
    predictions = []
    for model in models:
        predictions.append(model.predict(X_test[all_features]))
    predictions = np.concatenate(predictions, axis=1)
    predictions = stats.mode(predictions, axis=1)[0].reshape(-1)
    return models, predictions


def make_classifier(iterations=6000):
    clf = catboost.CatBoostClassifier(
                               loss_function='MultiClass',
                                eval_metric="WKappa",
                               task_type="CPU",
                               #learning_rate=0.01,
                               iterations=iterations,
                               od_type="Iter",
                                #depth=4,
                               early_stopping_rounds=500,
                                #l2_leaf_reg=10,
                                #border_count=96,
                               random_seed=42,
                                #use_best_model=use_best_model
                              )
        
    return clf
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

def model_XGBOOST(X_train,y_train,final_test,n_splits=3):
    scores=[]
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
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pre=np.zeros((len(final_test),4),dtype=float)
    final_test=xgb.DMatrix(final_test)


    for train_index, val_index in kf.split(X_train, y_train):
        train_X = X_train.iloc[train_index]
        val_X = X_train.iloc[val_index]
        train_y = y_train[train_index]
        val_y = y_train[val_index]
        xgb_train = xgb.DMatrix(train_X, train_y)
        xgb_eval = xgb.DMatrix(val_X, val_y)

        xgb_model = xgb.train(pars,
                      xgb_train,
                      num_boost_round=1000,
                      evals=[(xgb_train, 'train'), (xgb_eval, 'val')],
                      verbose_eval=False,
                      early_stopping_rounds=20
                     )

        val_X=xgb.DMatrix(val_X)
        pred_val=[np.argmax(x) for x in xgb_model.predict(val_X)]
        score=cohen_kappa_score(pred_val,val_y,weights='quadratic')
        scores.append(score)
        print('choen_kappa_score :',score)

        pred=xgb_model.predict(final_test)
        y_pre+=pred

    pred = np.asarray([np.argmax(line) for line in y_pre])
    print('Mean score:',np.mean(scores))
    
    return xgb_model,pred



def random_forest_param_selection(X_train, y_train,X_test, nfolds, n_jobs = None):
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
    pred = best_model.predict(X_test)
    return best_model, pred

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

def Convert_categorical_variables(X_train, X_test, cols_to_transform):
    '''
    Convert categorical variable into dummy/indicator variables.
    
    Parameters
    ----------
        
        X_train     (pandas DataFrame) train dataframe
        
        X_test      (pandas DataFrame)  test dataframe
        
        cols_to_transform
        
        
    Returns
    -------
        X_train     (pandas DataFrame) train dataframe with one-hot encoding
        X_test      (pandas DataFrame) test dataframe with one-hot encoding
        
    '''
    #concat X_train and X_test
    X_train['train'] = 1
    X_test['train'] = 0
    df = pd.concat([X_train, X_test], axis = 0)
    df.columns = X_train.columns

    #cols_to_transform = [col for col in df.columns if X_train[col].dtype == object]

    #Getting dummies variables
    df_dummies = pd.get_dummies( df, columns = cols_to_transform, drop_first=True )
    
    #Return to train and test dataframes and remove train columns 
    X_train = df_dummies.loc[df['train'] == 1]
    X_train.drop(columns = 'train', axis = 1, inplace = True)
    X_test = df_dummies.loc[df['train'] == 0]
    X_test.drop(columns = 'train', axis = 1, inplace = True)
    
    return X_train, X_test