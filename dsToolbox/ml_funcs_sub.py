import datetime as dt
import inspect
import math
import os
import re
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, jarque_bera, anderson, normaltest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf

from sklearn import metrics, tree
from sklearn.base import is_classifier
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet, ElasticNetCV, Lars, Lasso, LassoCV, LassoLars, LassoLarsIC,
    LinearRegression, LogisticRegression, Ridge, SGDRegressor
)
from sklearn.model_selection import cross_validate, StratifiedKFold, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC, NuSVC, SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from yellowbrick.features import Rank2D
from yellowbrick.regressor import ResidualsPlot, PredictionError, CooksDistance

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

warnings.filterwarnings('ignore')

# Constants
DEFAULT_RANDOM_STATE = 10
DEFAULT_IMPUTER_STRATEGY = 'mean'
DEFAULT_PCA_COMPONENTS = 5
DEFAULT_MAX_ITER = 1000
DEFAULT_CV_VERBOSE = 10

####------------------------------model performances----------------------------------------------------------------------------------------

METRIC_DICT = {
    # Classification metrics
    'accuracy': metrics.accuracy_score,
    'balanced_accuracy': metrics.balanced_accuracy_score,
    'top_k_accuracy': metrics.top_k_accuracy_score,
    'average_precision': metrics.average_precision_score,
    'aucpr': metrics.average_precision_score,
    'brier_score': metrics.brier_score_loss,
    'f1': metrics.f1_score,
    'f1_samples': metrics.f1_score,
    'log_loss': metrics.log_loss,
    'precision': metrics.precision_score,
    'recall': metrics.recall_score,
    'jaccard': metrics.jaccard_score,
    'auc': metrics.roc_auc_score,
    'roc_auc': metrics.roc_auc_score,
    'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score,
    'adjusted_rand_score': metrics.adjusted_rand_score,
    'completeness_score': metrics.completeness_score,
    'fowlkes_mallows_score': metrics.fowlkes_mallows_score,
    'homogeneity_score': metrics.homogeneity_score,
    'mutual_info_score': metrics.mutual_info_score,
    'normalized_mutual_info_score': metrics.normalized_mutual_info_score,
    'rand_score': metrics.rand_score,
    'v_measure_score': metrics.v_measure_score,
    
    # Regression metrics
    'explained_variance': metrics.explained_variance_score,
    'max_error': metrics.max_error,
    'mean_absolute_error': metrics.mean_absolute_error,
    'mean_squared_error': metrics.mean_squared_error,
    'mean_squared_log_error': metrics.mean_squared_log_error,
    'median_absolute_error': metrics.median_absolute_error,
    'R2': metrics.r2_score,
    'mean_poisson_deviance': metrics.mean_poisson_deviance,
    'mean_gamma_deviance': metrics.mean_gamma_deviance,
    'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error,

    # Additional metrics
    'mcc': metrics.matthews_corrcoef,
    'kappa': metrics.cohen_kappa_score,
}

#scoring_dict=dict(zip(scoring,[metrics.make_scorer(metric_dict.get(i)) for i in scoring]))

# from sklearn.metrics import matthews_corrcoef
# from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import f1_score
# scoring_dict=dict(zip(scoring, scoring))
# ## https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
# scoring_dict['mcc']=make_scorer(matthews_corrcoef)
# scoring_dict['kappa']=make_scorer(cohen_kappa_score)

def classifiers_template(
    y: pd.Series, 
    random_state: int = DEFAULT_RANDOM_STATE, 
    imputer_strategy: str = DEFAULT_IMPUTER_STRATEGY,
    pc_n_components: int = DEFAULT_PCA_COMPONENTS
) -> Dict[str, Pipeline]:
    """
    Create a template of classification models with preprocessing pipelines.
    
    Args:
        y: Target variable for calculating class weights
        random_state: Random state for reproducibility
        imputer_strategy: Strategy for imputing missing values
        pc_n_components: Number of principal components for PCA
        
    Returns:
        Dictionary of classifier pipelines
    """

    classifiers = {
        "Nearest_Neighbors_2": KNeighborsClassifier(2),
        "Nearest_Neighbors_3": KNeighborsClassifier(3),
        "Nearest_Neighbors_4": KNeighborsClassifier(4),
        "Nearest_Neighbors_5": KNeighborsClassifier(5),

        "Decision_Tree_depth5": DecisionTreeClassifier(max_depth=5, random_state=random_state),
        "Decision_Tree_depth10": DecisionTreeClassifier(max_depth=10, random_state=random_state),

        "Naive_Bayes": GaussianNB(),

        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),

        'logReg_mode_l1': LogisticRegression(
            penalty='l1', solver='liblinear', max_iter=DEFAULT_MAX_ITER, random_state=random_state
        ),
        'logReg_mode_l2': LogisticRegression(max_iter=DEFAULT_MAX_ITER, random_state=random_state),
        'logReg_model_pca': LogisticRegression(max_iter=DEFAULT_MAX_ITER, random_state=random_state),

        'RandomForest_model1': RandomForestClassifier(random_state=random_state),
        'RandomForest_model_balanced': RandomForestClassifier(
            class_weight='balanced', random_state=random_state
        ),
        'RandomForest_model_n200': RandomForestClassifier(n_estimators=200, random_state=random_state),
        'RandomForest_model_n300': RandomForestClassifier(n_estimators=300, random_state=random_state),

        "AdaBoost": AdaBoostClassifier(random_state=random_state),
    }
    
    # Add XGBoost classifiers if available
    if XGBClassifier is not None:
        xgb_classifiers = {
            'Xgboost_n200': XGBClassifier(n_estimators=200, random_state=random_state),
            'Xgboost_n200_dp10': XGBClassifier(
                n_estimators=200, max_depth=10, random_state=random_state
            ),
            'Xgboost_Weighted': XGBClassifier(
                scale_pos_weight=float(np.sum(y == 0)) / np.sum(y == 1), random_state=random_state
            ),
            'Xgboost_Weighted_n200': XGBClassifier(
                n_estimators=200, 
                scale_pos_weight=float(np.sum(y == 0)) / np.sum(y == 1), 
                random_state=random_state
            ),
            'Xgboost_Weighted_n200_dp10': XGBClassifier(
                n_estimators=200, 
                max_depth=10, 
                scale_pos_weight=float(np.sum(y == 0)) / np.sum(y == 1), 
                random_state=random_state
            ),
        }
        classifiers.update(xgb_classifiers)
    
    # Add LightGBM classifiers if available
    if LGBMClassifier is not None:
        lgbm_classifiers = {
            'LightGBM': LGBMClassifier(random_state=random_state),
            'LightGBM_n200': LGBMClassifier(n_estimators=200, random_state=random_state),
            'LightGBM_n400': LGBMClassifier(n_estimators=300, random_state=random_state),
            'LightGBM_n1000': LGBMClassifier(n_estimators=1000, random_state=random_state),
            'LightGBM_n200_dp10': LGBMClassifier(
                max_depth=10, n_estimators=200, random_state=random_state
            ),
            'LightGBM_n300_dp10': LGBMClassifier(
                max_depth=10, n_estimators=300, random_state=random_state
            ),
        }
        classifiers.update(lgbm_classifiers)
    
    # Add MLP classifiers
    mlp_classifiers = {
        'MLPClassifier1': MLPClassifier(alpha=1, max_iter=DEFAULT_MAX_ITER, random_state=random_state),
        'MLPClassifier_early_stopping': MLPClassifier(
            alpha=1, max_iter=DEFAULT_MAX_ITER, early_stopping=True, random_state=random_state
        ),
        'MLPClassifier3': MLPClassifier(
            alpha=1, max_iter=DEFAULT_MAX_ITER, solver='sgd', early_stopping=True, random_state=random_state
        ),
    }
    classifiers.update(mlp_classifiers)

    # Create pipelines for each classifier
    classifiers_with_pipelines = {}
    for name, classifier in classifiers.items():
        if any(x in name for x in ['pca', 'DiscriminantAnalysis']):
            classifiers_with_pipelines[name] = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy=imputer_strategy)),
                ("scaler", StandardScaler()),
                ("reduce_dims", PCA(n_components=pc_n_components)),
                (name, classifier)
            ])
        elif ('xgb' not in name.lower()) and ('gbm' not in name.lower()):
            classifiers_with_pipelines[name] = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy=imputer_strategy)),
                ("scaler", StandardScaler()),
                (name, classifier)
            ])
        else:
            classifiers_with_pipelines[name] = classifier
    
    return classifiers_with_pipelines

def regressors_template(random_state: int = DEFAULT_RANDOM_STATE) -> Dict[str, Pipeline]:
    """
    Create a template of regression models with preprocessing pipelines.
    
    Args:
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary of regressor pipelines
    """
    regressors = {
        "Nearest_Neighbors_2": KNeighborsRegressor(2),
        "Nearest_Neighbors_3": KNeighborsRegressor(3),
        "Nearest_Neighbors_4": KNeighborsRegressor(4),
        "Nearest_Neighbors_5": KNeighborsRegressor(5),

        "Decision_Tree_depth5": DecisionTreeRegressor(max_depth=5, random_state=random_state),
        "Decision_Tree_depth10": DecisionTreeRegressor(max_depth=10, random_state=random_state),

        "Naive_Bayes": GaussianNB(),

        'LinearRegression': LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "SGDRegressor": SGDRegressor(random_state=random_state),
        "ElasticNet": ElasticNet(random_state=random_state),
        "ElasticNetCV": ElasticNetCV(random_state=random_state),
        "Lars": Lars(random_state=random_state),
        "Lasso": Lasso(),
        "LassoCV": LassoCV(random_state=random_state),
        "LassoLars": LassoLars(random_state=random_state),
        "LassoLarsIC": LassoLarsIC(),
        
        'RandomForest_model1': RandomForestRegressor(random_state=random_state),
        'RandomForest_model_n200': RandomForestRegressor(n_estimators=200, random_state=random_state),
        'RandomForest_model_n300': RandomForestRegressor(n_estimators=300, random_state=random_state),

        "AdaBoost": AdaBoostRegressor(random_state=random_state),
    }
    
    # Add XGBoost regressors if available
    if XGBRegressor is not None:
        xgb_regressors = {
            'Xgboost_n200': XGBRegressor(n_estimators=200, random_state=random_state),
            'Xgboost_n200_dp10': XGBRegressor(
                n_estimators=200, max_depth=10, random_state=random_state
            ),
        }
        regressors.update(xgb_regressors)
    
    # Add LightGBM regressors if available
    if LGBMRegressor is not None:
        lgbm_regressors = {
            'LightGBM': LGBMRegressor(random_state=random_state),
            'LightGBM_n200': LGBMRegressor(n_estimators=200, random_state=random_state),
            'LightGBM_n400': LGBMRegressor(n_estimators=300, random_state=random_state),
            'LightGBM_n1000': LGBMRegressor(n_estimators=1000, random_state=random_state),
            'LightGBM_n200_dp10': LGBMRegressor(
                max_depth=10, n_estimators=200, random_state=random_state
            ),
            'LightGBM_n300_dp10': LGBMRegressor(
                max_depth=10, n_estimators=300, random_state=random_state
            ),
        }
        regressors.update(lgbm_regressors)
    
    # Add MLP regressors
    mlp_regressors = {
        'MLPRegressor1': MLPRegressor(alpha=1, max_iter=DEFAULT_MAX_ITER, random_state=random_state),
        'MLPRegressor_early_stopping': MLPRegressor(
            alpha=1, max_iter=DEFAULT_MAX_ITER, early_stopping=True, random_state=random_state
        ),
        'MLPRegressor3': MLPRegressor(
            alpha=1, max_iter=DEFAULT_MAX_ITER, solver='sgd', early_stopping=True, random_state=random_state
        ),
    }
    regressors.update(mlp_regressors)

    # Create pipelines for each regressor
    regressors_with_pipelines = {}
    for name, regressor in regressors.items():
        if any(x in name for x in ['pca', 'DiscriminantAnalysis']):
            regressors_with_pipelines[name] = Pipeline(steps=[
                ("scaler", RobustScaler()),
                ("imputer", SimpleImputer(strategy="median")),
                ("reduce_dims", PCA(n_components=20)),
                (name, regressor)
            ])
        elif ('xgb' not in name.lower()) and ('gbm' not in name.lower()):
            regressors_with_pipelines[name] = Pipeline(steps=[
                ("scaler", RobustScaler()),
                ("imputer", SimpleImputer(strategy="median")),
                (name, regressor)
            ])
        else:
            regressors_with_pipelines[name] = regressor
    
    return regressors_with_pipelines

def ml_scores(
    y_model: pd.DataFrame, 
    scores_names: List[str],
    multi_class: str = 'ovo',
    average: str = 'macro'  # {'micro', 'macro', 'samples', 'weighted'}
) -> pd.DataFrame:
    """
    Calculate various machine learning evaluation scores for a given model.
    
    Args:
        y_model: DataFrame containing the true labels and predicted labels. 
                 It must have columns 'y_true' and 'y_pred'. If 'CV_Iteration' 
                 column is not present, it will be added with a default value 'All_data'.
        scores_names: List of score names to be calculated. The score names should be keys 
                     in the `METRIC_DICT` which maps to the corresponding scoring functions.
        multi_class: Strategy for multi-class classification metrics
        average: Averaging strategy for metrics
        
    Returns:
        DataFrame containing the calculated scores for each cross-validation iteration, 
        along with the mean and standard deviation of the scores across iterations, 
        and the overall scores for the entire dataset.
    """
    if 'CV_Iteration' not in y_model.columns:
        y_model['CV_Iteration'] = 'All_data'
         
    scores_all = pd.Series(index=scores_names, dtype='float64', name='scores_all')
    scores = pd.DataFrame(index=y_model['CV_Iteration'].unique(), columns=scores_names)
    
    if (y_model.shape[1] > 5) and (average == 'binary'):
        print("It is a multiclass problem: average argument changed to 'macro'")
        average = 'macro'

    kwargs_dict = {'average': average, 'multi_class': multi_class}
        
    for con, score_name in enumerate(scores_names):
        try:
            umetric = METRIC_DICT.get(score_name)
            if umetric is None:
                print(f"Score '{score_name}' not found in METRIC_DICT")
                continue

            umetric_args = list(inspect.signature(umetric).parameters)
            kwargs_dict2 = {k: kwargs_dict[k] for k in kwargs_dict if k in umetric_args}

            scores_all.iloc[con] = umetric(y_model['y_true'], y_model['y_pred'], **kwargs_dict2)
            scores.loc[:, score_name] = y_model.groupby('CV_Iteration').apply(
                lambda x: pd.Series({
                    score_name: umetric(x['y_true'], x['y_pred'], **kwargs_dict2)
                })
            )
        except Exception as e:
            print(f"{score_name} wasn't added to scores data frame:\n {str(e)}")
            
    scores = pd.concat([
        scores,
        pd.DataFrame(scores.mean(axis=0)).T.rename({0: 'CV_scores_Mean'}, axis=0),
        pd.DataFrame(scores.std(axis=0)).T.rename({0: 'CV_scores_STD'}, axis=0),
        pd.DataFrame(scores_all).T
    ], axis=0)
    scores = scores.reset_index().rename({'index': 'CV'}, axis=1)

    return scores

def ml_scores_crossvalidate(**kwargs):
    """
    Perform cross-validation on a given estimator and return the results as a DataFrame.
    This function uses scikit-learn's `cross_validate` to perform cross-validation on the provided estimator
    and returns the results in a pandas DataFrame. The DataFrame includes the mean and standard deviation
    of the cross-validation scores.
    Parameters:
    **kwargs: 
      Keyword arguments to be passed to `sklearn.model_selection.cross_validate`. These typically include:
      - estimator: The object to use to fit the data.
      - X: The data to fit.
      - y: The target variable to try to predict.
      - scoring: A single string or a callable to evaluate the predictions on the test set.
      - cv: Determines the cross-validation splitting strategy.
      - return_train_score: Whether to include train scores.
    Returns:
    pandas.DataFrame:
      A DataFrame containing the cross-validation results. The DataFrame includes the mean and standard
      deviation of the cross-validation scores, with the keys 'CV_scores_Mean' and 'CV_scores_STD' respectively.
      The 'fit_time' and 'score_time' columns are removed from the results.
    """
    from sklearn.model_selection import cross_validate
    ##NOTE: you can't use cross_validate for early stopping
    ####scoring for cross_validate
    # scoring=[
    #         'accuracy',
    #         'roc_auc',
    #         'recall' ,
    #         'f1',
    #         'kappa',
    #         'mcc',
    #         'average_precision',
    #         'balanced_accuracy',
    #         'precision',
    #         ]
    # scoring_dict=dict(zip(scoring, scoring))
    # ## https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    # scoring_dict['mcc']=make_scorer(matthews_corrcoef)
    # scoring_dict['kappa']=make_scorer(cohen_kappa_score)

    # scoring_dict=dict(zip(scoring,[eval("make_scorer("+metric_dict.get(i)+")") for i in scoring])) 
    
    cv_results = cross_validate(**kwargs)
    cv_results=pd.DataFrame(cv_results)
    keys_to_remove = ['fit_time', 'score_time']
    for key in keys_to_remove:
      del cv_results[key]
    cv_results=cv_results.append(cv_results.mean(axis=0).rename("CV_scores_Mean"))
    cv_results=cv_results.append(cv_results.std(axis=0).rename("CV_scores_STD"))
    # print(cv_results)
    return cv_results  

def ml_prediction_sub_epochs(model):
  results = model.evals_result()
  df_epochs=pd.DataFrame()
  for metric_key in results['validation_0'].keys():
      val0=results['validation_0'][metric_key]
      val1=results['validation_1'][metric_key]
      tmp=pd.DataFrame([val0,val1],index=[f'Train_{metric_key}',f'Validation_{metric_key}']).T
      df_epochs=pd.concat([df_epochs, tmp], axis=1)
      
  df_epochs.index.name='epochs' 
  df_epochs=df_epochs.reset_index()
  df_epochs['best_ntree']=model.best_iteration

  return df_epochs

def ml_prediction(ml_model, X, y, sk_fold, X_test=None, y_test=None, callbacks=None, verbose=False):
  """
  Perform machine learning prediction with cross-validation and optional early stopping.
  Parameters:
  -----------
  ml_model : object
    The machine learning model to be used for prediction. Can be a single model or a pipeline.
  X : pandas.DataFrame
    The feature matrix for training.
  y : pandas.Series
    The target vector for training.
  sk_fold : object or list
    Cross-validation splitting strategy. Can be an instance of StratifiedKFold, TimeSeriesSplit, or a list containing validation data.
  X_test : pandas.DataFrame, optional
    The feature matrix for testing. Default is None.
  y_test : pandas.Series, optional
    The target vector for testing. Default is None.
  callbacks : list, optional
    List of callback functions for early stopping. Default is None.
  verbose : bool, optional
    If True, print progress messages. Default is False.
  Returns:
  --------
  y_model : pandas.DataFrame
    DataFrame containing predictions and true values for each cross-validation iteration.
  ml_models : list
    List of fitted machine learning models for each cross-validation iteration.
  df_epochs : pandas.DataFrame or None
    DataFrame containing epoch information for models with early stopping, or None if early stopping is not used.
  """
    y_model = pd.DataFrame([])   
    df_epochs = pd.DataFrame([])  
    ml_models = []

  umodel= ml_model[-1] if isinstance(ml_model, Pipeline) else ml_model
  model_name= umodel.__class__.__name__ 
  
  ##TODO:include all models with early_stopping
  
  early_stopping_rounds= umodel.early_stopping_rounds if 'xgb' in model_name.lower() else None
  if (X_test is not None) & ((isinstance(sk_fold, StratifiedKFold))|(isinstance(sk_fold, TimeSeriesSplit))) & (early_stopping_rounds is  None):
    print("Warning! Awkward senario. cross validaiton is on and predicition on a seprate test data set!")

  if sk_fold is None:
    print("Warning! training and validation data sets are the same")
    cv=zip([range(X.shape[0])], [range(X.shape[0])])

  elif isinstance(sk_fold, list):
    print("no cross validation ")
    X_val=sk_fold[0]
    y_val=sk_fold[1]
    train_no=X.shape[0]
    X=pd.concat([X, X_val], axis=0)
    y=pd.concat([y, y_val], axis=0)
    cv=zip([range(train_no)],[range(train_no, X.shape[0])])

  else:
    cv= sk_fold.split(X,y) 

  for cv_itr,(train_index, val_index) in enumerate(cv):
    if verbose:
      print(f"CV Itreation {cv_itr+1}") #{len(cv)}
    X_train, X_val = X.iloc[train_index,:], X.iloc[val_index,:] 
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    ##TODO: it is only for xgboost, cover other ml_models
    if early_stopping_rounds is not None:
      ###NOTE: use [(X_train, y_train), (X_val, y_val)] instead of [(X_val, y_val)] to save epochs. [(X_val, y_val)] is much faster.
      eval_set = [(X_train, y_train), (X_val, y_val)]

      ml_model.fit(X_train,
                  y_train,
                  eval_set              = eval_set,
                  callbacks             = callbacks,
                  verbose               = 10
                  )
      df_epochs_tmp           = ml_prediction_sub_epochs(ml_model)
      df_epochs_tmp['CV_Iteration'] = cv_itr
      df_epochs               = pd.concat([df_epochs,df_epochs_tmp],axis = 0)

      print("best_ntree=", ml_model.best_iteration, ", best_score=", ml_model.best_score)

    else:
      ml_model.fit(X_train,
                  y_train,
                  )

    ml_models.append(ml_model)

    if X_test is not None: X_val=X_test
    if y_test is not None: y_val=y_test
    
    if is_classifier(umodel):
      y_model0 = pd.DataFrame(ml_model.predict_proba(X_val), index=y_val.index)
      y_model0=pd.concat([y_model0, y_model0.idxmax(axis=1).rename('y_pred'), y_val.rename('y_true')], axis=1)
    else:
      y_model0 = pd.DataFrame(ml_model.predict(X_val), index=y_val.index)
      y_model0=pd.concat([y_model0,  y_val], axis=1)
      y_model0.columns=['y_pred','y_true']
      
    # print(y_model0)
    y_model0['CV_Iteration']= cv_itr
    y_model=pd.concat([y_model, y_model0],axis=0)
    
  # print('                    ')

  if early_stopping_rounds is not None:
    df_epochs['best_ntree']=df_epochs['best_ntree']==df_epochs['epochs']
  else:
    df_epochs=None

  return y_model, ml_models, df_epochs

def ml_comparison(ml_models,
                  X,
                  y,
                  scores_names,
                  sk_fold,
                  mapNames={},
                  multi_class='ovo',
                  average ='macro', 
                  plot=True,
                  verbose=True
                  ):
  """
  Compare multiple machine learning models using cross-validation and return their performance metrics.
  Parameters:
  -----------
  ml_models : list
    List of machine learning models to be compared.
  X : array-like or DataFrame
    Feature matrix.
  y : array-like or Series
    Target vector.
  scores_names : list
    List of scoring metrics to evaluate the models.
  sk_fold : int or cross-validation generator
    Number of folds or cross-validation generator.
  mapNames : dict, optional
    Dictionary mapping model indices to custom names. Default is an empty dictionary.
  plot : bool, optional
    If True, plot the comparison results. Default is True.
  verbose : bool, optional
    If True, print detailed information during the process. Default is True.
  Returns:
  --------
  metrics_all : DataFrame
    DataFrame containing the performance metrics of all models.
  """
  with warnings.catch_warnings():
    if verbose:
      warnings.simplefilter("default")
    else:
      warnings.simplefilter("ignore")  
    metrics_all=pd.DataFrame()
    for con, model in enumerate(ml_models):
      if con in mapNames.keys():
        model_name=mapNames.get(con)    
      elif isinstance(model, Pipeline):
        model_name='-->'.join([i.__class__.__name__ for i in model])
      else:
        model_name=model.__class__.__name__
      
      start_time=dt.datetime.now()
      print(model_name+'...')

      y_model,  _ , _ = ml_prediction(model,
                                        X,
                                        y,
                                        sk_fold,
                                        )

      cv_results= ml_scores(y_model, scores_names,
                            multi_class=multi_class,
                            average=average
                            )

      tmp=cv_results
      tmp.insert(0, "model", model_name)
      end_time = dt.datetime.now()
      run_time=end_time-start_time
      tmp['elapsed_time']= run_time

      #tmp.index=pd.MultiIndex.from_product([[model_name],tmp.index])
      metrics_all=pd.concat([metrics_all, tmp] ,axis=0)

      # print(tmp)
      # print('run_time:', run_time)
      if verbose:
        txt=metrics_all.loc[metrics_all['CV'].isin(["CV_scores_Mean",
                                                  "CV_scores_STD",
                                                  # "scores_all"
                                                  ]), :]
                                                                  # .sort_values(by=['CV','recall'],
                                                                    # ascending=[True, False]
                                                                    # ).set_index(['CV','model'])
        print('models summary:\n',txt)
        print("-------------------------------------------")
    #metrics_all=metrics_all.reset_index().rename({'level_0':'model','level_1':'CV'},axis=1)
    #    idx = pd.IndexSlice
    #    metrics_all_summary=metrics_all.loc[idx[:, ["CV_scores_Mean", "CV_scores_STD"]], :]
    if plot:
      ml_comparison_plot(# The above code is not doing anything. It is just a comment.
      metrics_all, outputFile=None)
    return metrics_all

def classifer_performance_batch(y_model,
                        map_lbls={0:'Low Loss', 1:'High Loss'},
                        scores_names=[
                                      'accuracy',
                                      # 'balanced_accuracy',
                                      'recall' ,
                                      'precision',
                                      # 'roc_auc',  
                                      # 'aucpr',
                                      ],
                      multi_class='raise',
                      average ='balanced',                       
                        ):
  """
  Evaluate the performance of a classifier model using various metrics.
  Parameters:
  y_model (dict): A dictionary containing the model's predictions and probabilities.
          Expected keys are 'y_true' for true labels and 'prob' for predicted probabilities.
  map_lbls (dict, optional): A dictionary mapping class labels to their descriptions. Default is {0: 'Low Loss', 1: 'High Loss'}.
  scores_names (list, optional): A list of score names to evaluate. Default is ['accuracy', 'recall', 'precision'].
  Returns:
  tuple: A tuple containing:
    - scores (dict): A dictionary of evaluated scores based on the provided score names.
    - confMats (dict): A dictionary containing confusion matrices for each class.
  """
  confMats=plot_confusion_matrix2(y_model, map_lbls, outputFile=None)

  model_prob=y_model['prob'] #y_model[map_lbls.get(1)]
  pos_label=1
  df_rp, thresholds= precision_recall_curve2(y_model['y_true'],
                                                      model_prob,
                                                      pos_label,
                                                      outputFile=None
                                                      )

  df_auc, thresholds= roc_curve2(y_model['y_true'],
                                          model_prob,
                                          pos_label,
                                          outputFile=None
                                        )

  # out, df_gain_chart, df_lift_chart = gainNlift(y_model['y_true'],
  #                                               model_prob,
  #                                               pos_label,
  #                                               outputFile=None,
  #                                               groupNo = 25
  #                                               )
  
  scores= ml_scores(y_model, scores_names,
                    multi_class=multi_class,
                    average=average)
  return scores, confMats

def ml_prediction_xValNest(ml_model, X, y, outter_fold, inner_fold):
  """
  Perform nested cross-validation for a given machine learning model.
  Parameters:
  ml_model : object
    The machine learning model to be used for training and prediction.
  X : pandas.DataFrame
    The input features for the model.
  y : pandas.Series
    The target variable.
  outter_fold : object
    The outer cross-validation fold (e.g., KFold or StratifiedKFold).
  inner_fold : object
    The inner cross-validation fold (e.g., KFold or StratifiedKFold).
  Returns:
  y_model : pandas.DataFrame
    DataFrame containing the predicted probabilities, predicted class, 
    true class, and cross-validation iteration for each test sample.
  df_epochs : pandas.DataFrame
    DataFrame containing the epochs information for each cross-validation iteration.
  """
  y_model=pd.DataFrame([])   
  df_epochs=pd.DataFrame([])   

  for cv_outter, (trainVal_index, tst_index) in enumerate(outter_fold.split(X, y)):
    X_trainVal, X_tst = X.iloc[trainVal_index,:], X.iloc[tst_index,:] 
    y_trainVal, y_tst = y.iloc[trainVal_index],   y.iloc[tst_index]

    for cv_itr, (train_index, val_index) in enumerate(inner_fold.split(X_trainVal, y_trainVal)):
      #print("Itreation ",cv_itr)

      X_train, X_val = X_trainVal.iloc[train_index,:], X_trainVal.iloc[val_index,:] 
      y_train, y_val = y_trainVal.iloc[train_index],   y_trainVal.iloc[val_index]

      ##TODO: it is only for xgboost, cover other ml_models
      eval_set = [(X_train, y_train), (X_val, y_val)]

      ml_model.fit(X_train,
                  y_train,
                  eval_set=eval_set,
                  verbose=200
                  )
      df_epochs_tmp=ml_prediction_sub_epochs(ml_model)
      df_epochs_tmp['CV_Iteration']=f'{cv_outter}_{cv_itr}'
      df_epochs=pd.concat([df_epochs,df_epochs_tmp],axis=0)

      print("best_ntree=", ml_model.best_iteration)
      print("best_score=", ml_model.best_score)

      ml_model2=XGBClassifier(n_estimators=ml_model.best_iteration)
      
      ml_model2.fit(X_trainVal,
                  y_trainVal,
                  )
            
      y_model0 = pd.DataFrame(ml_model2.predict_proba(X_tst),index=X_tst.index)
    #   y_model0.rename(columns=map_lbls,inplace=True)
      y_model0 = pd.concat([y_model0,y_model0.idxmax(axis=1).rename('y_pred')],axis=1)
    
      y_model0['CV_Iteration']=f'{cv_outter}_{cv_itr}'
      y_model0['y_true']=y_tst
      
      y_model = pd.concat([y_model,y_model0],axis=0)

  #print("--------------------------------------------------------")

  df_epochs['best_ntree']=df_epochs['best_ntree']==df_epochs['epochs']
    
  return y_model, df_epochs


###TODO: test it before using:
class LinearRegressionAssumptionsChecker:
    """
    A comprehensive class to check assumptions of Linear Regression Models
    including OLS, Ridge, and Lasso based on Analytics Vidhya methodology:
    1. Linear relationship between X and Y
    2. Independence of residuals (No Autocorrelation)
    3. Multicollinearity (adapted for regularized models)
    4. Homoscedasticity (Equal Variance)
    5. Normality of residuals
    6. No influential outliers
    
    Note: For Ridge/Lasso, multicollinearity checks are less critical
    as these models handle correlated features differently.

    Usage:
    # Step 0: Fit your model
    model = LinearRegression()
    model.fit(X, y)

    # Step 1: Initialize the checker
    checker = LinearRegressionAssumptionsChecker(
        model=model, 
        X=X, 
        y=y, 
        feature_names=feature_names
    )

    # Step 2: Run full diagnostic
    results = checker.run_full_diagnostic()

    # Step 3: Individual checks (optional)
    checker.check_linearity()
    checker.check_independence()
    checker.check_multicollinearity()
    checker.check_homoscedasticity()
    checker.check_normality()
    checker.check_outliers_influence()
    """
    
    def __init__(self, model, X, y, feature_names=None, model_type='ols'):
        """
        Initialize the checker with fitted model and data
        
        Parameters:
        -----------
        model : fitted regression model (sklearn or statsmodels)
        X : array-like, feature matrix
        y : array-like, target variable
        feature_names : list, names of features (optional)
        model_type : str, type of model ('ols', 'ridge', 'lasso')
        """
        self.model = model
        self.X = np.array(X)
        self.y = np.array(y)
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(self.X.shape[1])]
        self.model_type = model_type.lower()
        
        # Get predictions and residuals
        self.y_pred = model.predict(self.X)
        self.residuals = self.y - self.y_pred
        
        # Standardized residuals
        self.std_residuals = self.residuals / np.std(self.residuals)
        
        # For statsmodels compatibility
        if hasattr(model, 'resid'):
            self.residuals = model.resid
            self.y_pred = model.fittedvalues
    
    def check_linearity(self, plot=True):
        """
        Check Assumption 1: Linear relationship between X and Y
        Using residuals vs fitted values plot
        """
        print("="*60)
        print("ASSUMPTION 1: LINEAR RELATIONSHIP")
        print("="*60)
        
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs Fitted Plot
            axes[0].scatter(self.y_pred, self.residuals, alpha=0.7, color='blue')
            axes[0].axhline(y=0, color='red', linestyle='--')
            axes[0].set_xlabel('Fitted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Fitted Values\n(Check for Linearity)')
            
            # Add smooth line
            z = np.polyfit(self.y_pred, self.residuals, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(self.y_pred.min(), self.y_pred.max(), 100)
            axes[0].plot(x_smooth, p(x_smooth), color='red', linewidth=2)
            
            # Actual vs Predicted Plot
            axes[1].scatter(self.y, self.y_pred, alpha=0.7, color='green')
            min_val = min(self.y.min(), self.y_pred.min())
            max_val = max(self.y.max(), self.y_pred.max())
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            axes[1].set_xlabel('Actual Values')
            axes[1].set_ylabel('Predicted Values')
            axes[1].set_title('Actual vs Predicted Values')
            
            plt.tight_layout()
            plt.show()
        
        # Analysis
        residual_pattern = np.polyfit(self.y_pred, self.residuals, 2)[0]
        
        print(f"📊 Analysis:")
        print(f"   • Residual pattern coefficient: {residual_pattern:.6f}")
        
        if abs(residual_pattern) < 0.001:
            print("   ✅ PASSED: Linear relationship assumption is satisfied")
            print("   • Residuals are randomly scattered around zero")
        else:
            print("   ❌ VIOLATED: Non-linear relationship detected")
            print("   • Consider polynomial features or non-linear transformations")
            print("   • Suggested: log(X), √X, or X² transformations")
        
        return abs(residual_pattern) < 0.001
    
    def check_independence(self, plot=True):
        """
        Check Assumption 2: Independence of residuals (No Autocorrelation)
        Using Durbin-Watson test
        """
        print("\n" + "="*60)
        print("ASSUMPTION 2: INDEPENDENCE OF RESIDUALS")
        print("="*60)
        
        # Durbin-Watson Test
        dw_stat = durbin_watson(self.residuals)
        
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs Index (Time Series Plot)
            axes[0].plot(self.residuals, marker='o', linestyle='-', alpha=0.7)
            axes[0].axhline(y=0, color='red', linestyle='--')
            axes[0].set_xlabel('Observation Index')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Index\n(Check for Autocorrelation)')
            
            # ACF Plot (simplified)
            from statsmodels.tsa.stattools import acf
            autocorr = acf(self.residuals, nlags=20, fft=True)
            axes[1].bar(range(len(autocorr)), autocorr, alpha=0.7)
            axes[1].axhline(y=0, color='red', linestyle='--')
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('Autocorrelation')
            axes[1].set_title('Autocorrelation Function')
            
            plt.tight_layout()
            plt.show()
        
        print(f"📊 Durbin-Watson Test:")
        print(f"   • DW Statistic: {dw_stat:.4f}")
        print(f"   • Range: 0 to 4 (2 = no autocorrelation)")
        
        if 1.5 <= dw_stat <= 2.5:
            print("   ✅ PASSED: No significant autocorrelation detected")
        elif dw_stat < 1.5:
            print("   ❌ VIOLATED: Positive autocorrelation detected")
            print("   • Solution: Add lag variables or use time series models")
        else:
            print("   ❌ VIOLATED: Negative autocorrelation detected")
            print("   • Solution: Check model specification")
        
        return 1.5 <= dw_stat <= 2.5
    
    def check_multicollinearity(self, plot=True):
        """
        Check Assumption 3: No Multicollinearity
        Using VIF (Variance Inflation Factor) and correlation matrix
        
        Note: For Ridge/Lasso models, multicollinearity is less problematic:
        - Ridge: Handles multicollinearity by shrinking coefficients
        - Lasso: Performs automatic feature selection
        """
        print("\n" + "="*60)
        print("ASSUMPTION 3: NO MULTICOLLINEARITY")
        if self.model_type in ['ridge', 'lasso']:
            print(f"📝 Note: {self.model_type.upper()} regression handles multicollinearity")
        print("="*60)
        
        # Calculate VIF
        X_with_const = sm.add_constant(self.X)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = ['const'] + self.feature_names
        vif_data["VIF"] = [variance_inflation_factor(X_with_const, i) 
                          for i in range(X_with_const.shape[1])]
        
        # Correlation Matrix
        corr_matrix = np.corrcoef(self.X.T)
        
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # VIF Plot
            vif_features = vif_data[vif_data['Feature'] != 'const']
            axes[0].barh(vif_features['Feature'], vif_features['VIF'], color='skyblue')
            axes[0].axvline(x=4, color='orange', linestyle='--', label='Moderate (VIF=4)')
            axes[0].axvline(x=10, color='red', linestyle='--', label='High (VIF=10)')
            axes[0].set_xlabel('VIF Value')
            axes[0].set_title('Variance Inflation Factor')
            axes[0].legend()
            
            # Correlation Heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, xticklabels=self.feature_names, 
                       yticklabels=self.feature_names, ax=axes[1])
            axes[1].set_title('Feature Correlation Matrix')
            
            plt.tight_layout()
            plt.show()
        
        print(f"📊 VIF Analysis:")
        for _, row in vif_data.iterrows():
            if row['Feature'] != 'const':
                vif_val = row['VIF']
                if vif_val <= 4:
                    status = "✅ Good"
                elif vif_val <= 10:
                    status = "⚠️  Moderate"
                else:
                    status = "❌ High"
                print(f"   • {row['Feature']}: {vif_val:.3f} {status}")
        
        # Check for high correlations
        high_corr_pairs = []
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                if abs(corr_matrix[i, j]) > 0.8:
                    high_corr_pairs.append((self.feature_names[i], 
                                          self.feature_names[j], 
                                          corr_matrix[i, j]))
        
        max_vif = vif_data[vif_data['Feature'] != 'const']['VIF'].max()
        
        if self.model_type == 'ols':
            # Standard OLS interpretation
            if max_vif <= 4 and len(high_corr_pairs) == 0:
                print("   ✅ PASSED: No multicollinearity issues detected")
            else:
                print("   ❌ VIOLATED: Multicollinearity detected")
                if high_corr_pairs:
                    print("   • High correlations found:")
                    for feat1, feat2, corr in high_corr_pairs:
                        print(f"     - {feat1} & {feat2}: {corr:.3f}")
                print("   • Solutions: Remove highly correlated features, PCA, Ridge regression")
            return max_vif <= 4 and len(high_corr_pairs) == 0
        
        else:
            # Ridge/Lasso interpretation
            print(f"   ℹ️  INFO: {self.model_type.upper()} regression analysis:")
            if self.model_type == 'ridge':
                print("   • Ridge handles multicollinearity by shrinking coefficients")
                print("   • High VIF/correlations are less problematic")
                if hasattr(self.model, 'alpha_'):
                    print(f"   • Regularization strength (α): {self.model.alpha_:.4f}")
            elif self.model_type == 'lasso':
                print("   • Lasso performs automatic feature selection")
                print("   • Correlated features may be automatically removed")
                if hasattr(self.model, 'alpha_'):
                    print(f"   • Regularization strength (α): {self.model.alpha_:.4f}")
                
                # Show which features have non-zero coefficients
                if hasattr(self.model, 'coef_'):
                    non_zero_features = np.sum(np.abs(self.model.coef_) > 1e-10)
                    print(f"   • Features selected: {non_zero_features}/{len(self.feature_names)}")
            
            if max_vif > 10 or len(high_corr_pairs) > 0:
                print("   ⚠️  WARNING: High multicollinearity detected")
                print("   • Monitor model stability and coefficient interpretation")
            else:
                print("   ✅ ACCEPTABLE: Multicollinearity within reasonable bounds")
            
            # For regularized models, we're more lenient
            return True
    
    def check_homoscedasticity(self, plot=True):
        """
        Check Assumption 4: Homoscedasticity (Equal Variance)
        Using Breusch-Pagan test and residual plots
        """
        print("\n" + "="*60)
        print("ASSUMPTION 4: HOMOSCEDASTICITY (EQUAL VARIANCE)")
        print("="*60)
        
        # Breusch-Pagan Test
        bp_test = het_breuschpagan(self.residuals, self.X)
        bp_stat, bp_p_value = bp_test[0], bp_test[1]
        
        # White Test
        white_test = het_white(self.residuals, self.X)
        white_stat, white_p_value = white_test[0], white_test[1]
        
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs Fitted (Homoscedasticity check)
            axes[0].scatter(self.y_pred, self.residuals, alpha=0.7, color='purple')
            axes[0].axhline(y=0, color='red', linestyle='--')
            axes[0].set_xlabel('Fitted Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs Fitted\n(Check for Homoscedasticity)')
            
            # Scale-Location Plot (Square root of standardized residuals)
            sqrt_std_resid = np.sqrt(np.abs(self.std_residuals))
            axes[1].scatter(self.y_pred, sqrt_std_resid, alpha=0.7, color='orange')
            axes[1].set_xlabel('Fitted Values')
            axes[1].set_ylabel('√|Standardized Residuals|')
            axes[1].set_title('Scale-Location Plot')
            
            # Add smooth line
            z = np.polyfit(self.y_pred, sqrt_std_resid, 1)
            p = np.poly1d(z)
            x_smooth = np.linspace(self.y_pred.min(), self.y_pred.max(), 100)
            axes[1].plot(x_smooth, p(x_smooth), color='red', linewidth=2)
            
            plt.tight_layout()
            plt.show()
        
        print(f"📊 Statistical Tests:")
        print(f"   • Breusch-Pagan Test:")
        print(f"     - Statistic: {bp_stat:.4f}")
        print(f"     - P-value: {bp_p_value:.6f}")
        print(f"   • White Test:")
        print(f"     - Statistic: {white_stat:.4f}")
        print(f"     - P-value: {white_p_value:.6f}")
        
        alpha = 0.05
        if bp_p_value > alpha and white_p_value > alpha:
            print("   ✅ PASSED: Homoscedasticity assumption satisfied")
            print("   • Equal variance across all fitted values")
        else:
            print("   ❌ VIOLATED: Heteroscedasticity detected")
            print("   • Solutions: log(Y), √Y transformation, or weighted least squares")
        
        return bp_p_value > alpha and white_p_value > alpha
    
    def check_normality(self, plot=True):
        """
        Check Assumption 5: Normality of residuals
        Using Q-Q plot and statistical tests
        """
        print("\n" + "="*60)
        print("ASSUMPTION 5: NORMALITY OF RESIDUALS")
        print("="*60)
        
        # Statistical Tests
        shapiro_stat, shapiro_p = shapiro(self.residuals)
        jb_stat, jb_p = jarque_bera(self.residuals)
        anderson_stat = anderson(self.residuals, dist='norm')
        
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Q-Q Plot
            stats.probplot(self.residuals, dist="norm", plot=axes[0,0])
            axes[0,0].set_title('Q-Q Plot\n(Check for Normality)')
            axes[0,0].grid(True)
            
            # Histogram of residuals
            axes[0,1].hist(self.residuals, bins=30, density=True, alpha=0.7, color='lightblue')
            mu, sigma = np.mean(self.residuals), np.std(self.residuals)
            x = np.linspace(self.residuals.min(), self.residuals.max(), 100)
            axes[0,1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal')
            axes[0,1].set_xlabel('Residuals')
            axes[0,1].set_ylabel('Density')
            axes[0,1].set_title('Histogram of Residuals')
            axes[0,1].legend()
            
            # Box plot
            axes[1,0].boxplot(self.residuals, vert=True)
            axes[1,0].set_ylabel('Residuals')
            axes[1,0].set_title('Box Plot of Residuals')
            
            # Residuals vs Normal Quantiles
            from yellowbrick.regressor import ResidualsPlot
            # Since we can't directly use yellowbrick here, create a simple scatter
            axes[1,1].scatter(range(len(self.residuals)), sorted(self.residuals), alpha=0.7)
            axes[1,1].set_xlabel('Index')
            axes[1,1].set_ylabel('Sorted Residuals')
            axes[1,1].set_title('Sorted Residuals Plot')
            
            plt.tight_layout()
            plt.show()
        
        print(f"📊 Normality Tests:")
        print(f"   • Shapiro-Wilk Test:")
        print(f"     - Statistic: {shapiro_stat:.4f}")
        print(f"     - P-value: {shapiro_p:.6f}")
        print(f"   • Jarque-Bera Test:")
        print(f"     - Statistic: {jb_stat:.4f}")
        print(f"     - P-value: {jb_p:.6f}")
        print(f"   • Anderson-Darling Test:")
        print(f"     - Statistic: {anderson_stat.statistic:.4f}")
        
        alpha = 0.05
        normality_tests = [shapiro_p > alpha, jb_p > alpha]
        
        if all(normality_tests):
            print("   ✅ PASSED: Residuals are normally distributed")
        else:
            print("   ❌ VIOLATED: Residuals are not normally distributed")
            print("   • Solutions: Transform Y or X variables, remove outliers")
        
        return all(normality_tests)
    
    def check_outliers_influence(self, plot=True):
        """
        Check Assumption 6: No influential outliers
        Using Cook's distance and leverage analysis
        """
        print("\n" + "="*60)
        print("ASSUMPTION 6: NO INFLUENTIAL OUTLIERS")
        print("="*60)
        
        # Calculate Cook's distance (simplified calculation)
        n = len(self.residuals)
        p = self.X.shape[1] + 1  # number of parameters
        
        # Leverage values (hat matrix diagonal)
        X_design = sm.add_constant(self.X)
        hat_matrix = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
        leverage = np.diag(hat_matrix)
        
        # Cook's distance approximation
        cooks_d = (self.std_residuals**2 / p) * (leverage / (1 - leverage)**2)
        
        # Thresholds
        cook_threshold = 4 / n
        leverage_threshold = 2 * p / n
        
        if plot:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Cook's Distance Plot
            axes[0,0].bar(range(n), cooks_d, alpha=0.7, color='red')
            axes[0,0].axhline(y=cook_threshold, color='black', linestyle='--', 
                             label=f'Threshold: {cook_threshold:.4f}')
            axes[0,0].set_xlabel('Observation Index')
            axes[0,0].set_ylabel("Cook's Distance")
            axes[0,0].set_title("Cook's Distance Plot")
            axes[0,0].legend()
            
            # Residuals vs Leverage
            axes[0,1].scatter(leverage, self.std_residuals, alpha=0.7, color='blue')
            axes[0,1].axhline(y=0, color='red', linestyle='--')
            axes[0,1].axhline(y=2, color='orange', linestyle='--', label='|Std Residual| = 2')
            axes[0,1].axhline(y=-2, color='orange', linestyle='--')
            axes[0,1].axvline(x=leverage_threshold, color='green', linestyle='--', 
                             label=f'Leverage threshold: {leverage_threshold:.4f}')
            axes[0,1].set_xlabel('Leverage')
            axes[0,1].set_ylabel('Standardized Residuals')
            axes[0,1].set_title('Residuals vs Leverage')
            axes[0,1].legend()
            
            # Standardized Residuals Plot
            axes[1,0].scatter(range(n), self.std_residuals, alpha=0.7, color='green')
            axes[1,0].axhline(y=0, color='red', linestyle='--')
            axes[1,0].axhline(y=2, color='orange', linestyle='--', alpha=0.7)
            axes[1,0].axhline(y=-2, color='orange', linestyle='--', alpha=0.7)
            axes[1,0].set_xlabel('Observation Index')
            axes[1,0].set_ylabel('Standardized Residuals')
            axes[1,0].set_title('Standardized Residuals vs Index')
            
            # Influence Plot (Cook's D vs Leverage)
            axes[1,1].scatter(leverage, cooks_d, alpha=0.7, color='purple')
            axes[1,1].axhline(y=cook_threshold, color='red', linestyle='--')
            axes[1,1].axvline(x=leverage_threshold, color='green', linestyle='--')
            axes[1,1].set_xlabel('Leverage')
            axes[1,1].set_ylabel("Cook's Distance")
            axes[1,1].set_title('Influence Plot')
            
            plt.tight_layout()
            plt.show()
        
        # Identify influential points
        high_cook = np.where(cooks_d > cook_threshold)[0]
        high_leverage = np.where(leverage > leverage_threshold)[0]
        outlier_residuals = np.where(np.abs(self.std_residuals) > 2)[0]
        
        print(f"📊 Outlier Analysis:")
        print(f"   • Cook's Distance threshold: {cook_threshold:.4f}")
        print(f"   • Leverage threshold: {leverage_threshold:.4f}")
        print(f"   • High Cook's Distance points: {len(high_cook)} observations")
        print(f"   • High Leverage points: {len(high_leverage)} observations")
        print(f"   • Outlier residuals (|z| > 2): {len(outlier_residuals)} observations")
        
        if len(high_cook) > 0:
            print(f"   • Influential observations (high Cook's D): {high_cook[:10]}...")  # Show first 10
        
        total_outliers = len(set(list(high_cook) + list(high_leverage) + list(outlier_residuals)))
        outlier_percentage = (total_outliers / n) * 100
        
        if outlier_percentage < 5:
            print("   ✅ PASSED: No significant influential outliers detected")
        else:
            print("   ❌ ATTENTION: Potential influential outliers detected")
            print(f"   • {outlier_percentage:.1f}% of observations may be influential")
            print("   • Solutions: Investigate outliers, robust regression, or remove if justified")
        
        return outlier_percentage < 5
    
    def run_full_diagnostic(self):
        """
        Run all assumption checks and provide summary report
        """
        print("🔍 LINEAR REGRESSION ASSUMPTIONS DIAGNOSTIC REPORT")
        print(f"🎯 Model Type: {self.model_type.upper()}")
        print("📖 Based on Analytics Vidhya methodology")
        print("📅 Reference: https://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/")
        print("\n")
        
        # Run all checks
        results = {
            'Linearity': self.check_linearity(),
            'Independence': self.check_independence(), 
            'No Multicollinearity': self.check_multicollinearity(),
            'Homoscedasticity': self.check_homoscedasticity(),
            'Normality': self.check_normality(),
            'No Influential Outliers': self.check_outliers_influence()
        }
        
        # Summary Report
        print("\n" + "="*60)
        print("📋 SUMMARY REPORT")
        print("="*60)
        
        passed = sum(results.values())
        total = len(results)
        
        print(f"✅ Assumptions Satisfied: {passed}/{total}")
        print(f"📊 Model Reliability Score: {(passed/total)*100:.1f}%")
        print("\n📝 Detailed Results:")
        
        for assumption, passed in results.items():
            status = "✅ PASSED" if passed else "❌ VIOLATED"
            print(f"   • {assumption}: {status}")
        
        # Model-specific interpretation
        if self.model_type in ['ridge', 'lasso']:
            print(f"\n🎯 {self.model_type.upper()} Regression Notes:")
            if self.model_type == 'ridge':
                print("   • Multicollinearity is handled by coefficient shrinkage")
                print("   • More robust to outliers than OLS")
            elif self.model_type == 'lasso':
                print("   • Automatic feature selection reduces multicollinearity")
                print("   • Some features may have zero coefficients")
        
        if passed == total:
            print(f"\n🎉 EXCELLENT! All assumptions satisfied.")
            print(f"   Your {self.model_type} regression model is statistically sound.")
        elif passed >= total * 0.75:
            print(f"\n👍 GOOD! Most assumptions satisfied.")
            print(f"   Minor violations may not significantly impact model performance.")
        elif passed >= total * 0.5:
            print(f"\n⚠️  MODERATE! Several assumptions violated.")
            print(f"   Consider model improvements or alternative approaches.")
        else:
            print(f"\n❌ POOR! Major assumption violations detected.")
            if self.model_type == 'ols':
                print(f"   Consider: Ridge/Lasso regression, non-linear models, or data transformations.")
            else:
                print(f"   Consider: non-linear models, data transformations, or ensemble methods.")
        
        return results
