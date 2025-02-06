import pandas as pd
import numpy as np
import math 

import os ,sys
import re
####------------------------------model performances----------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
from sklearn import metrics

metric_dict={
            'accuracy'                          : metrics.accuracy_score,
            'balanced_accuracy'                 : metrics.balanced_accuracy_score,
            'top_k_accuracy'                    : metrics.top_k_accuracy_score,
            'average_precision'                 : metrics.average_precision_score,
            'aucpr'                             : metrics.average_precision_score,
            'brier_score'                       : metrics.brier_score_loss,
            'f1'                                : metrics.f1_score,
            'f1_samples'                        : metrics.f1_score,
            'log_loss'                          : metrics.log_loss,
            'precision'                         : metrics.precision_score,
            'recall'                            : metrics.recall_score,
            'jaccard'                           : metrics.jaccard_score,
            'auc'                               : metrics.roc_auc_score,
            'roc_auc'                           : metrics.roc_auc_score,
            'adjusted_mutual_info_score'        : metrics.adjusted_mutual_info_score,
            'adjusted_rand_score'               : metrics.adjusted_rand_score,
            'completeness_score'                : metrics.completeness_score,
            'fowlkes_mallows_score'             : metrics.fowlkes_mallows_score,
            'homogeneity_score'                 : metrics.homogeneity_score,
            'mutual_info_score'                 : metrics.mutual_info_score,
            'normalized_mutual_info_score'      : metrics.normalized_mutual_info_score,
            'rand_score'                        : metrics.rand_score,
            'v_measure_score'                   : metrics.v_measure_score,
            
            'explained_variance'                : metrics.explained_variance_score,
            'max_error'                         : metrics.max_error,
            'mean_absolute_error'               : metrics.mean_absolute_error,
            'mean_squared_error'                : metrics.mean_squared_error,
            'mean_squared_log_error'            : metrics.mean_squared_log_error,
            'median_absolute_error'             : metrics.median_absolute_error,
            'R2'                                : metrics.r2_score,
            'mean_poisson_deviance'             : metrics.mean_poisson_deviance,
            'mean_gamma_deviance'               : metrics.mean_gamma_deviance,
            'mean_absolute_percentage_error'    : metrics.mean_absolute_percentage_error,

            'mcc'                               : metrics.matthews_corrcoef,
            'kappa'                             : metrics.cohen_kappa_score,

            }

#scoring_dict=dict(zip(scoring,[metrics.make_scorer(metric_dict.get(i)) for i in scoring]))

# from sklearn.metrics import matthews_corrcoef
# from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import f1_score
# scoring_dict=dict(zip(scoring, scoring))
# ## https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
# scoring_dict['mcc']=make_scorer(matthews_corrcoef)
# scoring_dict['kappa']=make_scorer(cohen_kappa_score)

##TODO: change y to y_ratio
def classifiers_template(y, random_state=10, imputer_strategy='mean' ,pc_n_components=5):
  
  import numpy as np

  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  from sklearn.impute import SimpleImputer
  
  from sklearn.tree import DecisionTreeClassifier
  from sklearn import tree
  from sklearn.svm import SVC, LinearSVC, NuSVC
  from sklearn.gaussian_process import GaussianProcessClassifier
  from sklearn.gaussian_process.kernels import RBF

  from xgboost import XGBClassifier
  from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
  from lightgbm import LGBMClassifier

  from sklearn.linear_model import LogisticRegression

  from sklearn.naive_bayes import GaussianNB

  from sklearn.neighbors import KNeighborsClassifier

  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

  from sklearn.neural_network import MLPClassifier
  from sklearn.ensemble import GradientBoostingClassifier

  classifiers={
  "Nearest_Neighbors_2":            KNeighborsClassifier(2),
  "Nearest_Neighbors_3":            KNeighborsClassifier(3),
  "Nearest_Neighbors_4":            KNeighborsClassifier(4),
  "Nearest_Neighbors_5":            KNeighborsClassifier(5),

  "Decision_Tree_depth5":         DecisionTreeClassifier(max_depth=5, random_state=random_state),
  "Decision_Tree_depth10":        DecisionTreeClassifier(max_depth=10, random_state=random_state),

  "Naive_Bayes":                  GaussianNB(),

  "LinearDiscriminantAnalysis"    :LinearDiscriminantAnalysis(),
  "QuadraticDiscriminantAnalysis" :QuadraticDiscriminantAnalysis(),

  'logReg_mode_l1'   :            LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=random_state),
  'logReg_mode_l2'   :            LogisticRegression(max_iter=1000, random_state=random_state),
  'logReg_model_pca':             LogisticRegression(max_iter=1000, random_state=random_state),

  'RandomForest_model1':          RandomForestClassifier(random_state=random_state),
  'RandomForest_model_balanced':  RandomForestClassifier(class_weight='balanced', random_state=random_state),
  'RandomForest_model_n200':      RandomForestClassifier(n_estimators=200, random_state=random_state),
  'RandomForest_model_n300':      RandomForestClassifier(n_estimators=300, random_state=random_state),
  'Xgboost_n200' :                XGBClassifier(n_estimators=200, random_state=random_state),
  'Xgboost_n200_dp10':            XGBClassifier(n_estimators=200, max_depth=10, random_state=random_state) ,
  'Xgboost_Weighted':             XGBClassifier(scale_pos_weight=float(np.sum(y == 0)) / np.sum(y==1), random_state=random_state) ,
  'Xgboost_Weighted_n200':        XGBClassifier(n_estimators=200, scale_pos_weight=float(np.sum(y == 0)) / np.sum(y==1), random_state=random_state) ,
  'Xgboost_Weighted_n200_dp10':   XGBClassifier(n_estimators=200, max_depth=10, scale_pos_weight=float(np.sum(y == 0)) / np.sum(y==1), random_state=random_state) ,

  'LightGBM':                     LGBMClassifier(random_state=random_state),
  'LightGBM_n200':                LGBMClassifier(n_estimators=200, random_state=random_state),
  'LightGBM_n400':                LGBMClassifier(n_estimators=300, random_state=random_state),
  'LightGBM_n1000':                LGBMClassifier(n_estimators=1000, random_state=random_state),
  'LightGBM_n200_dp10':           LGBMClassifier(max_depth=10,  n_estimators=200, random_state=random_state),
  'LightGBM_n300_dp10':           LGBMClassifier(max_depth=10,  n_estimators=300, random_state=random_state),

  'MLPClassifier1'   :            MLPClassifier(alpha=1, max_iter=1000, random_state=random_state),
  'MLPClassifier_early_stopping'   :  MLPClassifier(alpha=1, max_iter=1000, early_stopping=True, random_state=random_state),
  'MLPClassifier3'   :            MLPClassifier(alpha=1, max_iter=1000, solver='sgd', early_stopping=True, random_state=random_state),

  # "Linear_SVM":                  SVC(kernel="linear", C=0.025, probability=True, random_state=random_state),
  # "RBF_SVM":                     SVC(kernel="rbf", C=0.025, probability=True, random_state=random_state),
  # "NuSVC":                       NuSVC(probability=True, random_state=random_state),

  "AdaBoost":                     AdaBoostClassifier(random_state=random_state),
  # 'bagging':                      BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=random_state),

  }

  basic_params = {"random_state":random_state}

  classifiers2={}
  for name, classifier in classifiers.items():
    ##TODO: find a way to add randomstate here
    # params={**basic_params,**classifier.get_params()}
    # print(params)
    if any([x in name for x in ['pca','DiscriminantAnalysis']]):  
        classifiers2[name]= Pipeline(steps=[("imputer", SimpleImputer(strategy=imputer_strategy)),
                                            ("scaler", StandardScaler()),
                                            ("reduce_dims", PCA(n_components=pc_n_components)),
                                            (name, classifier)])
        
    elif ('xgb' not in name.lower()) & ('gbm' not in name.lower()) :
        classifiers2[name]= Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                            ("scaler", StandardScaler()),
                                            (name, classifier)])
    else:
        classifiers2[name]=classifier
  return classifiers2

def regressors_template(random_state=10):
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.svm import SVR, LinearSVR, NuSVR
  from sklearn.gaussian_process import GaussianProcessRegressor
  from sklearn.gaussian_process.kernels import RBF

  from xgboost import XGBRegressor
  from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor
  from lightgbm import LGBMRegressor

  from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, ElasticNet, ElasticNetCV, Lars, Lasso, LassoLars, LassoLarsIC, LassoCV

  from sklearn.naive_bayes import GaussianNB

  from sklearn.neighbors import KNeighborsRegressor

  from sklearn.neural_network import MLPRegressor  
  from sklearn.ensemble import GradientBoostingRegressor

  from sklearn.pipeline import Pipeline
  import numpy as np
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import RobustScaler
  Regressors={
  "Nearest_Neighbors_2":          KNeighborsRegressor(2),
  "Nearest_Neighbors_3":          KNeighborsRegressor(3),
  "Nearest_Neighbors_4":          KNeighborsRegressor(4),
  "Nearest_Neighbors_5":          KNeighborsRegressor(5),

  "Decision_Tree_depth5":         DecisionTreeRegressor(max_depth=5, random_state=random_state),
  "Decision_Tree_depth10":        DecisionTreeRegressor(max_depth=10, random_state=random_state),

  "Naive_Bayes":                  GaussianNB(),

  'LinearRegression' :            LinearRegression(),
  "Ridge"        :                Ridge(random_state=random_state)        ,
  "SGDRegressor" :                SGDRegressor(random_state=random_state) ,
  "ElasticNet"   :                ElasticNet(random_state=random_state)   ,
  "ElasticNetCV" :                ElasticNetCV(random_state=random_state) ,
  "Lars"         :                Lars(random_state=random_state)         ,
  "Lasso"        :                Lasso()        ,
  "LassoCV"      :                LassoCV(random_state=random_state)    ,
  "LassoLars"    :                LassoLars(random_state=random_state)    ,
  "LassoLarsIC"  :                LassoLarsIC()  ,
  
  'RandomForest_model1':          RandomForestRegressor(random_state=random_state),
  'RandomForest_model_n200':      RandomForestRegressor(n_estimators=200, random_state=random_state),
  'RandomForest_model_n300':      RandomForestRegressor(n_estimators=300, random_state=random_state),
  'Xgboost_n200' :                XGBRegressor(n_estimators=200, random_state=random_state),
  'Xgboost_n200_dp10':            XGBRegressor(n_estimators=200, max_depth=10, random_state=random_state) ,

  'LightGBM':                     LGBMRegressor(random_state=random_state),
  'LightGBM_n200':                LGBMRegressor(n_estimators=200, random_state=random_state),
  'LightGBM_n400':                LGBMRegressor(n_estimators=300, random_state=random_state),
  'LightGBM_n1000':               LGBMRegressor(n_estimators=1000, random_state=random_state),
  'LightGBM_n200_dp10':           LGBMRegressor(max_depth=10,  n_estimators=200, random_state=random_state),
  'LightGBM_n300_dp10':           LGBMRegressor(max_depth=10,  n_estimators=300, random_state=random_state),

  'MLPRegressor1'   :             MLPRegressor(alpha=1, max_iter=1000, random_state=random_state),
  'MLPRegressor_early_stopping':  MLPRegressor(alpha=1, max_iter=1000, early_stopping=True, random_state=random_state),
  'MLPRegressor3'   :             MLPRegressor(alpha=1, max_iter=1000, solver='sgd', early_stopping=True, random_state=random_state),

  # "Linear_SVM":                  SVR(kernel="linear", C=0.025, probability=True, random_state=random_state),
  # "RBF_SVM":                     SVR(kernel="rbf", C=0.025, probability=True, random_state=random_state),
  # "NuSVR":                       NuSVR(probability=True, random_state=random_state),

  "AdaBoost":                     AdaBoostRegressor(random_state=random_state),
  # 'bagging':                      BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10, random_state=random_state),

  }

  basic_params = {"random_state":random_state}

  regressors2={}
  for name, Regressor in Regressors.items():
    # params={**basic_params,**Regressor.get_params()}
    # print(params)
    ##TODO: refactor it:
    if any([x in name for x in ['pca','DiscriminantAnalysis']]):  
        regressors2[name]= Pipeline(steps=[("scaler", RobustScaler()),
                                            ("imputer", SimpleImputer(strategy="median")),
                                            ("reduce_dims", PCA(n_components=20)),
                                            (name, Regressor)])
        
    elif ('xgb' not in name.lower()) & ('gbm' not in name.lower()) :
        regressors2[name]= Pipeline(steps=[("scaler", RobustScaler()),
                                            ("imputer", SimpleImputer(strategy="median")),
                                            (name, Regressor)])
    else:
        regressors2[name]=Regressor
  return regressors2

def ml_scores(y_model, scores_names ,
               multi_class='ovo',
               average ='macro' ##{‘micro’, ‘macro’, ‘samples’, ‘weighted’}
               ):
  """
  Calculate various machine learning evaluation scores for a given model.
  Parameters:
  y_model (pd.DataFrame): DataFrame containing the true labels and predicted labels. 
              It must have columns 'y_true' and 'y_pred'. If 'CV_Iteration' 
              column is not present, it will be added with a default value 'All_data'.
  scores_names (list): List of score names to be calculated. The score names should be keys 
             in the `metric_dict` which maps to the corresponding scoring functions.
  Returns:
  pd.DataFrame: DataFrame containing the calculated scores for each cross-validation iteration, 
          along with the mean and standard deviation of the scores across iterations, 
          and the overall scores for the entire dataset.
  """
  import inspect

  if 'CV_Iteration' not in y_model.columns:
    y_model['CV_Iteration']='All_data'
     
  scores_all=pd.Series(index=scores_names, dtype='float64',name='scores_all')
  scores=pd.DataFrame(index=y_model['CV_Iteration'].unique(), columns=scores_names)
  
  if (y_model.shape[1]>5) &(average =='binary'):
    print("It is a multiclass problem: average argument changed to 'macro'")
    average='macro'

  kwargs_dict={'average':average, 
              'multi_class':multi_class}
    
  for con, score_name in enumerate(scores_names):
    try:
      umetric=metric_dict.get(score_name)

      umetric_args = list(inspect.signature(umetric).parameters)

      kwargs_dict2 = {k: kwargs_dict[k] for k in kwargs_dict if k in umetric_args}

      scores_all.iloc[con] = umetric(y_model['y_true'], y_model['y_pred'], **kwargs_dict2)
      scores.loc[:,score_name] = y_model.groupby('CV_Iteration').apply(
          lambda x: pd.Series({
              score_name: umetric(x['y_true'], x['y_pred'], **kwargs_dict2)
          })
      )
    except Exception as e:
      print (f"{score_name} wasn't added to scores data frame:\n " , str(e))
      
  scores=pd.concat([scores,
                   pd.DataFrame(scores.mean(axis=0)).T.rename({0:'CV_scores_Mean'},axis=0),
                   pd.DataFrame(scores.std(axis=0)).T.rename({0:'CV_scores_STD'},axis=0),
                   pd.DataFrame(scores_all).T
                   ],
                   axis=0,
                  )
  scores=scores.reset_index().rename({'index':'CV'}, axis=1)

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
  from sklearn.pipeline import Pipeline
  from sklearn.model_selection import StratifiedKFold
  from sklearn.model_selection import TimeSeriesSplit
  from sklearn.base import is_classifier
    
  y_model=pd.DataFrame([])   
  df_epochs=pd.DataFrame([])  
  ml_models=[]

  umodel= ml_model[-1] if isinstance(ml_model, Pipeline) else ml_model
  model_name= umodel.__class__.__name__ 
  
  ##TODO:include all models with early_stopping
  
  early_stopping_rounds= ml_model.early_stopping_rounds if 'xgb' in model_name.lower() else None
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
  import warnings
  from sklearn.pipeline import Pipeline
  import datetime as dt
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
  from xgboost import XGBClassifier
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

####------------------------------plot model performance----------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
#import pydotplus
import pylab as pl
import seaborn as sns
sns.set_style("darkgrid")
sns.set(rc = {'figure.figsize':(30,20)})

def ml_comparison_plot(metrics_all, outputFile=None):
    """
    Generates a comparison box plot for machine learning model metrics.
    Parameters:
    metrics_all (pd.DataFrame): DataFrame containing the metrics for different models.
                  The DataFrame should have columns such as 'CV', 'model', 'elapsed_time', 'Feature_nos', and other metric columns.
    outputFile (str, optional): Path to save the output plot. If None, the plot will not be saved. Default is None.
    Returns:
    None: The function displays the plot and optionally saves it to the specified file.
    Notes:
    - The function filters out rows where the 'CV' column contains "CV_scores_Mean", "CV_scores_STD", or "scores_all".
    - If the 'model' column is present in the DataFrame, it is used as the hue for the plot; otherwise, 'CV' is used.
    - The function creates a box plot using seaborn to compare the metrics.
    - The x-axis labels are rotated for better readability.
    - The plot is displayed with a grid and without outliers.
    - If an output file path is provided, the plot is saved to that location.
    """
    ##---plot comparison box plot
    df_tmp=metrics_all.loc[~metrics_all['CV'].isin(["CV_scores_Mean",
                                                    "CV_scores_STD",
                                                    "scores_all"]
                                                  ),
                           :]

    if 'model' in df_tmp.columns.tolist():
      df_tmp=df_tmp.drop('CV',axis=1)
      hue=id_vars='model'
    else:
      id_vars='CV'
      hue='scores'
    ucols=[col for col in df_tmp if col not in ['elapsed_time','Feature_nos']]
    df_long = pd.melt(df_tmp[ucols], id_vars=[id_vars], var_name=['scores'])  
    # sns.set_style("darkgrid")
    plt.figure(figsize = (25,15))
    
    uplot   = sns.boxplot(x="scores",
                          y="value",
                          hue=hue,
                          data=df_long,
                          # orient='h',  ##it takes forever
                          showfliers=False,
                          ) 
    
    uplot.set_xticklabels(uplot.get_xticklabels(),rotation=90)
    uplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    uplot.grid()

    if outputFile is not None:
      # graphfile=os.path.join(outputFile,'compare_models.png')
      print("plot save in %s" %outputFile)
      plt.savefig(outputFile)
      plt.show()
      plt.close() 
 
def learning_curve_early_stopping(df_epochs, outputFile=None):
  """
  Plots the learning curve with early stopping for XGBoost models.
  This function visualizes the training and validation performance over epochs
  for cross-validation iterations, highlighting the point of early stopping.
  Parameters:
  df_epochs (pd.DataFrame): DataFrame containing the epochs data. It should include columns for epochs, 
                CV_Iteration, and best_ntree, along with performance metrics for training 
                and validation.
  outputFile (str, optional): Path to save the output plot. If None, the plot is not saved. Default is None.
  Returns:
  None
  """
  ###https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/

  ##TODO: it is only for xgb now (best_ntree)
  cols=df_epochs.columns[~df_epochs.columns.str.contains('Validation_|Train_')].tolist()
  df_epochs_melted=df_epochs.melt(id_vars=cols)
  uPlot=sns.relplot(
                    data=df_epochs_melted,
                    y="value",
                    x="epochs",
                    col="CV_Iteration",
                    hue="variable",
                    style="variable",
                    kind="line",
        #             markers=True,
                    palette=['green', 'black'],
                    col_wrap=3
                )

  axes = uPlot.axes.flatten()

  sns.set(rc = {'figure.figsize':(60,30)})
  for con, ax in enumerate(axes):
      data_tmp=df_epochs_melted[df_epochs_melted['CV_Iteration']==con]
      xc=data_tmp.loc[data_tmp['best_ntree'],'epochs']
      ax.axvline(xc.iloc[0], ls='-', linewidth=3, color='red', alpha=0.75)
    
  # fig, ax = plt.subplots(math.ceil((df_epochs_melted['CV_Iteration'].nunique())/3), 3, figsize=(30, 20))
  # for subplot, var in enumerate(df_epochs_melted['CV_Iteration'].unique()):
  #   axs=ax.flatten()[subplot]
  #   data_tmp=df_epochs_melted[df_epochs_melted['CV_Iteration']==var]
  #   sns.lineplot(
  #                ax=axs,
  #                data=data_tmp,
  #                 y="value",
  #                 x="epochs",
  #                 hue="variable",
  #                 style="variable",
  #               )
  #   xc=data_tmp.loc[data_tmp['best_ntree'],'epochs']
  #   axs.title.set_text(f'cv_itr: {var}')
  #   axs.axvline(x=xc.iloc[0],
  #               color='red',
  #               linestyle='--')
  
  if outputFile is not None:
    figure = uPlot.get_figure()
    # ,"learning_curve.png")
    figure.savefig(outputFile, bbox_inches='tight')
    plt.close('all')

def gainNlift(y, model_prob, pos_label, outputFile, groupNo=25):
    """
    Calculate and plot Gain and Lift charts for a given model's predictions.
    
    Parameters:
    y (pd.Series): True labels.
    model_prob (pd.Series): Predicted probabilities.
    pos_label (int): The label of the positive class.
    outputFile (list): List of file paths to save the Gain and Lift charts.
    groupNo (int): Number of groups to divide the data into for the charts.
    
    Returns:
    out (pd.DataFrame): DataFrame containing the gain and lift values.
    df_gain_chart (pd.DataFrame): DataFrame for the gain chart.
    df_lift_chart (pd.DataFrame): DataFrame for the lift chart.
    """
    ## Lift/cumulative gains charts aren't a good way to evaluate a model (as it cannot be used for comparison between ml_models), and are instead a means of evaluating the results where your resources are finite. Either because there's a cost to action each result (in a marketing scenario) or you want to ignore a certain number of guaranteed voters, and only action those that are on the fence. Where your model is very good, and has high classification accuracy for all results, you won't get much lift from ordering your results by confidence.(https://stackoverflow.com/questions/42699243/how-to-build-a-lift-chart-a-k-a-gains-chart-in-python)

    ## gain Interpretation:
    ## % of targets (events) covered at a given decile level. For example,  80% of targets covered in top 20% of data based in model. In the case of propensity to buy model, we can say we can identify and target 80% of customers who are likely to buy the product by just sending email to 20% of total customers.
    ## lift Interpretation:
    ## The Cum Lift of 4.03 for top two deciles, means that when selecting 20% of the records based on the model, one can expect 4.03 times the total number of targets (events) found by randomly selecting 20%-of-file without a model.

    ##note: when intersted column has much more freq than the other:
    #  df[df.columns[0].value_counts()
            # bluecurvetv        127289
            # Decline_offer    2853
    ##the lift and gain chart of intersted column doesnot show any supererioty of using ML,(model and random output shows same output)

    df=pd.concat([y,model_prob],axis=1)
    df.sort_values(by=df.columns[1], ascending=False, inplace=True)

    def gain_stp1(subset):
        pos_event=sum(subset[y.name]==pos_label)
        return  len(subset), pos_event

    # subset=np.array_split(df,groupNo)[0]
    tmp=list(map(gain_stp1,np.array_split(df,groupNo)))
    out = pd.DataFrame(tmp,columns=['case','event'])
    out['event%']=out['event']/out['event'].sum()*100
    out['cum_case'] = out['case'].cumsum()
    out['cum_case%'] = out['cum_case']/out['case'].sum()*100
    out['gain'] = out['event%'].cumsum()
    out['cum_lift'] = out['gain']/out['cum_case%']

    row_no=int(out.shape[0])

    df_gain_chart=pd.DataFrame(out['gain'].tolist()+out['cum_case%'].tolist(),columns=['values'])
    df_gain_chart['x']=pd.Series(out['cum_case%'].tolist()*2)
    df_gain_chart['selection method']=pd.Series(['model']*row_no+['random']*row_no)
    df_gain_chart = pd.concat([df_gain_chart, pd.DataFrame.from_dict({'values':[0,0],"x":[0,0],'selection method':['model','random']})], ignore_index=True)

    df_lift_chart=pd.DataFrame(out['cum_lift'].tolist()+[1]*row_no,columns=['values'])
    df_lift_chart['x']=pd.Series(out['cum_case%'].tolist()*row_no)
    df_lift_chart['selection method']=pd.Series(['model']*row_no+['random']*row_no)
        
    fig, ax = plt.subplots(2,1,figsize=(20, 10))   
    
    uPlot1=sns.lineplot(data=df_gain_chart,
                        ax=ax[0],
                        x='x',
                        y='values',
                        hue='selection method',
                        style='selection method',
                        markers=True
                       )
    uPlot1.set(xlabel='', ylabel='% of events')
    ax[0].set_title('Gain Chart')

    uPlot2=sns.lineplot(data=df_lift_chart,
                        ax=ax[1],
                        x='x',
                        y='values',
                        hue='selection method',
                        style='selection method',
                        markers=True
                       )
    uPlot2.set(xlabel='% 0f data sets', ylabel='Lift')
    ax[1].set_title('Lift Chart')
    
    plt.ylim(0,int(df_lift_chart['values'].max()+1))
    plt.xlim(0,100) 
    
    if outputFile is not None:
      uPlot1.get_figure().savefig(outputFile[0], bbox_inches='tight')
      uPlot2.get_figure().savefig(outputFile[1], bbox_inches='tight')
      plt.close('all')

    return out, df_gain_chart, df_lift_chart
  
def precision_recall_curve2(y, model_prob, pos_label, outputFile=None, **kwargs):
    """
    Plots the precision-recall curve and calculates the area under the curve (AUC).
    
    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True binary labels.
    model_prob : array-like of shape (n_samples,)
        Estimated probabilities or decision function.
    pos_label : int or str
        The label of the positive class.
    outputFile : str, optional
        If provided, the plot will be saved to this file.
    **kwargs : dict
        Additional arguments passed to sklearn's precision_recall_curve.
        
    Returns
    -------
    tuple
        - pandas.DataFrame: Contains precision, recall, and thresholds
        - list: Indices used for setting x-ticks on the plot
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve, auc
    
    # Reset any existing plots
    plt.clf()
    
    # Calculate precision-recall curve
    model_precision, model_recall, thresholds = precision_recall_curve(
        y_true=y,
        probas_pred=model_prob,
        pos_label=pos_label,
        **kwargs
    )
    
    # Calculate AUC
    model_auc_rp = auc(model_recall, model_precision)
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(20, 10))
    # Set background colors
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Add horizontal and vertical grid lines
    ax.grid(True, which='major', linestyle='--', alpha=0.7, color='gray')
    ax.grid(True, which='minor', linestyle=':', alpha=0.4, color='gray')
    ax.minorticks_on()  # Enable minor ticks
    
    # Set specific grid for both axes
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
    
    # Create DataFrame for plotting
    df_rp = pd.DataFrame({
        'Precision': model_precision[:-1],
        'Recall': model_recall[:-1],
        'thresholds': thresholds
    })
    
    # Create the precision-recall curve plot
    ax.plot(df_rp['Recall'], df_rp['Precision'], 
            marker='o', markersize=4, alpha=0.1, 
            linestyle='-', linewidth=1)
    
    # Configure plot aesthetics
    ax.set_title('Precision Recall Curve', pad=20, fontsize=12)
    ax.set_xlabel('Recall/(threshold)', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    
    # Calculate and plot no-skill line
    no_skill = len(y[y==1]) / len(y)
    ax.plot([0, 1], [no_skill, no_skill], 
            linestyle='--', color='black', label='No Skill')
    ax.text(0.9, no_skill, 'No skill line', 
            color='black', fontsize=10)
    
    # Configure x-axis ticks with thresholds
    df_rp_unique = df_rp.drop_duplicates(subset=['Recall']).reset_index()
    interval_no = min(15, df_rp_unique.shape[0])
    idx = list(np.linspace(
        df_rp_unique.index.min(),
        df_rp_unique.index.max(),
        interval_no,
        endpoint=True,
        dtype='int'
    ))
    
    # Set tick labels
    recall_values = df_rp_unique.iloc[idx]['Recall']
    threshold_values = df_rp_unique.iloc[idx]['thresholds'].round(3).astype(str)
    ax.set_xticks(recall_values)
    ax.set_xticklabels(
        [f"{round(x,2)}(t={y})" for x, y in zip(recall_values, threshold_values)],
        rotation=90
    )
    
    # Add AUC annotation
    ax.annotate(
        f'ROC of Precision Recall curve={model_auc_rp:.3f}',
        xy=(0.4, 0),
        xycoords='axes fraction',
        xytext=(-20, 25),
        textcoords='offset pixels',
        horizontalalignment='right',
        verticalalignment='bottom',
        fontsize=10
    )
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot if outputFile is provided
    if outputFile is not None:
        plt.savefig(outputFile, bbox_inches='tight', dpi=300)
        plt.show()
    
    return df_rp, idx

def roc_curve2(y, model_prob, pos_label, outputFile=None, **kwargs):
    """
    Generates and plots the ROC curve for a given set of true labels and predicted probabilities.
    Parameters:
    -----------
    y : array-like
      True binary labels.
    model_prob : array-like
      Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
    pos_label : int or str
      Label considered as positive and others are considered negative.
    outputFile : str, optional
      If provided, the plot will be saved to this file.
    **kwargs : dict, optional
      Additional keyword arguments to pass to `roc_auc_score`.
    Returns:
    --------
    df_roc : pandas.DataFrame
      DataFrame containing the false positive rate, true positive rate, and thresholds.
    model_auc : float
      Computed Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    Notes:
    ------
    - The function uses `roc_auc_score` from `sklearn.metrics` to compute the AUC.
    - The ROC curve is plotted using `matplotlib`.
    - The function can save the plot to a file if `outputFile` is provided.
    """
    ##TODO: add **kwargs to roc_curve, after sepration augs
    from sklearn.metrics import roc_auc_score

    model_auc = roc_auc_score(y_true=y, y_score=model_prob, **kwargs)
    ## NOTE:    Different result with roc_auc_score() and auc()
    # : https://stackoverflow.com/questions/31159157/different-result-with-roc-auc-score-and-auc
    # from sklearn.metrics import roc_curve,auc
    # model_fpr, model_tpr, _ = roc_curve(y.map(map_lbls_inv), model_prob)
    # model_auc2 = auc(model_fpr, model_tpr)

    tmpTxt='ROC AUC=%.3f (random selection=.5)' % (model_auc)
#     print(tmpTxt+'\n','green')
    from sklearn.metrics import roc_curve
    model_fpr, model_tpr, thresholds = roc_curve(y_true=y, y_score=model_prob, pos_label=pos_label)

    thresholds[0]=1
    df_roc=pd.DataFrame([model_fpr,
                      model_tpr,
                      thresholds], index=['False_Positive_Rate', 'True_Positive_Rate', 'thresholds']).T

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot([0,1],[0,1], linestyle='--', label='No Skill')    ###or ns_fpr, ns_tpr =[0,1],[0,1]= roc_curve(y, [0 for _ in range(len(y))], pos_label)
    plt.plot(model_fpr, model_tpr, marker='.', label='Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    ax.legend(loc='upper right', frameon=True)
    ax.annotate('ROC AUC=%.3f (random selection=.5)' % (model_auc),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20),
                textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom')

    df_roc_tmp=df_roc.drop_duplicates(subset=['False_Positive_Rate']).reset_index()
    interval_no=min(15,df_roc_tmp.shape[0])
    idx=list(np.linspace(df_roc_tmp.index.min(),df_roc_tmp.index.max(),interval_no,endpoint=True,dtype='int'))
    plt.xticks(df_roc_tmp.iloc[idx]['False_Positive_Rate'])
    ticks_loc = ax.get_xticks().tolist()
    threshs=(df_roc_tmp.iloc[idx]['thresholds'].round(3).astype(str)) 

    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels([str(round(x,2))+"(t="+y+")" for x,y in zip(ticks_loc,threshs)])
    plt.xticks(rotation=90)

    if outputFile is not None:
      plt.savefig(outputFile, bbox_inches='tight')
      plt.close('all')

    return df_roc, model_auc
  
def reliability_diagram(y, model_prob, pos_label, outputFile, **kwargs):
    """
    Plots a reliability diagram to visualize the calibration of a probabilistic classifier.
    Parameters:
    -----------
    y : array-like of shape (n_samples,)
      True labels of the data.
    model_prob : array-like of shape (n_samples,)
      Predicted probabilities of the positive class.
    pos_label : int or str
      The label of the positive class.
    outputFile : str or None
      The file path to save the plot. If None, the plot is not saved.
    **kwargs : additional keyword arguments
      Additional arguments to pass to the `calibration_curve` function.
    Returns:
    --------
    prob_true : array-like of shape (n_bins,)
      The true probabilities for each bin.
    prob_pred : array-like of shape (n_bins,)
      The predicted probabilities for each bin.
    prob_true_norm : array-like of shape (n_bins,)
      The true probabilities for each bin after normalization.
    prob_pred_norm : array-like of shape (n_bins,)
      The predicted probabilities for each bin after normalization.
    """
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred= calibration_curve(y_true=y, y_prob=model_prob, n_bins=50, normalize=False, **kwargs)  # pos_label=pos_label, 
    prob_true_norm, prob_pred_norm= calibration_curve(y_true=y, y_prob=model_prob,  n_bins=50,normalize=True, **kwargs)  # pos_label=pos_label, 

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot([0,1],[0,1]) 
    plt.plot(prob_pred_norm, prob_true_norm, label='Normlized') 
    plt.plot(prob_pred, prob_true, label='Original') 
    plt.grid() 
    plt.xlabel("Average probability")
    plt.ylabel("Fraction of positive")
    plt.title("Reliability diagram") 
    ax.legend(loc='upper right', frameon=True)
    
    if outputFile is not None:
      plt.savefig(outputFile, bbox_inches='tight')
      plt.close('all')
      
    return prob_true, prob_pred, prob_true_norm, prob_pred_norm

def plot_confusion_matrix2(y_model, map_lbls, outputFile=None, ncol=3, all_data_flag=True):
  """
  Plots confusion matrices for model predictions, optionally saving the plot to a file.
  Parameters:
  y_model (pd.DataFrame): DataFrame containing the true labels and predicted labels. 
              It should have columns 'y_true' and 'y_pred'. Optionally, it can have a 'CV_Iteration' column for cross-validation iterations.
  map_lbls (dict): Dictionary mapping the original labels to the desired labels for the confusion matrix.
  outputFile (str, optional): Path to save the output plot. If None, the plot is not saved. Default is None.
  ncol (int, optional): Number of columns for the subplot grid. Default is 3.
  Returns:
  pd.Series: A series of confusion matrices, indexed by the cross-validation iteration or 'All_data' if no cross-validation is used.
  """
  ##y_model=pd.concat([y_true, y_pred],axis=1)
  
  from sklearn.metrics import confusion_matrix
  y_model1=y_model.copy()
  
  if all_data_flag :
    ###TODO: use it for x-validation and not for multilabel
    if (y_model1['CV_Iteration'].nunique()!=1):
      y_model1['CV_Iteration']='cv_'+y_model1['CV_Iteration'].astype(str)
      y_model_all=y_model.copy()
      
      y_model_all['CV_Iteration']='All_data'
      y_model1=pd.concat([y_model1,y_model_all],axis=0)
      
    ncol=ncol
    fig_size=(25,17)
  else:
    y_model1['CV_Iteration']='All_data'
    ncol=1
    fig_size=(10,5)

  # print(y_model1)
  confMats=pd.Series([])
  #confMats=pd.Series([],index=y_model['CV_Iteration'].unique())    
  
  fig, axs = plt.subplots(math.ceil(y_model1['CV_Iteration'].nunique()/ncol), ncol, figsize=fig_size)   
  
  axs=np.array([axs]) if ncol==1 else axs

  for cont, (cv, y_model_sub) in  enumerate(y_model1.groupby(['CV_Iteration'])):  

    cv=cv[0] if isinstance(cv, tuple) else cv

    y_true=y_model_sub[['y_true']]
    y_pred=y_model_sub[['y_pred']]

    confMat = pd.DataFrame(confusion_matrix(y_true, y_pred))
    confMat=confMat.rename(columns=map_lbls).rename(map_lbls,axis=1).rename(map_lbls,axis=0)
    confMat.index.name='True label'
    confMat.columns.name='Predicted label'

    confMats[cv]=confMat
    uPlot=sns.heatmap(ax=axs.flatten()[cont],
                    data=confMat,
                    annot=True,
                    cmap="YlGnBu",
                    fmt="g",cbar=False
                   )

    axs.flatten()[cont].set_title(f'{cv}')    
    fig.tight_layout()

  if outputFile is not None:
    figure = uPlot.get_figure()
    figure.savefig(outputFile, bbox_inches='tight')
    plt.close('all')
    
  return confMats

def feature_importance_batch(umodel, X, y):
    """
    Calculate and plot feature importance for a given model and dataset.
    Parameters:
    umodel (object): The machine learning model or pipeline containing the model.
    X (pd.DataFrame): The input features for the model.
    y (pd.Series or np.array): The target variable.
    Returns:
    tuple: A tuple containing:
      - feature_importance (pd.Series or None): The importance of each feature, or None if the model is not tree-based.
      - sel_features (list): A list of selected features based on their importance.
    Notes:
    - This function currently supports only tree-based models, specifically XGBoost.
    - For XGBoost models, the function fits the model, plots the feature importance, and returns the importance values.
    - If the model is not tree-based, the function returns None for feature importance and all columns of X as selected features.
    """
    from sklearn.pipeline import Pipeline

    umodel1 = umodel[-1] if isinstance(umodel, Pipeline) else umodel
    model_name = umodel1.__class__.__name__

    ##TODO: generalize for any tree model:
    if "xgb" in model_name.lower():
        from xgboost import plot_importance

        umodel.fit(X, y)
        plt.figure(
            figsize=(100, 100),
            dpi=150,
        )
        sns.set(rc={"figure.figsize": (20, 10)}, font_scale=1)
        plot_importance(umodel, max_num_features=30)
        plt.show()
        plt.close()

        feature_importance = pd.Series(
            umodel.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        print(
            "non zero features:",
            feature_importance[feature_importance >= 0.01].index.tolist(),
        )
        print(
            "zero features:",
            feature_importance[feature_importance < 0.01].index.tolist(),
        )

        plt.figure(figsize=(5, 2), dpi=150)

        importance_hist = feature_importance.hist()

        sns.set(rc={"figure.figsize": (20, 10)}, font_scale=1.5)
        importance_hist

        sel_features = feature_importance[feature_importance > 0.01].index.tolist()
    else:
        print("The model is not tree based- returning None")
        feature_importance = None
        sel_features = X.columns

    return feature_importance, sel_features

def pdp_plot_batch(X, umodel, sel_features):
    """
    Generate partial dependence plots (PDP) and individual conditional expectation (ICE) plots for a given model and selected features.
    Parameters:
    X : array-like or DataFrame of shape (n_samples, n_features)
      The data to use for generating the plots.
    umodel : estimator object
      A fitted scikit-learn estimator (model) for which the partial dependence plots are to be computed.
    sel_features : list of str or list of int
      The features for which the partial dependence plots are to be generated. These can be specified by their names or column indices.
    Returns:
    None
      This function does not return any value. It generates and displays the partial dependence plots and individual conditional expectation plots.
    """
    # print('The scikit-learn version is {}.'.format(sklearn.__version__))
    from sklearn.inspection import PartialDependenceDisplay

    ###see https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#sphx-glr-auto-examples-inspection-plot-partial-dependence-py
    print(
        "Computing partial dependence plots and individual conditional expectation..."
    )

    _, ax = plt.subplots(
        # ncols=3, nrows=math.ceil(len(sel_features)/3),
        figsize=(30, 30),
        sharey=False,
        constrained_layout=True,
    )

    features_info = {
        "features": sel_features,
        "kind": "both",
        "centered": True,
        # "categorical_features": ['plant_K1'],
    }
    display = PartialDependenceDisplay.from_estimator(
        umodel,
        X,
        **features_info,
        ax=ax,
        # **common_params,
    )

def shap_plots_batch(X, y, umodel, test_size=0.2, kmeans=None, random_state=100):
    """
    Generates SHAP plots for a given model and dataset.
    Parameters:
    X (pd.DataFrame or np.ndarray): Feature matrix.
    y (pd.Series or np.ndarray): Target vector.
    umodel (object): The machine learning model to be used.
    test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
    kmeans (int, optional): Number of clusters for KMeans summarization. If None, the whole training set is used. Default is None.
    random_state (int, optional): Random seed for reproducibility. Default is 100.
    Returns:
    shap_values (list): SHAP values for the test set.
    """
    import shap
    from sklearn.model_selection import RandomizedSearchCV, train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
                                                        X, y,
                                                        test_size=test_size,
                                                        random_state=random_state
                                                      )
    # X_train, X_test= X, X

    ###TODO: why it is working only in xgboost:
    # explainer = shap.Explainer(umodel, X)
    # shap_values = explainer(X)
    # # shap.plots.beeswarm(shap_values)
    # shap.plots.bar(shap_values)
    # shap.summary_plot(shap_values, plot_type='violin')

    umodel.fit(X_train.values, y_train)

    ###https://www.kaggle.com/code/hwwang98/shapley-value-feature-research#kln-19:
    # rather than use the whole training set to estimate expected values, we summarize with
    # a set of weighted kmeans, each weighted by the number of points they represent.
    
    if kmeans is not None:
       udata= shap.kmeans(X_train, kmeans)
    else:
       udata=X_train.values
    explainer = shap.KernelExplainer(
                                      model=umodel.predict,
                                      data=udata,
                                      # link="identity"
                                    )
    shap_values = explainer.shap_values(
                                          X_test,
                                          #  nsamples=1000
                                      )

    # shap.plots.bar(shap_values)##not working
    shap.summary_plot(shap_values, X_test, plot_type="violin")
    # shap.force_plot(explainer.expected_value, shap_values, X_test) ##not working
    # shap.dependence_plot("NOH", shap_values, X_test)##not working

    return shap_values

####------------------------------Xgboost HyperParameter tuning----------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
# When working with imbalanced data sets, accuracy is not always the best metric to evaluate the performance of a classifier, because it can be misleading. Some alternative metrics that are better suited for imbalanced data sets are:

# Precision: the proportion of true positive predictions among all positive predictions. It measures the ability of the classifier to avoid false positives.
# Recall (Sensitivity or TPR): the proportion of true positive predictions among all actual positive observations. It measures the ability of the classifier to detect all positive observations.
# F1-score: the harmonic mean of precision and recall. It is a balance between precision and recall.
# AUC-ROC: the area under the Receiver Operating Characteristic curve. It measures the ability of the classifier to distinguish between positive and negative observations.
# AUC-PR: the area under the precision-recall curve. It also measures the ability of the classifier to distinguish between positive and negative observations, but it puts more emphasis on the true positive rate.
# G-mean: the geometric mean of recall and specificity. It is a balance between recall and specificity and is sensitive to imbalanced data.
# These metrics can be calculated using the following sklearn functions:

# Precision: sklearn.metrics.precision_score(y_true, y_pred)
# Recall: sklearn.metrics.recall_score(y_true, y_pred)
# F1-score: sklearn.metrics.f1_score(y_true, y_pred)
# AUC-ROC: sklearn.metrics.roc_auc_score(y_true, y_score)
# AUC-PR: sklearn.metrics.average_precision_score(y_true, y_score) or use    precision, recall, _ = precision_recall_curve(y_test, y_pred) and AUC_PR = auc(recall, precision)
# G-mean: sklearn.metrics.geometric_mean_score(y_true, y_pred)


### NOTE: xgboost parameters: https://xgboost.readthedocs.io/en/latest/parameter.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score5 

### NOTE: Handle Imbalanced Dataset:
### https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html:
### For common cases such as ads clickthrough log, the dataset is extremely imbalanced. This can affect the training of XGBoost ucase, and there are two ways to improve it:
### If you care only about the overall performance metric (AUC) of your prediction:
    ### Balance the positive and negative weights via scale_pos_weight
    ### Use AUC for evaluation
### If you care about predicting the right probability:
    ### In such a case, you cannot re-balance the dataset
    ### Set parameter max_delta_step to a finite number (say 1) to help convergence
## AUPRC and Average Precision:
###https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
## The baseline of AUPRC is equal to the fraction of positives. If a dataset consists of 8% cancer examples and 92% healthy examples, the baseline AUPRC is 0.08, so obtaining an AUPRC of 0.40 in this scenario is good! AUPRC is most useful when you care a lot about your model handling the positive examples correctly.

###using scale_pos_weight and sample_weight  leads to almost the same auc:
# when  "tree_method":"gpu_hist" and "predictor":"gpu_predictor" :  (0.842775:0.842165)
# without using gpu results                                      :  (0.842609:0.842686)

# The hyperparameters that have the greatest effect on XGBoost objective metrics are: alpha, min_child_weight, subsample, eta, and num_round.

# Fill reasonable values for key inputs:
# learning_rate: 0.01
# n_estimators: 100 if the size of your data is high, 1000 is if it is medium-low
# max_depth: 3
# subsample: 0.8
# colsample_bytree: 1
# gamma: 1

###  I usually use 50 rounds for early stopping with 1000 trees in the model. I’ve seen in many places recommendation to use about 10% of total number of trees for early stopping
### early_stopping_rounds offers a way to automatically find the ideal value for n_estimators. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.

##xgboost GPU parameter:
            ##NOte: Gpu is very fast, but it doesnot give good model performance like cpu, so tune your model with gpu and use cpu to generate final prediction on CPU 
#             "tree_method":"gpu_hist",
            ## "gpu_id":1,
#             "predictor":"gpu_predictor",

import sklearn.metrics as metrics
from sklearn.pipeline import Pipeline
def ml_tuner(trial,
             sk_model,
             model_params,
             X,
             y,
             sk_fold,
             var_in_model_params,
             Umetric='auc',
             use_early_Stopping=False,
             early_stopping_rounds=300,
             use_callbacks=False
            ):
  import optuna


  local_vars=locals()

  ##TODO: it is not working:
  # if 'Pipeline' in  str(type(sk_model)):
  #   model_sub = sk_model.steps[-1][1]
  #   sk_model.steps[-1][1]=model_sub(**model_params) 
  #   model = sk_model
  # else:
  #   model = sk_model(**model_params) 

  ##TODO: Revise logic:
  if model_params is not None:
    model = sk_model(**model_params) 
  ##Umetric is used when use_early_Stopping=True, otherwise it was infered from model_params('eval_metric')
    model_params={key:(eval(par, var_in_model_params, local_vars) if (isinstance(par, str) and (par[:14]=='trial.suggest_')) else par) for (key, par) in model_params.items()}
    print(model_params)
  else:
    model=sk_model

  if use_early_Stopping:
    #eval_metric used by early stopping comes from xgboost package and there is not same as metrics in sklearn:
      #https://xgboost.readthedocs.io/en/stable/parameter.html
      #https://scikit-learn.org/stable/modules/classes.html 
    #for instance use recall metric doesnot exists in eval_metric. Moreover same metric have difference name:  aucpr in eval_metric is same as average_precision_score in metrics
    ##so there is need to used dictionary to convert  eval_metric to metric
    ##TODO: add all metric to it:
    eval_metric_dict={'auc'    : 'auc',
                      'aucpr'  : 'aucpr',
                      }
  
    Umetric=eval_metric_dict.get(model_params.get('eval_metric'))
      
  if use_callbacks:
    # Each of ‘validation_0‘ and ‘validation_1‘ correspond to the order that datasets were provided to the eval_set argument in the call to fit(). 
    #eval_set = [(X_train, y_train), (X_val, y_val)]
    #it is much faster than [(X_train, y_train), (X_val, y_val)]
  
    #when len(eval_set)=2:
    observation_key="validation_1-"+model_params['eval_metric'] 
    #when len(eval_set)=1:
    # observation_key="validation_0-"+model_params['eval_metric']
    
  # Add a callback for pruning.
    pruning_callback = [optuna.integration.XGBoostPruningCallback(trial, observation_key)]
  
  else:
    pruning_callback=None
    
  y_model,_=ml_prediction(model,
                              X,
                              y,
                              sk_fold,
                              use_early_Stopping=use_early_Stopping,
                              early_stopping_rounds=early_stopping_rounds,
                              pruning_callback=pruning_callback,
                              )

  scores=ml_scores(y_model, [Umetric], 
                  multi_class='ovo',
                  average ='macro' )
  print(scores)
  scores=scores.loc[scores['CV']=='CV_scores_Mean',Umetric]
  
  # scores=[]    
  # for cv_itr,(train_index, val_index) in enumerate(sk_fold.split(X,y)):
  #   model = sk_model(**model_params) 

  #   X_train, X_val = X.iloc[train_index,:], X.iloc[val_index,:] 
  #   y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
  #   ##TODO: it is only for xgboost, cover other mls
  #   if use_early_Stopping:
  #     eval_set = [(X_val, y_val)]

  #     if use_callbacks:
  #       # Each of ‘validation_0‘ and ‘validation_1‘ correspond to the order that datasets were provided to the eval_set argument in the call to fit(). 
  #       #eval_set = [(X_train, y_train), (X_val, y_val)]
  #       #### it is much faster than [(X_train, y_train), (X_val, y_val)]
        
  #       if len(eval_set)==2:
  #         observation_key="validation_1-"+model_params['eval_metric']
  #       else:
  #         observation_key="validation_0-"+model_params['eval_metric']

  #       # Add a callback for pruning.
  #       pruning_callback = [optuna.integration.XGBoostPruningCallback(trial, observation_key)]

  #       model.fit(X_train,
  #                 y_train,
  #                 early_stopping_rounds=early_stopping_rounds,
  #                 eval_set=eval_set,
  #                 callbacks=pruning_callback,
  #                 verbose=200
  #                 )
  #     else:
  #       model.fit(X_train,
  #                 y_train,
  #                 early_stopping_rounds=early_stopping_rounds,
  #                 eval_set=eval_set,
  #                 verbose=200
  #                 )   
      
  #     ##TODO: add all metric to it:
  #     #eval_metric used by early stopping comes from xgboost package and there is not same as metrics in sklearn:
  #       #https://xgboost.readthedocs.io/en/stable/parameter.html
  #       #https://scikit-learn.org/stable/modules/classes.html 
  #     #for instance use recall metric doesnot exists in eval_metric. Moreover same metric have difference name:  aucpr in eval_metric is same as average_precision_score in metrics
  #     ##so there is need to used dictionary to convert  eval_metric to metric

  #     eval_metric_dict={'auc'    : metrics.roc_auc_score,
  #                       'aucpr'  : metrics.average_precision_score,
  #                       'logloss': metrics.accuracy_score
  #                      }

  #     Umetric_func=eval_metric_dict.get(model_params.get('eval_metric'))
    
  #   else:
  #     model.fit(X_train,
  #               y_train,
  #               )
      
  #     Umetric_func=metric_dict.get(Umetric)
      
    # y_val_model = model.predict(X_val)
    # score=Umetric_func(y_val, y_val_model)  
    # scores.append(score)
    #scores=np.mean(scores)
  return scores

from sklearn.metrics import accuracy_score, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from typing import Any, Dict, Union 

def hyperparameter_tuning(
                        space: Dict[str, Union[float, int]],
                        X: pd.DataFrame, y: pd.Series,
                        sk_fold,   #[X_test,y_test]
                        early_stopping_rounds: int=50,
                        Umetric:callable=accuracy_score)-> Dict[str, Any]:

  # When working with imbalanced data sets, accuracy is not always the best metric to evaluate the performance of a classifier, because it can be misleading. Some alternative metrics that are better suited for imbalanced data sets are:

  # Precision: the proportion of true positive predictions among all positive predictions. It measures the ability of the classifier to avoid false positives.
  # Recall (Sensitivity or TPR): the proportion of true positive predictions among all actual positive observations. It measures the ability of the classifier to detect all positive observations.
  # F1-score: the harmonic mean of precision and recall. It is a balance between precision and recall.
  # AUC-ROC: the area under the Receiver Operating Characteristic curve. It measures the ability of the classifier to distinguish between positive and negative observations.
  # AUC-PR: the area under the precision-recall curve. It also measures the ability of the classifier to distinguish between positive and negative observations, but it puts more emphasis on the true positive rate.
  # G-mean: the geometric mean of recall and specificity. It is a balance between recall and specificity and is sensitive to imbalanced data.
  # These metrics can be calculated using the following sklearn functions:

  # Precision: sklearn.metrics.precision_score(y_true, y_pred)
  # Recall: sklearn.metrics.recall_score(y_true, y_pred)
  # F1-score: sklearn.metrics.f1_score(y_true, y_pred)
  # AUC-ROC: sklearn.metrics.roc_auc_score(y_true, y_score)
  # AUC-PR: sklearn.metrics.average_precision_score(y_true, y_score) or use    precision, recall, _ = precision_recall_curve(y_test, y_pred) and AUC_PR = auc(recall, precision)
  # G-mean: sklearn.metrics.geometric_mean_score(y_true, y_pred)


  ### NOTE: xgboost parameters: https://xgboost.readthedocs.io/en/latest/parameter.html
  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score5 

  ### NOTE: Handle Imbalanced Dataset:
  ### https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html:
  ### For common cases such as ads clickthrough log, the dataset is extremely imbalanced. This can affect the training of XGBoost ucase, and there are two ways to improve it:
  ### If you care only about the overall performance metric (AUC) of your prediction:
      ### Balance the positive and negative weights via scale_pos_weight
      ### Use AUC for evaluation
  ### If you care about predicting the right probability:
      ### In such a case, you cannot re-balance the dataset
      ### Set parameter max_delta_step to a finite number (say 1) to help convergence
  ## AUPRC and Average Precision:
  ###https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
  ## The baseline of AUPRC is equal to the fraction of positives. If a dataset consists of 8% cancer examples and 92% healthy examples, the baseline AUPRC is 0.08, so obtaining an AUPRC of 0.40 in this scenario is good! AUPRC is most useful when you care a lot about your model handling the positive examples correctly.

  ###using scale_pos_weight and sample_weight  leads to almost the same auc:
  # when  "tree_method":"gpu_hist" and "predictor":"gpu_predictor" :  (0.842775:0.842165)
  # without using gpu results                                      :  (0.842609:0.842686)

  # The hyperparameters that have the greatest effect on XGBoost objective metrics are: alpha, min_child_weight, subsample, eta, and num_round.

  # Fill reasonable values for key inputs:
  # learning_rate: 0.01
  # n_estimators: 100 if the size of your data is high, 1000 is if it is medium-low
  # max_depth: 3
  # subsample: 0.8
  # colsample_bytree: 1
  # gamma: 1

  ###  I usually use 50 rounds for early stopping with 1000 trees in the model. I’ve seen in many places recommendation to use about 10% of total number of trees for early stopping
  ### early_stopping_rounds offers a way to automatically find the ideal value for n_estimators. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.

  ##xgboost GPU parameter:
              ##NOte: Gpu is very fast, but it doesnot give good model performance like cpu, so tune your model with gpu and use cpu to generate final prediction on CPU 
  #             "tree_method":"gpu_hist",
              ## "gpu_id":1,
  #             "predictor":"gpu_predictor",
  from xgboost import XGBClassifier

  int_vals=['max_depth', 'reg_alpha']
  space={k: (int(val) if k in int_vals else val)    
          for k,val in space.items()}
  space['early_stopping_rounds']=early_stopping_rounds

  model=XGBClassifier(**space)
  y_model, _, df_epochs = ml_prediction(model,
                                              X,
                                              y,
                                              sk_fold,
                                              )
  
  #eval_metric used by early stopping comes from xgboost package and there is not same as metrics in sklearn:
    #https://xgboost.readthedocs.io/en/stable/parameter.html
    #https://scikit-learn.org/stable/modules/classes.html 
  #for instance use recall metric doesnot exists in eval_metric. Moreover same metric have difference name:  aucpr in eval_metric is same as average_precision_score in metrics
  ##so there is need to used dictionary to convert  eval_metric to metric  or add more items in metric_dict       
                                      
  scores=ml_scores(y_model, [Umetric],
                   multi_class='ovo',
                  average ='macro')
  scores_sub=scores.loc[scores['CV']=='CV_scores_Mean',Umetric]

  return {'loss':-scores_sub, 'status':STATUS_OK, 'model':model}

def xgb_tuner(X_train, y_train,
              X_test,  y_test ,
              random_state,
              metric=roc_auc_score,
              stepWise=True):
  import xgboost as xgb 
  from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

  from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
  params = {'random_state': random_state}
  rounds = [{'max_depth'      : hp.quniform('max_depth', 1, 8, 1), # tree
            'min_child_weight': hp.loguniform('min_child_weight', -2, 3)},

            {'subsample'      : hp.uniform('subsample', .5, 1), # stochastic
            'colsample_bytree': hp.uniform('colsample_bytree', .5, 1)},

            {'reg_alpha'      : hp.uniform('reg_alpha', 0, 10),
            'reg_lambda'     : hp.uniform('reg_lambda', 1, 10),},
            
            {'gamma'          : hp.loguniform('gamma', -10, 10)}, # regularization
            {'learning_rate'  : hp.loguniform('learning_rate', -7, 0)} # boosting
            ]
  if not stepWise:
    rounds = [{'max_depth'      : hp.quniform('max_depth', 1, 8, 1), # tree
            'min_child_weight': hp.loguniform('min_child_weight', -2, 3),

            'subsample'      : hp.uniform('subsample', .5, 1), # stochastic
            'colsample_bytree': hp.uniform('colsample_bytree', .5, 1),

            'reg_alpha'      : hp.uniform('reg_alpha', 0, 10),
            'reg_lambda'     : hp.uniform('reg_lambda', 1, 10),
            
            'gamma'          : hp.loguniform('gamma', -10, 10), # regularization
            'learning_rate'  : hp.loguniform('learning_rate', -7, 0)} # boosting
            ]
  for round in rounds:
    params = {**params, **round}
    trials = Trials()
    best = fmin(fn=lambda space: hyperparameter_tuning(space,
                                                        X_train,y_train,
                                                        X_test, y_test,
                                                        metric=metric),
                space=params,
                algo=tpe.suggest,
                max_evals=200 if stepWise else 1500,
                trials=trials,
                )
  params = {**params, **best}

  params['max_depth']=int(params['max_depth'])

  return params, trials
####------------------------------PCA Functions--------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.decomposition import PCA

def pca_explainedVar(pcaML, output_Folder=None, plot_type='cumulative'):
  """
  Calculate and plot the variance explained by each principal component and the cumulative variance explained.
  
  Parameters:
  ----------
  pcaML : PCA object
    Fitted PCA object from sklearn.
  outputFile : str, optional
    Path to save the output plots. If None, the plots are not saved. Default is None.
  plot_type : str, optional
    Type of plot to generate. Options are 'scree', 'cumulative', or 'both'. Default is 'both'.
  
  Returns:
  -------
  explained_var : np.ndarray
    Array of explained variance ratios for each principal component.
  eigen_values : np.ndarray
    Array of eigenvalues for each principal component.
  """
  import plotly.graph_objects as go
  import plotly.io as pio

  eigen_values = pcaML.explained_variance_
  explained_var = np.round(pcaML.explained_variance_ratio_, decimals=3) * 100
  explained_var_cumsum = np.cumsum(explained_var)

  if plot_type in ['scree', 'both']:
    # Scree plot
    scree_fig = go.Figure()
    scree_fig.add_trace(go.Scatter(x=np.arange(1, pcaML.n_components_ + 1), y=explained_var, mode='lines+markers', name='Explained Variance'))
    scree_fig.update_layout(title='Scree Plot', xaxis_title='Principal Component', yaxis_title='Variance Explained (%)')
    if output_Folder:
      pio.write_image(scree_fig, os.path.join(output_Folder, 'scree_plot.png')) 
    scree_fig.show()

  if plot_type in ['cumulative', 'both']:
    # Cumulative explained variance plot
    cumulative_fig = go.Figure()
    cumulative_fig.add_trace(go.Scatter(x=np.arange(1, pcaML.n_components_ + 1), y=explained_var_cumsum, mode='lines+markers', name='Cumulative Explained Variance'))
    cumulative_fig.update_layout(title='Cumulative Explained Variance Ratio by Principal Components', xaxis_title='Number of Principal Components', yaxis_title='Cumulative Explained Variance (%)', yaxis=dict(range=[0, 100]))
    if output_Folder:
      pio.write_image(cumulative_fig,os.path.join(output_Folder, '_cumulative_variance.png'))
    cumulative_fig.show()

  return explained_var, eigen_values

def pca_ortho_rotation(lam,
                   method  = 'varimax',
                   gamma   = None,
                   eps     = 1e-6,
                   itermax = 100
                   ):
    """
    ##TODO: document it 
    ## A VARIMAX rotation is a change of coordinates used in principal component analysis1 (PCA) that maximizes the sum of the variances of the squared loadings
    ## https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py
    Return orthogal rotation matrix
    TODO: - other types beyond 
    """
    if gamma == None:
        if (method == 'varimax'):
            gamma = 1.0
        if (method == 'quartimax'):
            gamma = 0.0

    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0

    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp     = np.diag(np.sum(lam_rot ** 2, axis = 0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R       = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):
            break
        var = var_new

    return R

def pca_important_features(transformed_features, components_, columns):
    import math
        ##TODO: check it and make a function
    ###http://benalexkeen.com/principle-component-analysis-in-python/    
    """
    This function will return the most "important" 
    features so we can determine which have the most
    effect on multi-dimensional scaling
    """
    num_columns = len(columns)

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    # Sort each column by it's length. These are your *original*
    # columns, not the principal components.
    important_features = { columns[i] : math.sqrt(xvector[i]**2 + yvector[i]**2) for i in range(num_columns) }
    # important_features = sorted(zip(important_features.values(), important_features.keys()), reverse=True)
    important_features = pd.Series(important_features)
    important_features = important_features.sort_values(ascending=[False])
    return important_features

def pc_draw_vectors(transformed_features, components_, columns):
    """
    This funtion will project your *original* features
    onto your principal component feature-space, so that you can
    visualize how "important" each one was in the
    multi-dimensional scaling

    https://benalexkeen.com/principle-component-analysis-in-python/
    """
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    num_columns = len(columns)

    # Scale the principal components by the max value in
    # the transformed set belonging to that component
    xvector = components_[0] * max(transformed_features[:,0])
    yvector = components_[1] * max(transformed_features[:,1])

    ax = plt.axes()

    for i in range(num_columns):
    # Use an arrow to project each original feature as a
    # labeled vector on your principal component axes
        plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.0005, head_width=0.02, alpha=0.75)
        plt.text(xvector[i]*1.2, yvector[i]*1.2, list(columns)[i], color='b', alpha=0.75)

    return ax
####------------------------------other Functions--------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##TODO: retire if not be used:
def class_weight2(uclass_weight,y):
    """ crearte a numerical series of samples' weights based on class_weight dictionary 
    Parameters:
    ----------
    class_weight (dictionary) or “balanced” or None, default=None
    Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.

    y (pandas dataframe/series) with [sample*1] format

    returns:
    -------
    df_class_weight  (pandas series) with [sample*1] format:
    weight of samples

    """ 
    ##y=y_train.cat.codes
    if uclass_weight == 'balanced':
        from sklearn.utils import class_weight
        tmp = np.round(class_weight.compute_class_weight(
            'balanced', np.unique(y.sort_values()), y), 2)
        class_weight_map = dict(zip(y.sort_values().unique().tolist(), tmp))
        df_class_weight = (y.map(class_weight_map))
    elif uclass_weight is None :
        df_class_weight = pd.Series(np.tile(1, y.size))
    else:
        df_class_weight = uclass_weight

    return df_class_weight


####------------------------------MultiLabel Classification--------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------
def unify_cols(df1, df2, df1_name, df2_name):
    """
    Unifies the columns of two DataFrames by ensuring both DataFrames have the same columns.
    If one DataFrame has columns that the other does not, those columns are added to the other DataFrame with default values of 0.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    df1_name (str): The name of the first DataFrame (used for printing messages).
    df2_name (str): The name of the second DataFrame (used for printing messages).

    Returns:
    tuple: A tuple containing the two DataFrames with unified columns (df1, df2).
    """
    df1.index=df2.index
    def unify_cols__sub(df1, df2, df1_name, df2_name):
        diff1=np.setdiff1d(df1.columns, df2.columns)
        if diff1.size!=0:
            print(f'Adding following columns to {df2_name} as there are in {df1_name}:\n {diff1}')
            df2=pd.concat([df2,
                  pd.DataFrame(0, index=df2.index, columns=diff1)], axis=1)
            df2=df2[df1.columns]
        return df2
    df2=unify_cols__sub(df1, df2, df1_name, df2_name)
    df1=unify_cols__sub(df2, df1, df2_name, df1_name)
    return df1, df2

def binarizer(tags):
  """
  Convert a list of tags into a binary matrix representation.

  Parameters:
  tags (list of list of str): A list where each element is a list of tags.

  Returns:
  pd.DataFrame: A DataFrame with binary values indicating the presence of tags.
          Each column represents a unique tag, and each row corresponds
          to the input list of tags.
  
  Example:
  >>> tags = [['tag1', 'tag2'], ['tag2', 'tag3'], ['tag1']]
  >>> binarizer(tags)
     tag1  tag2  tag3
  0     1     1     0
  1     0     1     1
  2     1     0     0
  """
  import pandas as pd
  from sklearn.preprocessing import MultiLabelBinarizer
  tags_seri = pd.Series(tags)
  mlb = MultiLabelBinarizer()
  out = pd.DataFrame(mlb.fit_transform(tags_seri),
                    columns=mlb.classes_,
    index=tags_seri.index)
  return out

def split_multiLabel_data__index(X, y, test_size, random_state=None):
    """Iteratively stratified train/test split

    Parameters
    ----------
    test_size : float, [0,1]
        the proportion of the dataset to include in the test split, the rest will be put in the train set

    random_state : None | int | np.random.RandomState
        the random state seed (optional)

    Returns
    -------
    X_train, y_train, X_test, y_test
        stratified division into train/test split
    """
    # from skmultilearn.model_selection import IterativeStratification
    # stratifier = IterativeStratification(
    #     n_splits=2,
    #     order=2,
    #     sample_distribution_per_fold=[test_size, 1.0 - test_size],
    #     random_state=random_state,
    #     # shuffle=True if random_state is not None else False
    # )

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    stratifier = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=random_state)    
    train_indexes, test_indexes = next(stratifier.split(X, y))

    return train_indexes,test_indexes

def split_multiLabel_data(df_samples2, binarized_tags, random_state=None):
        """
        Splits multi-label data into training, evaluation, and test sets.
        Parameters:
        df_samples2 (pd.DataFrame): DataFrame containing the samples to be split.
        binarized_tags (pd.DataFrame): DataFrame containing the binarized tags corresponding to the samples.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.
        Returns:
        tuple: A tuple containing:
          - pd.DataFrame: DataFrame with an additional 'Set' column indicating the split ('train', 'eval', 'test').
          - pd.DataFrame: DataFrame of binarized tags with an additional 'Set' column indicating the split.
        """
        binarized_tags_lst=binarized_tags.apply(lambda x: x.tolist(), axis=1)
        
        train_rows, evalNtest_rows= split_multiLabel_data__index(df_samples2.to_numpy(),
                                                                        np.array(binarized_tags_lst.tolist()),
                                                                        test_size=0.30,
                                                                        random_state=None
                                                                )
        df_samples__eval_test=df_samples2.iloc[evalNtest_rows]
        eval_rows, test_rows= split_multiLabel_data__index(df_samples__eval_test.to_numpy(),
                                                                np.array(binarized_tags_lst.iloc[evalNtest_rows].tolist()),
                                                                test_size=0.30, random_state=random_state)


        train_idx, eval_idx, test_idx=df_samples2.iloc[train_rows].index, df_samples__eval_test.iloc[eval_rows].index, df_samples__eval_test.iloc[test_rows].index

        df_samples2.loc[df_samples2.index.isin(train_idx), 'Set'] = 'train'
        df_samples2.loc[df_samples2.index.isin(eval_idx), 'Set'] = 'eval'
        df_samples2.loc[df_samples2.index.isin(test_idx), 'Set'] = 'test'
        binarized_tags['Set']=df_samples2['Set']

        return df_samples2, binarized_tags

def evaluate_multiLabel(y_pred, y_true,
                        average_op='binary',
                        scores_names=['recall' ,
                                    'precision',
                                    'accuracy',
                                    'auc_weighted',
                                    # 'balanced_accuracy',
                                    # 'roc_auc',  
                                    # 'aucpr',
                                    'f1',
                                    'kappa',
                                    'mcc',
                                    ]
                ):
    """
    Evaluate multi-label classification performance.
    This function evaluates the performance of a multi-label classification model by calculating various metrics such as recall, precision, accuracy, AUC, F1 score, kappa, and MCC. It also computes macro, micro, and weighted averages of these metrics.
    Parameters:
    -----------
    y_pred : pd.DataFrame
      Predicted labels for the multi-label classification.
    y_true : pd.DataFrame
      True labels for the multi-label classification.
    Returns:
    --------
    model_performance : dict
      A dictionary containing:
      - 'yScore': pd.DataFrame with calculated metrics for each label and overall averages.
      - 'accuracy_overall': float, accuracy score of selecting entire sets of tags.
    y_model : pd.DataFrame
      DataFrame containing the true and predicted labels along with the cross-validation iteration information.
    Notes:
    ------
    - The function uses micro, macro, and weighted averages for evaluation.
    - The overall accuracy is calculated using the subset accuracy method.
    - The function assumes that the input dataframes `y_pred` and `y_true` have the same structure and columns.
    """
    y_pred, y_true=unify_cols(y_pred, y_true, 'y_pred', 'y_true')


    y_model=pd.concat([y_true.melt(value_name='y_true').set_index('variable'),
                    y_pred.melt(value_name='y_pred').set_index('variable')],
                    axis=1).reset_index().rename(columns={'variable':'CV_Iteration'})

    
    tmp=y_model.groupby('CV_Iteration')[['y_pred', 'y_true']].sum().sum(axis=1)
    # print('Number of tags in each CV_Iteration:', tmp)
    y_model=y_model[y_model['CV_Iteration'].isin(tmp[tmp>0].index)]
    #plot_confusion_matrix2(y_model, map_lbls={0:'N',1:'Y'}, ncol=5)

    yScore=ml_scores(y_model, scores_names,
                  multi_class='ovo',
                  average =average_op
                  ).set_index("CV")
    yScore.index.name='Tag'

    map_dict={'CV_scores_Mean':'macro_avg',
            'CV_scores_STD':'macro_avg_STD',
            'scores_all':'micro_avg'
            }
    yScore=yScore.rename(index=map_dict)
    yScore['Support_number']=y_model.groupby('CV_Iteration')['y_true'].sum()

    # https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f
    # These results mean that in multi-class classification cases where each observation has a single label, the micro-F1, micro-precision, micro-recall, and accuracy share the same value
    # https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.classification_report.html
    # Micro average (averaging the total true positives, false negatives and false positives) is only shown for multi-label or multi-class with a subset of classes, because it corresponds to accuracy otherwise and would be the same for all metrics. 

    idx=yScore.index.str.contains("_avg")
    tmp=pd.DataFrame(yScore[~idx].apply(lambda x: np.average(x, weights=yScore.loc[~idx, 'Support_number']), axis=0), columns=['weighted_avg']).T
    yScore=pd.concat([yScore, tmp ], axis=0)

    idx=yScore.index.str.contains("_avg")
    yScore.loc[idx,'Support_number']=yScore.loc[~idx, 'Support_number'].sum()
    yScore['Support_number']=yScore['Support_number'].astype(int)

    yScore_labels=yScore[~idx].sort_values(by=['mcc'], ascending=False)
    yScore_overall=yScore[idx]
    yScore=pd.concat([yScore_labels, yScore_overall],axis=0)

    # """In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0."""
    # https://www.kaggle.com/code/kmkarakaya/multi-label-model-evaluation
    from sklearn.metrics import accuracy_score
    accuracy_overall=accuracy_score(y_true, y_pred)
    print('Accuracy Score of selecting entire sets of tags: ', round(accuracy_overall,4))
    model_performance={'yScore':yScore,
                    'accuracy_overall':accuracy_overall,
                    }

    return model_performance, y_model.rename({'CV_Iteration':'Class'},axis=1) 
