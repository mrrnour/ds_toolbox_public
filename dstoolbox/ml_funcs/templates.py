"""Pre-built classifier and regressor pipelines, ready for benchmarking."""

import numpy as np


##TODO: change y to y_ratio
def classifiers_template(y, random_state=10, imputer_strategy='mean' ,pc_n_components=5):
  """Build a dictionary of pre-configured sklearn classifier pipelines.

  Most classifiers are wrapped in a Pipeline of imputer → scaler (→ PCA
  for the ``*_pca`` and ``*DiscriminantAnalysis`` variants). XGBoost and
  LightGBM models are returned bare since they handle missing values and
  scaling internally. XGBoost weighted variants set ``scale_pos_weight``
  to ``count(y==0) / count(y==1)``.

  Parameters
  ----------
  y : array-like
      Target labels; only used to compute ``scale_pos_weight`` for
      class-imbalanced XGBoost variants.
  random_state : int, optional
      Seed propagated to every estimator that supports it. Default 10.
  imputer_strategy : str, optional
      Strategy for ``SimpleImputer``. Default ``'mean'``.
  pc_n_components : int, optional
      Number of PCA components for the PCA-based pipelines. Default 5.

  Returns
  -------
  dict of {str: sklearn.pipeline.Pipeline or sklearn estimator}
      Mapping of model names to ready-to-fit pipelines.
  """
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
        classifiers2[name]= Pipeline(steps=[("imputer", SimpleImputer(strategy=imputer_strategy)),
                                            ("scaler", StandardScaler()),
                                            (name, classifier)])
    else:
        classifiers2[name]=classifier
  return classifiers2


def regressors_template(random_state=10):
  """Build a dictionary of pre-configured sklearn regressor pipelines.

  Most regressors are wrapped in a Pipeline of robust scaler → median
  imputer (→ PCA for the ``*_pca``/``*DiscriminantAnalysis`` variants).
  XGBoost and LightGBM models are returned bare.

  Parameters
  ----------
  random_state : int, optional
      Seed propagated to every estimator that supports it. Default 10.

  Returns
  -------
  dict of {str: sklearn.pipeline.Pipeline or sklearn estimator}
      Mapping of model names to ready-to-fit pipelines.
  """
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
