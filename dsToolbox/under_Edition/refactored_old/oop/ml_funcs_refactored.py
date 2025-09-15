"""
Machine Learning Toolbox - Refactored Object-Oriented Implementation

A comprehensive machine learning utilities library providing classes for model performance
evaluation, training pipelines, visualization, hyperparameter optimization, and more.

This module refactors a large procedural codebase into a clean, maintainable object-oriented
architecture following SOLID principles and ML engineering best practices.

Author: Refactored by Claude Code
License: Same as original
"""

import pandas as pd
import numpy as np
import math
import os
import sys
import re
import warnings
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from abc import ABC, abstractmethod

# Core ML libraries
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold

# Optional dependencies with graceful fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pylab as pl
    from matplotlib.ticker import MultipleLocator
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Plotting libraries not available. Visualization features disabled.")

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from scipy import stats
    from scipy.stats import shapiro, jarque_bera, anderson, normaltest
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    SCIPY_STATSMODELS_AVAILABLE = True
except ImportError:
    SCIPY_STATSMODELS_AVAILABLE = False


# ============================================================================
# EXCEPTION HIERARCHY
# ============================================================================

class MLToolboxError(Exception):
    """Base exception for ML Toolbox errors."""
    pass

class ModelTrainingError(MLToolboxError):
    """Exception raised during model training operations."""
    pass

class ValidationError(MLToolboxError):
    """Exception raised during data validation."""
    pass

class DataProcessingError(MLToolboxError):
    """Exception raised during data processing operations."""
    pass

class VisualizationError(MLToolboxError):
    """Exception raised during visualization operations."""
    pass

class HyperparameterTuningError(MLToolboxError):
    """Exception raised during hyperparameter tuning."""
    pass


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class ModelTrainingConfiguration:
    """Configuration parameters for model training operations."""
    random_state: int = 10
    cross_validation_folds: int = 5
    test_size: float = 0.2
    imputer_strategy: str = 'mean'
    pca_n_components: int = 5
    scoring_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'f1', 'precision', 'recall'])
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not isinstance(self.random_state, int) or self.random_state < 0:
            raise ValidationError("random_state must be a non-negative integer")
        if not isinstance(self.cross_validation_folds, int) or self.cross_validation_folds < 2:
            raise ValidationError("cross_validation_folds must be an integer >= 2")
        if not (0 < self.test_size < 1):
            raise ValidationError("test_size must be between 0 and 1")


@dataclass
class VisualizationConfiguration:
    """Configuration parameters for visualization operations."""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = 'default'
    color_palette: str = 'viridis'
    save_format: str = 'png'
    
    def __post_init__(self):
        """Validate visualization parameters."""
        if not PLOTTING_AVAILABLE:
            warnings.warn("Plotting libraries not available. Visualization disabled.")


@dataclass
class HyperparameterTuningConfiguration:
    """Configuration parameters for hyperparameter optimization."""
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    random_state: int = 10
    optimization_direction: str = 'maximize'
    
    def __post_init__(self):
        """Validate hyperparameter tuning parameters."""
        if not OPTUNA_AVAILABLE:
            raise HyperparameterTuningError("Optuna not available for hyperparameter tuning")


# ============================================================================
# MODEL PERFORMANCE ANALYZER
# ============================================================================

class ModelPerformanceAnalyzer:
    """
    Comprehensive model performance evaluation and scoring utilities.
    
    This class provides methods for evaluating machine learning models using various
    metrics, cross-validation strategies, and performance analysis techniques.
    
    Attributes:
        metric_registry (Dict): Registry of available evaluation metrics
        config (ModelTrainingConfiguration): Configuration for training operations
    
    Example:
        >>> analyzer = ModelPerformanceAnalyzer()
        >>> scores = analyzer.calculate_scores(y_true, y_pred, ['accuracy', 'f1'])
        >>> print(f"Accuracy: {scores['accuracy']:.3f}")
    """
    
    def __init__(self, config: Optional[ModelTrainingConfiguration] = None):
        """
        Initialize the model performance analyzer.
        
        Args:
            config: Configuration object for model training parameters
            
        Raises:
            ValidationError: If configuration parameters are invalid
        """
        self.config = config or ModelTrainingConfiguration()
        self.metric_registry = self._build_metric_registry()
        self.logger = logging.getLogger(__name__)
        
    def _build_metric_registry(self) -> Dict[str, Callable]:
        """
        Build registry of available evaluation metrics.
        
        Returns:
            Dictionary mapping metric names to sklearn metric functions
        """
        return {
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
            'mcc': metrics.matthews_corrcoef,
            'kappa': metrics.cohen_kappa_score,
            
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
        }
    
    def calculate_scores(self, 
                        y_true: np.ndarray, 
                        y_predicted: np.ndarray,
                        metric_names: List[str],
                        **metric_kwargs) -> Dict[str, float]:
        """
        Calculate multiple evaluation metrics for model predictions.
        
        Args:
            y_true: Ground truth labels
            y_predicted: Predicted labels or probabilities
            metric_names: List of metric names to calculate
            **metric_kwargs: Additional keyword arguments for specific metrics
            
        Returns:
            Dictionary mapping metric names to calculated scores
            
        Raises:
            ValidationError: If metric names are invalid or data shapes don't match
            
        Example:
            >>> y_true = [0, 1, 1, 0, 1]
            >>> y_pred = [0, 1, 0, 0, 1] 
            >>> scores = analyzer.calculate_scores(y_true, y_pred, ['accuracy', 'f1'])
        """
        if len(y_true) != len(y_predicted):
            raise ValidationError("y_true and y_predicted must have the same length")
            
        results = {}
        
        for metric_name in metric_names:
            if metric_name not in self.metric_registry:
                self.logger.warning(f"Unknown metric: {metric_name}. Skipping.")
                continue
                
            try:
                metric_func = self.metric_registry[metric_name]
                
                # Handle special cases for certain metrics
                if metric_name in ['f1', 'precision', 'recall', 'jaccard']:
                    # For multiclass, use macro average by default
                    if 'average' not in metric_kwargs:
                        metric_kwargs['average'] = 'macro'
                
                score = metric_func(y_true, y_predicted, **metric_kwargs)
                results[metric_name] = score
                
            except Exception as e:
                self.logger.error(f"Error calculating {metric_name}: {str(e)}")
                results[metric_name] = np.nan
                
        return results
    
    def cross_validate_model(self,
                           estimator,
                           training_features: np.ndarray,
                           target_values: np.ndarray,
                           scoring_metrics: Optional[List[str]] = None,
                           cv_folds: Optional[int] = None,
                           return_train_score: bool = False) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation of a model.
        
        Args:
            estimator: Sklearn-compatible estimator
            training_features: Feature matrix for training
            target_values: Target variable
            scoring_metrics: List of scoring metrics to use
            cv_folds: Number of cross-validation folds
            return_train_score: Whether to return training scores
            
        Returns:
            Dictionary containing cross-validation results
            
        Raises:
            ModelTrainingError: If cross-validation fails
            
        Example:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> model = RandomForestClassifier()
            >>> results = analyzer.cross_validate_model(model, X, y, ['accuracy', 'f1'])
        """
        scoring_metrics = scoring_metrics or self.config.scoring_metrics
        cv_folds = cv_folds or self.config.cross_validation_folds
        
        # Create cross-validation splitter
        if self._is_classification_task(target_values):
            cv_splitter = StratifiedKFold(n_splits=cv_folds, 
                                        shuffle=True, 
                                        random_state=self.config.random_state)
        else:
            cv_splitter = KFold(n_splits=cv_folds,
                              shuffle=True,
                              random_state=self.config.random_state)
        
        try:
            # Prepare scoring dictionary
            scoring_dict = {}
            for metric_name in scoring_metrics:
                if metric_name in self.metric_registry:
                    scoring_dict[metric_name] = metrics.make_scorer(
                        self.metric_registry[metric_name],
                        **self._get_metric_kwargs(metric_name)
                    )
            
            # Perform cross-validation
            cv_results = cross_validate(
                estimator=estimator,
                X=training_features,
                y=target_values,
                scoring=scoring_dict,
                cv=cv_splitter,
                return_train_score=return_train_score,
                n_jobs=-1
            )
            
            return cv_results
            
        except Exception as e:
            raise ModelTrainingError(f"Cross-validation failed: {str(e)}")
    
    def _is_classification_task(self, target_values: np.ndarray) -> bool:
        """
        Determine if the task is classification based on target values.
        
        Args:
            target_values: Target variable array
            
        Returns:
            True if classification task, False for regression
        """
        unique_values = np.unique(target_values)
        
        # Check if values are integers and reasonable number of unique values
        if len(unique_values) <= 20 and np.all(target_values == target_values.astype(int)):
            return True
        return False
    
    def _get_metric_kwargs(self, metric_name: str) -> Dict[str, Any]:
        """
        Get appropriate keyword arguments for specific metrics.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary of keyword arguments for the metric
        """
        if metric_name in ['f1', 'precision', 'recall', 'jaccard']:
            return {'average': 'macro', 'zero_division': 0}
        elif metric_name == 'roc_auc':
            return {'multi_class': 'ovr', 'average': 'macro'}
        return {}


# ============================================================================
# MODEL TEMPLATE FACTORY
# ============================================================================

class ModelTemplateFactory:
    """
    Factory class for creating pre-configured machine learning model templates.
    
    This class implements the Factory pattern to create standardized model pipelines
    with preprocessing steps, feature selection, and model-specific configurations.
    
    Example:
        >>> factory = ModelTemplateFactory()
        >>> classifiers = factory.create_classifier_templates()
        >>> regressors = factory.create_regressor_templates()
    """
    
    def __init__(self, config: Optional[ModelTrainingConfiguration] = None):
        """
        Initialize the model template factory.
        
        Args:
            config: Configuration object for model parameters
        """
        self.config = config or ModelTrainingConfiguration()
        self.logger = logging.getLogger(__name__)
    
    def create_classifier_templates(self, 
                                  target_distribution: Optional[np.ndarray] = None) -> Dict[str, Pipeline]:
        """
        Create pre-configured classifier templates with preprocessing pipelines.
        
        Args:
            target_distribution: Distribution of target variable for balancing
            
        Returns:
            Dictionary mapping classifier names to configured Pipeline objects
            
        Raises:
            ModelTrainingError: If classifier creation fails
            
        Example:
            >>> templates = factory.create_classifier_templates()
            >>> rf_model = templates['Random_Forest']
            >>> rf_model.fit(X_train, y_train)
        """
        try:
            # Import classifiers
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.svm import SVC, LinearSVC
            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
            from sklearn.neural_network import MLPClassifier
            from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
            
            # Base preprocessing pipeline
            preprocessing_steps = [
                ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=self.config.pca_n_components))
            ]
            
            classifiers = {
                "Logistic_Regression": Pipeline([
                    *preprocessing_steps,
                    ('classifier', LogisticRegression(
                        random_state=self.config.random_state,
                        max_iter=1000
                    ))
                ]),
                
                "Random_Forest": Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('classifier', RandomForestClassifier(
                        random_state=self.config.random_state,
                        n_estimators=100
                    ))
                ]),
                
                "Support_Vector_Machine": Pipeline([
                    *preprocessing_steps,
                    ('classifier', SVC(
                        random_state=self.config.random_state,
                        probability=True
                    ))
                ]),
                
                "K_Nearest_Neighbors_3": Pipeline([
                    *preprocessing_steps,
                    ('classifier', KNeighborsClassifier(n_neighbors=3))
                ]),
                
                "K_Nearest_Neighbors_5": Pipeline([
                    *preprocessing_steps,
                    ('classifier', KNeighborsClassifier(n_neighbors=5))
                ]),
                
                "Decision_Tree": Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('classifier', DecisionTreeClassifier(
                        random_state=self.config.random_state
                    ))
                ]),
                
                "Naive_Bayes": Pipeline([
                    *preprocessing_steps,
                    ('classifier', GaussianNB())
                ]),
                
                "Linear_Discriminant_Analysis": Pipeline([
                    *preprocessing_steps,
                    ('classifier', LinearDiscriminantAnalysis())
                ]),
                
                "Quadratic_Discriminant_Analysis": Pipeline([
                    *preprocessing_steps,
                    ('classifier', QuadraticDiscriminantAnalysis())
                ]),
                
                "Neural_Network": Pipeline([
                    *preprocessing_steps,
                    ('classifier', MLPClassifier(
                        random_state=self.config.random_state,
                        max_iter=500
                    ))
                ]),
                
                "AdaBoost": Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('classifier', AdaBoostClassifier(
                        random_state=self.config.random_state
                    ))
                ]),
                
                "Gradient_Boosting": Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('classifier', GradientBoostingClassifier(
                        random_state=self.config.random_state
                    ))
                ])
            }
            
            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                classifiers["XGBoost"] = Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('classifier', XGBClassifier(
                        random_state=self.config.random_state,
                        eval_metric='logloss'
                    ))
                ])
            
            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                classifiers["LightGBM"] = Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('classifier', LGBMClassifier(
                        random_state=self.config.random_state,
                        verbose=-1
                    ))
                ])
            
            return classifiers
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to create classifier templates: {str(e)}")
    
    def create_regressor_templates(self) -> Dict[str, Pipeline]:
        """
        Create pre-configured regressor templates with preprocessing pipelines.
        
        Returns:
            Dictionary mapping regressor names to configured Pipeline objects
            
        Raises:
            ModelTrainingError: If regressor creation fails
            
        Example:
            >>> templates = factory.create_regressor_templates()
            >>> rf_model = templates['Random_Forest_Regressor']
            >>> rf_model.fit(X_train, y_train)
        """
        try:
            # Import regressors
            from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.svm import SVR
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
            from sklearn.neural_network import MLPRegressor
            
            # Base preprocessing pipeline
            preprocessing_steps = [
                ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                ('scaler', StandardScaler())
            ]
            
            regressors = {
                "Linear_Regression": Pipeline([
                    *preprocessing_steps,
                    ('regressor', LinearRegression())
                ]),
                
                "Ridge_Regression": Pipeline([
                    *preprocessing_steps,
                    ('regressor', Ridge(
                        alpha=1.0,
                        random_state=self.config.random_state
                    ))
                ]),
                
                "Lasso_Regression": Pipeline([
                    *preprocessing_steps,
                    ('regressor', Lasso(
                        alpha=1.0,
                        random_state=self.config.random_state,
                        max_iter=1000
                    ))
                ]),
                
                "ElasticNet_Regression": Pipeline([
                    *preprocessing_steps,
                    ('regressor', ElasticNet(
                        alpha=1.0,
                        random_state=self.config.random_state,
                        max_iter=1000
                    ))
                ]),
                
                "Random_Forest_Regressor": Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('regressor', RandomForestRegressor(
                        random_state=self.config.random_state,
                        n_estimators=100
                    ))
                ]),
                
                "Support_Vector_Regressor": Pipeline([
                    *preprocessing_steps,
                    ('regressor', SVR())
                ]),
                
                "K_Nearest_Neighbors_Regressor": Pipeline([
                    *preprocessing_steps,
                    ('regressor', KNeighborsRegressor(n_neighbors=5))
                ]),
                
                "Decision_Tree_Regressor": Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('regressor', DecisionTreeRegressor(
                        random_state=self.config.random_state
                    ))
                ]),
                
                "Neural_Network_Regressor": Pipeline([
                    *preprocessing_steps,
                    ('regressor', MLPRegressor(
                        random_state=self.config.random_state,
                        max_iter=500
                    ))
                ]),
                
                "AdaBoost_Regressor": Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('regressor', AdaBoostRegressor(
                        random_state=self.config.random_state
                    ))
                ]),
                
                "Gradient_Boosting_Regressor": Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('regressor', GradientBoostingRegressor(
                        random_state=self.config.random_state
                    ))
                ])
            }
            
            # Add XGBoost if available
            if XGBOOST_AVAILABLE:
                regressors["XGBoost_Regressor"] = Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('regressor', XGBRegressor(
                        random_state=self.config.random_state
                    ))
                ])
            
            # Add LightGBM if available
            if LIGHTGBM_AVAILABLE:
                regressors["LightGBM_Regressor"] = Pipeline([
                    ('imputer', SimpleImputer(strategy=self.config.imputer_strategy)),
                    ('regressor', LGBMRegressor(
                        random_state=self.config.random_state,
                        verbose=-1
                    ))
                ])
            
            return regressors
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to create regressor templates: {str(e)}")


# ============================================================================
# LINEAR REGRESSION ASSUMPTIONS CHECKER
# ============================================================================

class LinearRegressionAssumptionsChecker:
    """
    A comprehensive class to check assumptions of Linear Regression Models including
    OLS, Ridge, and Lasso based on statistical best practices.
    
    Assumptions checked:
    1. Linear relationship between features and target
    2. Independence of residuals (No Autocorrelation)  
    3. Multicollinearity (adapted for regularized models)
    4. Homoscedasticity (Equal Variance)
    5. Normality of residuals
    6. No influential outliers
    
    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> model = LinearRegression()
        >>> model.fit(X_train, y_train)
        >>> checker = LinearRegressionAssumptionsChecker(model, X_train, y_train)
        >>> results = checker.run_full_diagnostic()
    """
    
    def __init__(self, 
                 fitted_model, 
                 training_features: np.ndarray, 
                 target_values: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 model_type: str = 'ols'):
        """
        Initialize the assumptions checker with fitted model and data.
        
        Args:
            fitted_model: Fitted regression model (sklearn or statsmodels)
            training_features: Feature matrix used for training
            target_values: Target variable values
            feature_names: Names of features (optional)
            model_type: Type of model ('ols', 'ridge', 'lasso')
            
        Raises:
            ValidationError: If inputs are invalid
        """
        if not SCIPY_STATSMODELS_AVAILABLE:
            raise ValidationError("SciPy and StatsModels required for assumptions checking")
            
        self.fitted_model = fitted_model
        self.training_features = np.array(training_features)
        self.target_values = np.array(target_values)
        self.feature_names = feature_names or [f'Feature_{i}' for i in range(self.training_features.shape[1])]
        self.model_type = model_type.lower()
        self.logger = logging.getLogger(__name__)
        
        # Validate inputs
        if len(self.training_features) != len(self.target_values):
            raise ValidationError("training_features and target_values must have same length")
        
        # Get predictions and residuals
        try:
            self.predicted_values = fitted_model.predict(self.training_features)
            self.residuals = self.target_values - self.predicted_values
            self.standardized_residuals = self.residuals / np.std(self.residuals)
        except Exception as e:
            raise ValidationError(f"Error computing residuals: {str(e)}")
        
        # For statsmodels compatibility
        if hasattr(fitted_model, 'resid'):
            self.residuals = fitted_model.resid
            self.predicted_values = fitted_model.fittedvalues
    
    def check_linearity(self, create_plot: bool = True) -> Dict[str, Any]:
        """
        Check assumption 1: Linear relationship between features and target.
        
        Uses residuals vs fitted values plot to detect non-linear patterns.
        
        Args:
            create_plot: Whether to create diagnostic plots
            
        Returns:
            Dictionary containing linearity test results
        """
        results = {'assumption': 'linearity', 'passed': True, 'warnings': []}
        
        if create_plot and PLOTTING_AVAILABLE:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Residuals vs Fitted Plot
                axes[0].scatter(self.predicted_values, self.residuals, alpha=0.7, color='blue')
                axes[0].axhline(y=0, color='red', linestyle='--')
                axes[0].set_xlabel('Fitted Values')
                axes[0].set_ylabel('Residuals')
                axes[0].set_title('Residuals vs Fitted Values\n(Check for Linearity)')
                
                # Add trend line
                z = np.polyfit(self.predicted_values, self.residuals, 2)
                p = np.poly1d(z)
                x_smooth = np.linspace(self.predicted_values.min(), self.predicted_values.max(), 100)
                axes[0].plot(x_smooth, p(x_smooth), color='red', linewidth=2)
                
                # Actual vs Predicted Plot
                axes[1].scatter(self.target_values, self.predicted_values, alpha=0.7, color='green')
                min_val = min(self.target_values.min(), self.predicted_values.min())
                max_val = max(self.target_values.max(), self.predicted_values.max())
                axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                axes[1].set_xlabel('Actual Values')
                axes[1].set_ylabel('Predicted Values')
                axes[1].set_title('Actual vs Predicted Values')
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                results['warnings'].append(f"Could not create linearity plots: {str(e)}")
        
        # Statistical test for non-linearity
        try:
            correlation_coef = np.corrcoef(self.predicted_values, self.residuals)[0, 1]
            if abs(correlation_coef) > 0.1:
                results['passed'] = False
                results['warnings'].append(f"High correlation between fitted values and residuals: {correlation_coef:.3f}")
        except Exception as e:
            results['warnings'].append(f"Could not compute correlation: {str(e)}")
        
        results['correlation_fitted_residuals'] = correlation_coef if 'correlation_coef' in locals() else None
        return results
    
    def check_independence(self) -> Dict[str, Any]:
        """
        Check assumption 2: Independence of residuals (No Autocorrelation).
        
        Uses Durbin-Watson test to detect autocorrelation in residuals.
        
        Returns:
            Dictionary containing independence test results
        """
        results = {'assumption': 'independence', 'passed': True, 'warnings': []}
        
        try:
            # Durbin-Watson test
            dw_statistic = durbin_watson(self.residuals)
            results['durbin_watson_statistic'] = dw_statistic
            
            # DW statistic interpretation:
            # ~2.0: No autocorrelation
            # <1.5: Positive autocorrelation
            # >2.5: Negative autocorrelation
            if dw_statistic < 1.5:
                results['passed'] = False
                results['warnings'].append(f"Positive autocorrelation detected (DW={dw_statistic:.3f})")
            elif dw_statistic > 2.5:
                results['passed'] = False
                results['warnings'].append(f"Negative autocorrelation detected (DW={dw_statistic:.3f})")
                
        except Exception as e:
            results['warnings'].append(f"Could not perform Durbin-Watson test: {str(e)}")
            results['durbin_watson_statistic'] = None
        
        return results
    
    def check_multicollinearity(self) -> Dict[str, Any]:
        """
        Check assumption 3: Multicollinearity among features.
        
        Uses Variance Inflation Factor (VIF) to detect multicollinearity.
        Note: For Ridge/Lasso, high multicollinearity is less problematic.
        
        Returns:
            Dictionary containing multicollinearity test results
        """
        results = {'assumption': 'multicollinearity', 'passed': True, 'warnings': []}
        
        try:
            # Calculate VIF for each feature
            vif_scores = {}
            for i, feature_name in enumerate(self.feature_names):
                try:
                    vif_value = variance_inflation_factor(self.training_features, i)
                    vif_scores[feature_name] = vif_value
                    
                    # VIF > 10 indicates high multicollinearity
                    if vif_value > 10:
                        if self.model_type == 'ols':
                            results['passed'] = False
                            results['warnings'].append(f"High VIF for {feature_name}: {vif_value:.2f}")
                        else:
                            results['warnings'].append(f"High VIF for {feature_name}: {vif_value:.2f} (acceptable for {self.model_type.upper()})")
                            
                except Exception:
                    vif_scores[feature_name] = np.inf
                    results['warnings'].append(f"Could not calculate VIF for {feature_name}")
            
            results['vif_scores'] = vif_scores
            
        except Exception as e:
            results['warnings'].append(f"Could not perform multicollinearity check: {str(e)}")
            results['vif_scores'] = None
        
        return results
    
    def check_homoscedasticity(self, create_plot: bool = True) -> Dict[str, Any]:
        """
        Check assumption 4: Homoscedasticity (Equal Variance of residuals).
        
        Uses Breusch-Pagan and White tests for heteroscedasticity.
        
        Args:
            create_plot: Whether to create diagnostic plots
            
        Returns:
            Dictionary containing homoscedasticity test results
        """
        results = {'assumption': 'homoscedasticity', 'passed': True, 'warnings': []}
        
        if create_plot and PLOTTING_AVAILABLE:
            try:
                plt.figure(figsize=(10, 6))
                plt.scatter(self.predicted_values, np.abs(self.standardized_residuals), alpha=0.7)
                plt.xlabel('Fitted Values')
                plt.ylabel('|Standardized Residuals|')
                plt.title('Scale-Location Plot\n(Check for Homoscedasticity)')
                
                # Add trend line
                z = np.polyfit(self.predicted_values, np.abs(self.standardized_residuals), 1)
                p = np.poly1d(z)
                plt.plot(self.predicted_values, p(self.predicted_values), 'r-', linewidth=2)
                plt.show()
                
            except Exception as e:
                results['warnings'].append(f"Could not create homoscedasticity plot: {str(e)}")
        
        try:
            # Breusch-Pagan test
            bp_statistic, bp_pvalue, bp_fstatistic, bp_f_pvalue = het_breuschpagan(
                self.residuals, self.training_features
            )
            results['breusch_pagan_pvalue'] = bp_pvalue
            
            if bp_pvalue < 0.05:
                results['passed'] = False
                results['warnings'].append(f"Breusch-Pagan test indicates heteroscedasticity (p={bp_pvalue:.4f})")
            
            # White test
            white_statistic, white_pvalue, white_fstatistic, white_f_pvalue = het_white(
                self.residuals, self.training_features
            )
            results['white_test_pvalue'] = white_pvalue
            
            if white_pvalue < 0.05:
                results['passed'] = False
                results['warnings'].append(f"White test indicates heteroscedasticity (p={white_pvalue:.4f})")
                
        except Exception as e:
            results['warnings'].append(f"Could not perform heteroscedasticity tests: {str(e)}")
        
        return results
    
    def check_normality(self, create_plot: bool = True) -> Dict[str, Any]:
        """
        Check assumption 5: Normality of residuals.
        
        Uses multiple normality tests: Shapiro-Wilk, Jarque-Bera, Anderson-Darling.
        
        Args:
            create_plot: Whether to create diagnostic plots
            
        Returns:
            Dictionary containing normality test results
        """
        results = {'assumption': 'normality', 'passed': True, 'warnings': []}
        
        if create_plot and PLOTTING_AVAILABLE:
            try:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Q-Q Plot
                stats.probplot(self.residuals, dist="norm", plot=axes[0])
                axes[0].set_title('Q-Q Plot of Residuals')
                
                # Histogram of residuals
                axes[1].hist(self.residuals, bins=30, density=True, alpha=0.7, color='blue')
                
                # Overlay normal distribution
                mu, sigma = np.mean(self.residuals), np.std(self.residuals)
                x = np.linspace(self.residuals.min(), self.residuals.max(), 100)
                axes[1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
                axes[1].set_xlabel('Residuals')
                axes[1].set_ylabel('Density')
                axes[1].set_title('Distribution of Residuals')
                axes[1].legend()
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                results['warnings'].append(f"Could not create normality plots: {str(e)}")
        
        # Statistical tests
        try:
            # Shapiro-Wilk test (good for small samples)
            if len(self.residuals) <= 5000:
                shapiro_stat, shapiro_pvalue = shapiro(self.residuals)
                results['shapiro_wilk_pvalue'] = shapiro_pvalue
                
                if shapiro_pvalue < 0.05:
                    results['passed'] = False
                    results['warnings'].append(f"Shapiro-Wilk test rejects normality (p={shapiro_pvalue:.4f})")
            
            # Jarque-Bera test
            jb_stat, jb_pvalue = jarque_bera(self.residuals)
            results['jarque_bera_pvalue'] = jb_pvalue
            
            if jb_pvalue < 0.05:
                results['passed'] = False
                results['warnings'].append(f"Jarque-Bera test rejects normality (p={jb_pvalue:.4f})")
            
            # Anderson-Darling test
            ad_stat, ad_critical_values, ad_significance_levels = anderson(self.residuals, dist='norm')
            results['anderson_darling_statistic'] = ad_stat
            
            # Check at 5% significance level
            if ad_stat > ad_critical_values[2]:  # 5% level is typically index 2
                results['passed'] = False
                results['warnings'].append(f"Anderson-Darling test rejects normality (stat={ad_stat:.4f})")
                
        except Exception as e:
            results['warnings'].append(f"Could not perform normality tests: {str(e)}")
        
        return results
    
    def check_outliers_influence(self, create_plot: bool = True) -> Dict[str, Any]:
        """
        Check assumption 6: No influential outliers.
        
        Identifies outliers using standardized residuals and leverage values.
        
        Args:
            create_plot: Whether to create diagnostic plots
            
        Returns:
            Dictionary containing outlier detection results
        """
        results = {'assumption': 'no_influential_outliers', 'passed': True, 'warnings': []}
        
        try:
            # Identify outliers using standardized residuals
            outlier_threshold = 3
            outlier_indices = np.where(np.abs(self.standardized_residuals) > outlier_threshold)[0]
            
            results['outlier_indices'] = outlier_indices.tolist()
            results['n_outliers'] = len(outlier_indices)
            
            if len(outlier_indices) > 0:
                outlier_percentage = (len(outlier_indices) / len(self.residuals)) * 100
                results['outlier_percentage'] = outlier_percentage
                
                if outlier_percentage > 5:  # More than 5% outliers
                    results['passed'] = False
                    results['warnings'].append(f"High percentage of outliers detected: {outlier_percentage:.1f}%")
                else:
                    results['warnings'].append(f"Some outliers detected: {len(outlier_indices)} ({outlier_percentage:.1f}%)")
            
            if create_plot and PLOTTING_AVAILABLE:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(range(len(self.standardized_residuals)), self.standardized_residuals, alpha=0.7)
                    plt.axhline(y=outlier_threshold, color='red', linestyle='--', label=f'Outlier threshold (+{outlier_threshold})')
                    plt.axhline(y=-outlier_threshold, color='red', linestyle='--', label=f'Outlier threshold (-{outlier_threshold})')
                    
                    # Highlight outliers
                    if len(outlier_indices) > 0:
                        plt.scatter(outlier_indices, self.standardized_residuals[outlier_indices], 
                                  color='red', s=100, alpha=0.8, label='Outliers')
                    
                    plt.xlabel('Observation Index')
                    plt.ylabel('Standardized Residuals')
                    plt.title('Standardized Residuals Plot\n(Outlier Detection)')
                    plt.legend()
                    plt.show()
                    
                except Exception as e:
                    results['warnings'].append(f"Could not create outlier plot: {str(e)}")
        
        except Exception as e:
            results['warnings'].append(f"Could not perform outlier detection: {str(e)}")
        
        return results
    
    def run_full_diagnostic(self, create_plots: bool = True) -> Dict[str, Any]:
        """
        Run complete regression assumptions diagnostic.
        
        Performs all assumption checks and provides comprehensive results.
        
        Args:
            create_plots: Whether to create diagnostic plots
            
        Returns:
            Dictionary containing all diagnostic results
            
        Example:
            >>> checker = LinearRegressionAssumptionsChecker(model, X, y)
            >>> results = checker.run_full_diagnostic()
            >>> print(f"Overall passed: {results['overall_passed']}")
        """
        self.logger.info("Running full regression assumptions diagnostic...")
        
        diagnostic_results = {
            'model_type': self.model_type,
            'n_observations': len(self.target_values),
            'n_features': self.training_features.shape[1],
            'assumption_results': {},
            'overall_passed': True,
            'summary_warnings': []
        }
        
        # Run all assumption checks
        checks = [
            ('linearity', self.check_linearity),
            ('independence', self.check_independence),
            ('multicollinearity', self.check_multicollinearity),
            ('homoscedasticity', self.check_homoscedasticity),
            ('normality', self.check_normality),
            ('outliers', self.check_outliers_influence)
        ]
        
        for check_name, check_method in checks:
            try:
                if check_name in ['linearity', 'homoscedasticity', 'normality', 'outliers']:
                    result = check_method(create_plot=create_plots)
                else:
                    result = check_method()
                
                diagnostic_results['assumption_results'][check_name] = result
                
                if not result['passed']:
                    diagnostic_results['overall_passed'] = False
                    diagnostic_results['summary_warnings'].extend(result['warnings'])
                    
            except Exception as e:
                self.logger.error(f"Error in {check_name} check: {str(e)}")
                diagnostic_results['assumption_results'][check_name] = {
                    'assumption': check_name,
                    'passed': False,
                    'warnings': [f"Check failed: {str(e)}"]
                }
                diagnostic_results['overall_passed'] = False
        
        # Print summary
        self.logger.info(f"Diagnostic complete. Overall passed: {diagnostic_results['overall_passed']}")
        if diagnostic_results['summary_warnings']:
            self.logger.warning("Issues found:")
            for warning in diagnostic_results['summary_warnings']:
                self.logger.warning(f"  - {warning}")
        
        return diagnostic_results


# ============================================================================
# ML VISUALIZATION ENGINE
# ============================================================================

class MLVisualizationEngine:
    """
    Comprehensive visualization engine for machine learning model evaluation and analysis.
    
    Provides methods for creating various ML-specific plots including performance curves,
    confusion matrices, feature importance plots, and diagnostic visualizations.
    
    Example:
        >>> viz = MLVisualizationEngine()
        >>> viz.plot_roc_curve(y_true, y_prob, save_path='roc_curve.png')
        >>> viz.plot_confusion_matrix(y_true, y_pred, class_names=['A', 'B', 'C'])
    """
    
    def __init__(self, config: Optional[VisualizationConfiguration] = None):
        """
        Initialize the visualization engine.
        
        Args:
            config: Configuration object for visualization parameters
        """
        self.config = config or VisualizationConfiguration()
        self.logger = logging.getLogger(__name__)
        
        if not PLOTTING_AVAILABLE:
            raise VisualizationError("Plotting libraries not available. Install matplotlib and seaborn.")
        
        # Set default style
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
    
    def plot_model_comparison(self, 
                            performance_metrics: pd.DataFrame,
                            output_file_path: Optional[str] = None,
                            metric_to_highlight: str = 'accuracy') -> None:
        """
        Create comprehensive model comparison visualization.
        
        Args:
            performance_metrics: DataFrame with models as index, metrics as columns
            output_file_path: Path to save the plot (optional)
            metric_to_highlight: Primary metric for sorting models
            
        Raises:
            VisualizationError: If plotting fails
            
        Example:
            >>> metrics_df = pd.DataFrame({'accuracy': [0.85, 0.82], 'f1': [0.83, 0.80]})
            >>> viz.plot_model_comparison(metrics_df, 'comparison.png')
        """
        try:
            if performance_metrics.empty:
                raise VisualizationError("performance_metrics DataFrame is empty")
            
            # Sort by primary metric
            if metric_to_highlight in performance_metrics.columns:
                performance_metrics = performance_metrics.sort_values(
                    by=metric_to_highlight, ascending=False
                )
            
            n_metrics = len(performance_metrics.columns)
            n_models = len(performance_metrics.index)
            
            # Create subplots
            fig, axes = plt.subplots(1, min(n_metrics, 4), 
                                   figsize=(min(n_metrics * 4, 20), max(6, n_models * 0.3)))
            if n_metrics == 1:
                axes = [axes]
            
            # Plot each metric
            for idx, metric_name in enumerate(performance_metrics.columns[:4]):
                ax = axes[idx] if idx < len(axes) else axes[-1]
                
                metric_values = performance_metrics[metric_name]
                colors = plt.cm.viridis(np.linspace(0, 1, len(metric_values)))
                
                bars = ax.barh(range(len(metric_values)), metric_values, color=colors)
                ax.set_yticks(range(len(metric_values)))
                ax.set_yticklabels(performance_metrics.index, fontsize=10)
                ax.set_xlabel(metric_name.capitalize(), fontsize=12)
                ax.set_title(f'Model Comparison - {metric_name.capitalize()}', fontsize=14)
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, metric_values)):
                    ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{value:.3f}', ha='left', va='center', fontsize=9)
                
                ax.set_xlim(0, max(1.0, metric_values.max() * 1.1))
                ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            if output_file_path:
                self._save_plot(fig, output_file_path)
            else:
                plt.show()
                
        except Exception as e:
            raise VisualizationError(f"Failed to create model comparison plot: {str(e)}")
    
    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_probabilities: np.ndarray,
                      positive_class_label: Union[str, int] = 1,
                      output_file_path: Optional[str] = None,
                      **plot_kwargs) -> Dict[str, float]:
        """
        Create ROC (Receiver Operating Characteristic) curve plot.
        
        Args:
            y_true: True binary labels
            y_probabilities: Predicted probabilities for positive class
            positive_class_label: Label of the positive class
            output_file_path: Path to save the plot (optional)
            **plot_kwargs: Additional plotting parameters
            
        Returns:
            Dictionary containing AUC score and optimal threshold
            
        Raises:
            VisualizationError: If ROC curve creation fails
            
        Example:
            >>> roc_results = viz.plot_roc_curve(y_true, y_prob_positive)
            >>> print(f"AUC: {roc_results['auc']:.3f}")
        """
        try:
            # Calculate ROC curve
            false_positive_rates, true_positive_rates, thresholds = metrics.roc_curve(
                y_true, y_probabilities, pos_label=positive_class_label
            )
            roc_auc_score = metrics.auc(false_positive_rates, true_positive_rates)
            
            # Find optimal threshold (Youden's index)
            youden_index = true_positive_rates - false_positive_rates
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[optimal_idx]
            
            # Create plot
            plt.figure(figsize=self.config.figure_size)
            plt.plot(false_positive_rates, true_positive_rates, 
                    linewidth=3, label=f'ROC Curve (AUC = {roc_auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            
            # Mark optimal threshold
            plt.plot(false_positive_rates[optimal_idx], true_positive_rates[optimal_idx],
                    'ro', markersize=10, label=f'Optimal Threshold = {optimal_threshold:.3f}')
            
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve Analysis', fontsize=14)
            plt.legend(loc='lower right', fontsize=11)
            plt.grid(alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            
            if output_file_path:
                self._save_plot(plt.gcf(), output_file_path)
            else:
                plt.show()
            
            return {
                'auc': roc_auc_score,
                'optimal_threshold': optimal_threshold,
                'optimal_tpr': true_positive_rates[optimal_idx],
                'optimal_fpr': false_positive_rates[optimal_idx]
            }
            
        except Exception as e:
            raise VisualizationError(f"Failed to create ROC curve: {str(e)}")
    
    def plot_precision_recall_curve(self,
                                   y_true: np.ndarray,
                                   y_probabilities: np.ndarray,
                                   positive_class_label: Union[str, int] = 1,
                                   output_file_path: Optional[str] = None,
                                   **plot_kwargs) -> Dict[str, float]:
        """
        Create Precision-Recall curve plot.
        
        Args:
            y_true: True binary labels
            y_probabilities: Predicted probabilities for positive class
            positive_class_label: Label of the positive class
            output_file_path: Path to save the plot (optional)
            **plot_kwargs: Additional plotting parameters
            
        Returns:
            Dictionary containing PR AUC score and optimal threshold
            
        Example:
            >>> pr_results = viz.plot_precision_recall_curve(y_true, y_prob)
            >>> print(f"PR AUC: {pr_results['pr_auc']:.3f}")
        """
        try:
            # Calculate Precision-Recall curve
            precision_scores, recall_scores, thresholds = metrics.precision_recall_curve(
                y_true, y_probabilities, pos_label=positive_class_label
            )
            pr_auc_score = metrics.auc(recall_scores, precision_scores)
            
            # Calculate baseline (random classifier performance)
            positive_class_ratio = np.sum(y_true == positive_class_label) / len(y_true)
            
            # Find optimal threshold (F1 score maximization)
            f1_scores = 2 * (precision_scores[:-1] * recall_scores[:-1]) / (precision_scores[:-1] + recall_scores[:-1])
            f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Create plot
            plt.figure(figsize=self.config.figure_size)
            plt.plot(recall_scores, precision_scores, 
                    linewidth=3, label=f'PR Curve (AUC = {pr_auc_score:.3f})')
            plt.axhline(y=positive_class_ratio, color='red', linestyle='--', 
                       linewidth=2, label=f'Random Classifier (Baseline = {positive_class_ratio:.3f})')
            
            # Mark optimal threshold
            plt.plot(recall_scores[optimal_idx], precision_scores[optimal_idx],
                    'ro', markersize=10, label=f'Optimal Threshold = {optimal_threshold:.3f}')
            
            plt.xlabel('Recall (Sensitivity)', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curve Analysis', fontsize=14)
            plt.legend(loc='lower left', fontsize=11)
            plt.grid(alpha=0.3)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            
            if output_file_path:
                self._save_plot(plt.gcf(), output_file_path)
            else:
                plt.show()
            
            return {
                'pr_auc': pr_auc_score,
                'optimal_threshold': optimal_threshold,
                'optimal_precision': precision_scores[optimal_idx],
                'optimal_recall': recall_scores[optimal_idx],
                'optimal_f1': f1_scores[optimal_idx],
                'baseline': positive_class_ratio
            }
            
        except Exception as e:
            raise VisualizationError(f"Failed to create Precision-Recall curve: {str(e)}")
    
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_predicted: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            normalize: Optional[str] = None,
                            output_file_path: Optional[str] = None) -> np.ndarray:
        """
        Create confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_predicted: Predicted labels
            class_names: Names of classes for labeling
            normalize: Normalization method ('true', 'pred', 'all', or None)
            output_file_path: Path to save the plot (optional)
            
        Returns:
            Confusion matrix as numpy array
            
        Example:
            >>> cm = viz.plot_confusion_matrix(y_true, y_pred, ['Class A', 'Class B'])
        """
        try:
            # Calculate confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_true, y_predicted, normalize=normalize)
            
            # Set up class names
            unique_labels = np.unique(np.concatenate([y_true, y_predicted]))
            if class_names is None:
                class_names = [f'Class {label}' for label in unique_labels]
            elif len(class_names) != len(unique_labels):
                self.logger.warning("class_names length doesn't match number of classes, using default names")
                class_names = [f'Class {label}' for label in unique_labels]
            
            # Create plot
            plt.figure(figsize=self.config.figure_size)
            
            # Use appropriate colormap based on normalization
            colormap = 'Blues' if normalize else 'Blues'
            
            im = plt.imshow(confusion_matrix, interpolation='nearest', cmap=colormap)
            plt.title('Confusion Matrix', fontsize=14)
            plt.colorbar(im)
            
            # Set ticks and labels
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45, ha='right')
            plt.yticks(tick_marks, class_names)
            
            # Add text annotations
            threshold = confusion_matrix.max() / 2.0
            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    value = confusion_matrix[i, j]
                    display_text = f'{value:.2f}' if normalize else f'{int(value)}'
                    plt.text(j, i, display_text,
                            ha='center', va='center',
                            color='white' if value > threshold else 'black',
                            fontsize=12)
            
            plt.ylabel('True Labels', fontsize=12)
            plt.xlabel('Predicted Labels', fontsize=12)
            plt.tight_layout()
            
            if output_file_path:
                self._save_plot(plt.gcf(), output_file_path)
            else:
                plt.show()
            
            return confusion_matrix
            
        except Exception as e:
            raise VisualizationError(f"Failed to create confusion matrix: {str(e)}")
    
    def plot_learning_curves(self,
                           training_sizes: np.ndarray,
                           training_scores: np.ndarray,
                           validation_scores: np.ndarray,
                           output_file_path: Optional[str] = None,
                           metric_name: str = 'Score') -> None:
        """
        Create learning curves visualization showing training vs validation performance.
        
        Args:
            training_sizes: Array of training set sizes
            training_scores: Training scores for each size
            validation_scores: Validation scores for each size  
            output_file_path: Path to save the plot (optional)
            metric_name: Name of the performance metric
            
        Example:
            >>> viz.plot_learning_curves(sizes, train_scores, val_scores, 'learning_curve.png')
        """
        try:
            # Calculate means and standard deviations
            train_scores_mean = np.mean(training_scores, axis=1)
            train_scores_std = np.std(training_scores, axis=1)
            validation_scores_mean = np.mean(validation_scores, axis=1)
            validation_scores_std = np.std(validation_scores, axis=1)
            
            plt.figure(figsize=self.config.figure_size)
            
            # Plot training scores
            plt.plot(training_sizes, train_scores_mean, 'o-', color='blue', 
                    label='Training Score', linewidth=2, markersize=6)
            plt.fill_between(training_sizes, 
                           train_scores_mean - train_scores_std,
                           train_scores_mean + train_scores_std,
                           alpha=0.1, color='blue')
            
            # Plot validation scores
            plt.plot(training_sizes, validation_scores_mean, 'o-', color='red',
                    label='Cross-validation Score', linewidth=2, markersize=6)
            plt.fill_between(training_sizes,
                           validation_scores_mean - validation_scores_std,
                           validation_scores_mean + validation_scores_std,
                           alpha=0.1, color='red')
            
            plt.xlabel('Training Set Size', fontsize=12)
            plt.ylabel(metric_name, fontsize=12)
            plt.title('Learning Curves', fontsize=14)
            plt.legend(loc='lower right', fontsize=11)
            plt.grid(alpha=0.3)
            
            if output_file_path:
                self._save_plot(plt.gcf(), output_file_path)
            else:
                plt.show()
                
        except Exception as e:
            raise VisualizationError(f"Failed to create learning curves: {str(e)}")
    
    def _save_plot(self, figure, file_path: str) -> None:
        """
        Save plot to file with proper validation.
        
        Args:
            figure: Matplotlib figure object
            file_path: Path where to save the plot
            
        Raises:
            VisualizationError: If file saving fails
        """
        try:
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Save with high DPI
            figure.savefig(file_path, 
                          dpi=self.config.dpi,
                          format=self.config.save_format,
                          bbox_inches='tight',
                          facecolor='white',
                          edgecolor='none')
            
            self.logger.info(f"Plot saved to: {file_path}")
            
        except Exception as e:
            raise VisualizationError(f"Failed to save plot to {file_path}: {str(e)}")
    
    def create_gain_lift_chart(self,
                              y_true: np.ndarray,
                              y_probabilities: np.ndarray,
                              positive_class_label: Union[str, int] = 1,
                              n_groups: int = 10,
                              output_file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create Gain and Lift charts for model evaluation.
        
        Args:
            y_true: True binary labels
            y_probabilities: Predicted probabilities for positive class
            positive_class_label: Label of the positive class
            n_groups: Number of groups to divide the data into
            output_file_path: Path to save the plot (optional)
            
        Returns:
            DataFrame containing gain and lift calculations
            
        Example:
            >>> gain_lift_df = viz.create_gain_lift_chart(y_true, y_prob, n_groups=10)
        """
        try:
            # Create DataFrame and sort by probability
            results_df = pd.DataFrame({
                'actual': y_true,
                'probability': y_probabilities
            })
            results_df = results_df.sort_values('probability', ascending=False).reset_index(drop=True)
            
            # Calculate group size and create groups
            group_size = len(results_df) // n_groups
            results_df['group'] = results_df.index // group_size
            results_df['group'] = results_df['group'].clip(upper=n_groups-1)  # Handle remainder
            
            # Calculate cumulative statistics
            total_positives = (results_df['actual'] == positive_class_label).sum()
            total_records = len(results_df)
            
            gain_lift_stats = []
            
            for group_idx in range(n_groups):
                # Cumulative statistics up to this group
                cumulative_data = results_df[results_df['group'] <= group_idx]
                cumulative_positives = (cumulative_data['actual'] == positive_class_label).sum()
                cumulative_records = len(cumulative_data)
                
                # Calculate metrics
                percentage_records = (cumulative_records / total_records) * 100
                percentage_positives_captured = (cumulative_positives / total_positives) * 100 if total_positives > 0 else 0
                
                # Gain = % positives captured
                gain = percentage_positives_captured
                
                # Lift = (% positives captured) / (% records used)
                lift = gain / percentage_records if percentage_records > 0 else 0
                
                gain_lift_stats.append({
                    'group': group_idx + 1,
                    'percentage_records': percentage_records,
                    'cumulative_positives': cumulative_positives,
                    'gain': gain,
                    'lift': lift
                })
            
            gain_lift_df = pd.DataFrame(gain_lift_stats)
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gain Chart
            ax1.plot(gain_lift_df['percentage_records'], gain_lift_df['gain'], 
                    'o-', linewidth=3, markersize=8, label='Cumulative Gain')
            ax1.plot([0, 100], [0, 100], 'k--', linewidth=2, label='Random Model')
            ax1.set_xlabel('Percentage of Records (%)', fontsize=12)
            ax1.set_ylabel('Percentage of Positives Captured (%)', fontsize=12)
            ax1.set_title('Cumulative Gain Chart', fontsize=14)
            ax1.legend(fontsize=11)
            ax1.grid(alpha=0.3)
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 100)
            
            # Lift Chart
            ax2.plot(gain_lift_df['percentage_records'], gain_lift_df['lift'],
                    'o-', linewidth=3, markersize=8, color='red', label='Lift')
            ax2.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Random Model')
            ax2.set_xlabel('Percentage of Records (%)', fontsize=12)
            ax2.set_ylabel('Lift', fontsize=12)
            ax2.set_title('Lift Chart', fontsize=14)
            ax2.legend(fontsize=11)
            ax2.grid(alpha=0.3)
            ax2.set_xlim(0, 100)
            ax2.set_ylim(0, max(gain_lift_df['lift'].max() * 1.1, 2))
            
            plt.tight_layout()
            
            if output_file_path:
                self._save_plot(fig, output_file_path)
            else:
                plt.show()
            
            return gain_lift_df
            
        except Exception as e:
            raise VisualizationError(f"Failed to create gain/lift chart: {str(e)}")


# ============================================================================
# HYPERPARAMETER OPTIMIZER
# ============================================================================

class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using Optuna framework.
    
    Supports multiple optimization strategies, objective functions, and model types
    with built-in cross-validation and early stopping capabilities.
    
    Example:
        >>> optimizer = HyperparameterOptimizer()
        >>> best_params = optimizer.optimize_classifier(X_train, y_train, 'xgboost')
        >>> print(f"Best parameters: {best_params}")
    """
    
    def __init__(self, config: Optional[HyperparameterTuningConfiguration] = None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            config: Configuration object for optimization parameters
            
        Raises:
            HyperparameterTuningError: If Optuna is not available
        """
        if not OPTUNA_AVAILABLE:
            raise HyperparameterTuningError("Optuna library not available. Install with: pip install optuna")
            
        self.config = config or HyperparameterTuningConfiguration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance analyzer for evaluation
        self.performance_analyzer = ModelPerformanceAnalyzer()
    
    def optimize_classifier(self,
                          training_features: np.ndarray,
                          target_values: np.ndarray,
                          model_type: str,
                          scoring_metric: str = 'accuracy',
                          cv_folds: int = 5,
                          study_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a classification model.
        
        Args:
            training_features: Feature matrix for training
            target_values: Target variable
            model_type: Type of model ('xgboost', 'lightgbm', 'rf', 'svm')
            scoring_metric: Metric to optimize ('accuracy', 'f1', 'roc_auc', etc.)
            cv_folds: Number of cross-validation folds
            study_name: Optional name for the optimization study
            
        Returns:
            Dictionary containing best parameters and optimization results
            
        Raises:
            HyperparameterTuningError: If optimization fails
            
        Example:
            >>> best_results = optimizer.optimize_classifier(X, y, 'xgboost', 'f1')
            >>> model = XGBClassifier(**best_results['best_params'])
        """
        try:
            study_name = study_name or f"{model_type}_classification_optimization"
            
            # Create Optuna study
            study = optuna.create_study(
                direction=self.config.optimization_direction,
                study_name=study_name
            )
            
            # Create objective function
            def objective(trial):
                return self._classification_objective(
                    trial, training_features, target_values, 
                    model_type, scoring_metric, cv_folds
                )
            
            # Run optimization
            self.logger.info(f"Starting hyperparameter optimization for {model_type}")
            study.optimize(
                objective, 
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs
            )
            
            optimization_results = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'best_trial': study.best_trial.number,
                'n_trials': len(study.trials),
                'model_type': model_type,
                'scoring_metric': scoring_metric,
                'optimization_history': [(trial.number, trial.value) for trial in study.trials if trial.value is not None]
            }
            
            self.logger.info(f"Optimization complete. Best {scoring_metric}: {study.best_value:.4f}")
            return optimization_results
            
        except Exception as e:
            raise HyperparameterTuningError(f"Classification optimization failed: {str(e)}")
    
    def optimize_regressor(self,
                         training_features: np.ndarray,
                         target_values: np.ndarray,
                         model_type: str,
                         scoring_metric: str = 'neg_mean_squared_error',
                         cv_folds: int = 5,
                         study_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a regression model.
        
        Args:
            training_features: Feature matrix for training
            target_values: Target variable
            model_type: Type of model ('xgboost', 'lightgbm', 'rf', 'svr')
            scoring_metric: Metric to optimize ('neg_mean_squared_error', 'r2', etc.)
            cv_folds: Number of cross-validation folds
            study_name: Optional name for the optimization study
            
        Returns:
            Dictionary containing best parameters and optimization results
            
        Example:
            >>> best_results = optimizer.optimize_regressor(X, y, 'xgboost', 'r2')
        """
        try:
            study_name = study_name or f"{model_type}_regression_optimization"
            
            # Create Optuna study
            direction = 'maximize' if scoring_metric in ['r2', 'explained_variance'] else 'minimize'
            study = optuna.create_study(direction=direction, study_name=study_name)
            
            # Create objective function
            def objective(trial):
                return self._regression_objective(
                    trial, training_features, target_values,
                    model_type, scoring_metric, cv_folds
                )
            
            # Run optimization
            self.logger.info(f"Starting hyperparameter optimization for {model_type}")
            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs
            )
            
            optimization_results = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'best_trial': study.best_trial.number,
                'n_trials': len(study.trials),
                'model_type': model_type,
                'scoring_metric': scoring_metric,
                'optimization_history': [(trial.number, trial.value) for trial in study.trials if trial.value is not None]
            }
            
            self.logger.info(f"Optimization complete. Best {scoring_metric}: {study.best_value:.4f}")
            return optimization_results
            
        except Exception as e:
            raise HyperparameterTuningError(f"Regression optimization failed: {str(e)}")
    
    def _classification_objective(self,
                                trial,
                                training_features: np.ndarray,
                                target_values: np.ndarray,
                                model_type: str,
                                scoring_metric: str,
                                cv_folds: int) -> float:
        """
        Objective function for classification hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            training_features: Feature matrix
            target_values: Target variable
            model_type: Type of model to optimize
            scoring_metric: Evaluation metric
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation score for the trial
        """
        # Get model with suggested parameters
        model = self._suggest_classification_model(trial, model_type)
        
        # Perform cross-validation
        cv_results = self.performance_analyzer.cross_validate_model(
            model, training_features, target_values,
            scoring_metrics=[scoring_metric],
            cv_folds=cv_folds
        )
        
        # Return mean score
        test_scores = cv_results[f'test_{scoring_metric}']
        return np.mean(test_scores)
    
    def _regression_objective(self,
                            trial,
                            training_features: np.ndarray,
                            target_values: np.ndarray,
                            model_type: str,
                            scoring_metric: str,
                            cv_folds: int) -> float:
        """
        Objective function for regression hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            training_features: Feature matrix
            target_values: Target variable
            model_type: Type of model to optimize
            scoring_metric: Evaluation metric
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation score for the trial
        """
        # Get model with suggested parameters
        model = self._suggest_regression_model(trial, model_type)
        
        # Perform cross-validation
        cv_results = self.performance_analyzer.cross_validate_model(
            model, training_features, target_values,
            scoring_metrics=[scoring_metric],
            cv_folds=cv_folds
        )
        
        # Return mean score
        test_scores = cv_results[f'test_{scoring_metric}']
        return np.mean(test_scores)
    
    def _suggest_classification_model(self, trial, model_type: str):
        """
        Suggest classification model with hyperparameters for given trial.
        
        Args:
            trial: Optuna trial object
            model_type: Type of model
            
        Returns:
            Configured model instance
        """
        if model_type.lower() == 'xgboost' and XGBOOST_AVAILABLE:
            return XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
                reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
                random_state=self.config.random_state,
                eval_metric='logloss'
            )
        
        elif model_type.lower() == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return LGBMClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
                reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
                random_state=self.config.random_state,
                verbose=-1
            )
        
        elif model_type.lower() in ['rf', 'random_forest']:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                random_state=self.config.random_state
            )
        
        elif model_type.lower() == 'svm':
            from sklearn.svm import SVC
            return SVC(
                C=trial.suggest_float('C', 0.01, 100, log=True),
                gamma=trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('kernel', ['rbf', 'poly']) == 'rbf' else 'scale',
                kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                random_state=self.config.random_state,
                probability=True
            )
        
        else:
            raise HyperparameterTuningError(f"Unsupported classification model type: {model_type}")
    
    def _suggest_regression_model(self, trial, model_type: str):
        """
        Suggest regression model with hyperparameters for given trial.
        
        Args:
            trial: Optuna trial object
            model_type: Type of model
            
        Returns:
            Configured model instance
        """
        if model_type.lower() == 'xgboost' and XGBOOST_AVAILABLE:
            return XGBRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
                reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
                random_state=self.config.random_state
            )
        
        elif model_type.lower() == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return LGBMRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
                reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
                random_state=self.config.random_state,
                verbose=-1
            )
        
        elif model_type.lower() in ['rf', 'random_forest']:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                random_state=self.config.random_state
            )
        
        elif model_type.lower() == 'svr':
            from sklearn.svm import SVR
            return SVR(
                C=trial.suggest_float('C', 0.01, 100, log=True),
                gamma=trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('kernel', ['rbf', 'poly']) == 'rbf' else 'scale',
                kernel=trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                epsilon=trial.suggest_float('epsilon', 0.01, 1.0)
            )
        
        else:
            raise HyperparameterTuningError(f"Unsupported regression model type: {model_type}")


# ============================================================================
# FEATURE ANALYSIS TOOLKIT
# ============================================================================

class FeatureAnalysisToolkit:
    """
    Comprehensive toolkit for feature analysis, importance evaluation, and interpretation.
    
    Provides methods for feature importance analysis, SHAP values, PDP plots,
    dimensionality reduction, and feature selection techniques.
    
    Example:
        >>> toolkit = FeatureAnalysisToolkit()
        >>> importance = toolkit.calculate_feature_importance(model, X, y)
        >>> shap_values = toolkit.generate_shap_analysis(model, X)
    """
    
    def __init__(self, config: Optional[ModelTrainingConfiguration] = None):
        """
        Initialize the feature analysis toolkit.
        
        Args:
            config: Configuration object for analysis parameters
        """
        self.config = config or ModelTrainingConfiguration()
        self.logger = logging.getLogger(__name__)
        self.visualization_engine = MLVisualizationEngine()
    
    def calculate_feature_importance(self,
                                   fitted_model,
                                   training_features: np.ndarray,
                                   target_values: np.ndarray,
                                   feature_names: Optional[List[str]] = None,
                                   importance_type: str = 'built_in') -> pd.DataFrame:
        """
        Calculate feature importance using various methods.
        
        Args:
            fitted_model: Trained model with feature_importances_ attribute
            training_features: Feature matrix
            target_values: Target variable
            feature_names: Names of features
            importance_type: Type of importance ('built_in', 'permutation')
            
        Returns:
            DataFrame with feature names and importance scores
            
        Example:
            >>> importance_df = toolkit.calculate_feature_importance(model, X, y)
            >>> print(importance_df.head())
        """
        try:
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(training_features.shape[1])]
            
            if importance_type == 'built_in':
                if hasattr(fitted_model, 'feature_importances_'):
                    importances = fitted_model.feature_importances_
                elif hasattr(fitted_model, 'coef_'):
                    importances = np.abs(fitted_model.coef_).flatten()
                else:
                    self.logger.warning("Model doesn't have built-in feature importances, using permutation importance")
                    return self._calculate_permutation_importance(fitted_model, training_features, target_values, feature_names)
                
            elif importance_type == 'permutation':
                return self._calculate_permutation_importance(fitted_model, training_features, target_values, feature_names)
            
            else:
                raise ValidationError(f"Unsupported importance type: {importance_type}")
            
            # Create DataFrame and sort by importance
            importance_df = pd.DataFrame({
                'feature_name': feature_names,
                'importance': importances,
                'importance_normalized': importances / np.sum(importances) * 100
            })
            
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df = importance_df.reset_index(drop=True)
            
            return importance_df
            
        except Exception as e:
            raise DataProcessingError(f"Feature importance calculation failed: {str(e)}")
    
    def _calculate_permutation_importance(self,
                                        fitted_model,
                                        training_features: np.ndarray,
                                        target_values: np.ndarray,
                                        feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate permutation-based feature importance.
        
        Args:
            fitted_model: Trained model
            training_features: Feature matrix
            target_values: Target variable
            feature_names: Names of features
            
        Returns:
            DataFrame with permutation importance scores
        """
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            fitted_model, training_features, target_values,
            n_repeats=10, random_state=self.config.random_state
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature_name': feature_names,
            'importance': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std,
            'importance_normalized': perm_importance.importances_mean / np.sum(perm_importance.importances_mean) * 100
        })
        
        return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def generate_shap_analysis(self,
                             fitted_model,
                             training_features: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             sample_size: Optional[int] = 1000,
                             plot_types: List[str] = ['summary', 'waterfall', 'force']) -> Dict[str, Any]:
        """
        Generate SHAP (SHapley Additive exPlanations) analysis for model interpretability.
        
        Args:
            fitted_model: Trained model
            training_features: Feature matrix
            feature_names: Names of features
            sample_size: Number of samples for SHAP analysis
            plot_types: Types of SHAP plots to generate
            
        Returns:
            Dictionary containing SHAP values and analysis results
            
        Raises:
            DataProcessingError: If SHAP analysis fails
            
        Example:
            >>> shap_results = toolkit.generate_shap_analysis(model, X_train)
            >>> shap_values = shap_results['shap_values']
        """
        if not SHAP_AVAILABLE:
            raise DataProcessingError("SHAP library not available. Install with: pip install shap")
        
        try:
            # Sample data if too large
            if sample_size and len(training_features) > sample_size:
                indices = np.random.choice(len(training_features), sample_size, replace=False)
                sample_features = training_features[indices]
            else:
                sample_features = training_features
            
            # Create appropriate SHAP explainer
            if hasattr(fitted_model, 'predict_proba'):
                explainer = shap.TreeExplainer(fitted_model) if hasattr(fitted_model, 'get_booster') else shap.Explainer(fitted_model)
            else:
                explainer = shap.Explainer(fitted_model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(sample_features)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            shap_results = {
                'shap_values': shap_values,
                'base_value': explainer.expected_value,
                'feature_names': feature_names or [f'feature_{i}' for i in range(training_features.shape[1])],
                'sample_features': sample_features
            }
            
            # Generate plots if requested
            if 'summary' in plot_types and PLOTTING_AVAILABLE:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, sample_features, feature_names=shap_results['feature_names'], show=False)
                plt.title('SHAP Summary Plot')
                plt.tight_layout()
                plt.show()
            
            if 'waterfall' in plot_types and PLOTTING_AVAILABLE:
                shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                   base_values=explainer.expected_value, 
                                                   data=sample_features[0],
                                                   feature_names=shap_results['feature_names']))
            
            return shap_results
            
        except Exception as e:
            raise DataProcessingError(f"SHAP analysis failed: {str(e)}")
    
    def perform_pca_analysis(self,
                           features_matrix: np.ndarray,
                           target_values: Optional[np.ndarray] = None,
                           n_components: Optional[int] = None,
                           feature_names: Optional[List[str]] = None,
                           create_plots: bool = True) -> Dict[str, Any]:
        """
        Perform Principal Component Analysis with comprehensive diagnostics.
        
        Args:
            features_matrix: Feature matrix for PCA
            target_values: Optional target values for colored plots
            n_components: Number of components to extract
            feature_names: Names of original features
            create_plots: Whether to create diagnostic plots
            
        Returns:
            Dictionary containing PCA results and diagnostics
            
        Example:
            >>> pca_results = toolkit.perform_pca_analysis(X, y, n_components=5)
            >>> transformed_features = pca_results['transformed_features']
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize features
            scaler = StandardScaler()
            standardized_features = scaler.fit_transform(features_matrix)
            
            # Determine number of components
            if n_components is None:
                n_components = min(features_matrix.shape[0], features_matrix.shape[1])
            
            # Perform PCA
            pca_model = PCA(n_components=n_components, random_state=self.config.random_state)
            transformed_features = pca_model.fit_transform(standardized_features)
            
            # Calculate cumulative explained variance
            explained_variance_ratio = pca_model.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Find components explaining 95% variance
            components_95_variance = np.argmax(cumulative_variance >= 0.95) + 1
            
            pca_results = {
                'pca_model': pca_model,
                'transformed_features': transformed_features,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_explained_variance': cumulative_variance,
                'components_95_variance': components_95_variance,
                'feature_loadings': pca_model.components_.T,
                'scaler': scaler
            }
            
            if create_plots and PLOTTING_AVAILABLE:
                self._create_pca_plots(pca_results, target_values, feature_names)
            
            return pca_results
            
        except Exception as e:
            raise DataProcessingError(f"PCA analysis failed: {str(e)}")
    
    def _create_pca_plots(self,
                        pca_results: Dict[str, Any],
                        target_values: Optional[np.ndarray] = None,
                        feature_names: Optional[List[str]] = None) -> None:
        """
        Create comprehensive PCA diagnostic plots.
        
        Args:
            pca_results: Results from PCA analysis
            target_values: Optional target values for coloring
            feature_names: Names of original features
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Explained Variance
        axes[0, 0].bar(range(1, len(pca_results['explained_variance_ratio']) + 1), 
                      pca_results['explained_variance_ratio'])
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].set_title('Explained Variance by Component')
        
        # Plot 2: Cumulative Explained Variance
        axes[0, 1].plot(range(1, len(pca_results['cumulative_explained_variance']) + 1),
                       pca_results['cumulative_explained_variance'], 'bo-')
        axes[0, 1].axhline(y=0.95, color='red', linestyle='--', label='95% Variance')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].legend()
        
        # Plot 3: PC1 vs PC2 Scatter
        if target_values is not None:
            scatter = axes[1, 0].scatter(pca_results['transformed_features'][:, 0],
                                       pca_results['transformed_features'][:, 1],
                                       c=target_values, alpha=0.7, cmap='viridis')
            plt.colorbar(scatter, ax=axes[1, 0])
        else:
            axes[1, 0].scatter(pca_results['transformed_features'][:, 0],
                             pca_results['transformed_features'][:, 1], alpha=0.7)
        axes[1, 0].set_xlabel(f'PC1 ({pca_results["explained_variance_ratio"][0]:.2%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({pca_results["explained_variance_ratio"][1]:.2%} variance)')
        axes[1, 0].set_title('First Two Principal Components')
        
        # Plot 4: Feature Loadings Heatmap
        if feature_names:
            loadings_matrix = pca_results['feature_loadings'][:, :min(5, loadings_matrix.shape[1])]
            im = axes[1, 1].imshow(loadings_matrix.T, cmap='RdBu_r', aspect='auto')
            axes[1, 1].set_xticks(range(len(feature_names)))
            axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
            axes[1, 1].set_yticks(range(min(5, loadings_matrix.shape[1])))
            axes[1, 1].set_yticklabels([f'PC{i+1}' for i in range(min(5, loadings_matrix.shape[1]))])
            axes[1, 1].set_title('Feature Loadings (First 5 PCs)')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()


# ============================================================================
# DATA PROCESSING UTILITIES  
# ============================================================================

class DataProcessingUtilities:
    """
    Comprehensive utilities for data preprocessing, transformation, and validation.
    
    Provides methods for data cleaning, encoding, splitting, scaling, and
    handling missing values with ML-specific considerations.
    
    Example:
        >>> processor = DataProcessingUtilities()
        >>> X_clean = processor.handle_missing_values(X, strategy='iterative')
        >>> X_scaled = processor.scale_features(X_clean, method='standard')
    """
    
    def __init__(self, config: Optional[ModelTrainingConfiguration] = None):
        """
        Initialize the data processing utilities.
        
        Args:
            config: Configuration object for processing parameters
        """
        self.config = config or ModelTrainingConfiguration()
        self.logger = logging.getLogger(__name__)
    
    def handle_missing_values(self,
                            data_matrix: np.ndarray,
                            strategy: str = 'mean',
                            fill_value: Optional[Union[str, int, float]] = None) -> Tuple[np.ndarray, Any]:
        """
        Handle missing values using various imputation strategies.
        
        Args:
            data_matrix: Data matrix with potential missing values
            strategy: Imputation strategy ('mean', 'median', 'mode', 'constant', 'iterative')
            fill_value: Fill value for 'constant' strategy
            
        Returns:
            Tuple of (imputed_data, fitted_imputer)
            
        Raises:
            DataProcessingError: If imputation fails
            
        Example:
            >>> X_imputed, imputer = processor.handle_missing_values(X, 'median')
        """
        try:
            if strategy in ['mean', 'median', 'most_frequent', 'constant']:
                from sklearn.impute import SimpleImputer
                
                imputer = SimpleImputer(
                    strategy=strategy,
                    fill_value=fill_value,
                    add_indicator=False
                )
                
            elif strategy == 'iterative':
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                
                imputer = IterativeImputer(
                    random_state=self.config.random_state,
                    add_indicator=False
                )
                
            elif strategy == 'knn':
                from sklearn.impute import KNNImputer
                
                imputer = KNNImputer(n_neighbors=5, weights='uniform')
                
            else:
                raise ValidationError(f"Unsupported imputation strategy: {strategy}")
            
            # Fit and transform
            imputed_data = imputer.fit_transform(data_matrix)
            
            self.logger.info(f"Missing value imputation completed using {strategy} strategy")
            return imputed_data, imputer
            
        except Exception as e:
            raise DataProcessingError(f"Missing value imputation failed: {str(e)}")
    
    def scale_features(self,
                      data_matrix: np.ndarray,
                      method: str = 'standard',
                      feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Any]:
        """
        Scale features using various normalization methods.
        
        Args:
            data_matrix: Feature matrix to scale
            method: Scaling method ('standard', 'minmax', 'robust', 'quantile')
            feature_range: Range for MinMax scaling
            
        Returns:
            Tuple of (scaled_data, fitted_scaler)
            
        Example:
            >>> X_scaled, scaler = processor.scale_features(X, 'robust')
        """
        try:
            if method == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                
            elif method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=feature_range)
                
            elif method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                
            elif method == 'quantile':
                from sklearn.preprocessing import QuantileTransformer
                scaler = QuantileTransformer(
                    output_distribution='uniform',
                    random_state=self.config.random_state
                )
                
            else:
                raise ValidationError(f"Unsupported scaling method: {method}")
            
            # Fit and transform
            scaled_data = scaler.fit_transform(data_matrix)
            
            self.logger.info(f"Feature scaling completed using {method} method")
            return scaled_data, scaler
            
        except Exception as e:
            raise DataProcessingError(f"Feature scaling failed: {str(e)}")
    
    def encode_categorical_variables(self,
                                   data_matrix: np.ndarray,
                                   categorical_columns: List[int],
                                   encoding_method: str = 'onehot') -> Tuple[np.ndarray, Any]:
        """
        Encode categorical variables using various encoding methods.
        
        Args:
            data_matrix: Data matrix containing categorical variables
            categorical_columns: Indices of categorical columns
            encoding_method: Encoding method ('onehot', 'ordinal', 'target')
            
        Returns:
            Tuple of (encoded_data, fitted_encoder)
            
        Example:
            >>> X_encoded, encoder = processor.encode_categorical_variables(X, [0, 2], 'onehot')
        """
        try:
            if encoding_method == 'onehot':
                from sklearn.preprocessing import OneHotEncoder
                
                encoder = OneHotEncoder(
                    categories='auto',
                    drop='first',
                    sparse=False,
                    handle_unknown='ignore'
                )
                
                # Extract categorical columns
                categorical_data = data_matrix[:, categorical_columns]
                numerical_data = np.delete(data_matrix, categorical_columns, axis=1)
                
                # Encode categorical data
                encoded_categorical = encoder.fit_transform(categorical_data)
                
                # Combine encoded categorical with numerical
                if numerical_data.shape[1] > 0:
                    encoded_data = np.hstack([numerical_data, encoded_categorical])
                else:
                    encoded_data = encoded_categorical
                
            elif encoding_method == 'ordinal':
                from sklearn.preprocessing import OrdinalEncoder
                
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                
                # Create copy of data
                encoded_data = data_matrix.copy()
                
                # Encode categorical columns
                encoded_data[:, categorical_columns] = encoder.fit_transform(
                    data_matrix[:, categorical_columns]
                )
                
            else:
                raise ValidationError(f"Unsupported encoding method: {encoding_method}")
            
            self.logger.info(f"Categorical encoding completed using {encoding_method} method")
            return encoded_data, encoder
            
        except Exception as e:
            raise DataProcessingError(f"Categorical encoding failed: {str(e)}")
    
    def split_multilabel_data(self,
                            features_matrix: np.ndarray,
                            multilabel_targets: np.ndarray,
                            test_size: float = 0.2,
                            stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split multi-label data maintaining label distribution.
        
        Args:
            features_matrix: Feature matrix
            multilabel_targets: Multi-label target matrix
            test_size: Proportion of data for testing
            stratify: Whether to maintain label distribution
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            
        Example:
            >>> X_train, X_test, y_train, y_test = processor.split_multilabel_data(X, y_multi)
        """
        try:
            if stratify:
                from skmultilearn.model_selection import iterative_train_test_split
                
                X_train, y_train, X_test, y_test = iterative_train_test_split(
                    features_matrix, multilabel_targets, test_size=test_size
                )
                
            else:
                from sklearn.model_selection import train_test_split
                
                X_train, X_test, y_train, y_test = train_test_split(
                    features_matrix, multilabel_targets,
                    test_size=test_size,
                    random_state=self.config.random_state
                )
            
            self.logger.info(f"Multi-label data split completed. Train: {len(X_train)}, Test: {len(X_test)}")
            return X_train, X_test, y_train, y_test
            
        except ImportError:
            self.logger.warning("scikit-multilearn not available, using regular train_test_split")
            from sklearn.model_selection import train_test_split
            
            return train_test_split(
                features_matrix, multilabel_targets,
                test_size=test_size,
                random_state=self.config.random_state
            )
            
        except Exception as e:
            raise DataProcessingError(f"Multi-label data splitting failed: {str(e)}")
    
    def calculate_class_weights(self,
                              target_values: np.ndarray,
                              method: str = 'balanced') -> Dict[Any, float]:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Args:
            target_values: Target variable array
            method: Weight calculation method ('balanced', 'balanced_subsample')
            
        Returns:
            Dictionary mapping class labels to weights
            
        Example:
            >>> weights = processor.calculate_class_weights(y, 'balanced')
        """
        try:
            from sklearn.utils.class_weight import compute_class_weight
            
            unique_classes = np.unique(target_values)
            
            class_weights = compute_class_weight(
                class_weight=method,
                classes=unique_classes,
                y=target_values
            )
            
            class_weight_dict = dict(zip(unique_classes, class_weights))
            
            self.logger.info(f"Class weights calculated: {class_weight_dict}")
            return class_weight_dict
            
        except Exception as e:
            raise DataProcessingError(f"Class weight calculation failed: {str(e)}")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

class ModelTrainingPipeline:
    """
    Comprehensive machine learning training pipeline orchestrator.
    
    Coordinates model training, evaluation, comparison, and result reporting
    using all the specialized components of the ML toolbox.
    
    Example:
        >>> pipeline = ModelTrainingPipeline()
        >>> results = pipeline.run_classification_pipeline(X, y, ['rf', 'xgboost'])
        >>> best_model = results['best_model']
    """
    
    def __init__(self,
                 training_config: Optional[ModelTrainingConfiguration] = None,
                 viz_config: Optional[VisualizationConfiguration] = None):
        """
        Initialize the training pipeline.
        
        Args:
            training_config: Configuration for training operations
            viz_config: Configuration for visualization operations
        """
        self.training_config = training_config or ModelTrainingConfiguration()
        self.viz_config = viz_config or VisualizationConfiguration()
        
        # Initialize components
        self.performance_analyzer = ModelPerformanceAnalyzer(self.training_config)
        self.model_factory = ModelTemplateFactory(self.training_config)
        self.visualization_engine = MLVisualizationEngine(self.viz_config)
        self.feature_toolkit = FeatureAnalysisToolkit(self.training_config)
        self.data_processor = DataProcessingUtilities(self.training_config)
        self.hyperparameter_optimizer = HyperparameterOptimizer() if OPTUNA_AVAILABLE else None
        
        self.logger = logging.getLogger(__name__)
    
    def run_classification_pipeline(self,
                                  training_features: np.ndarray,
                                  target_values: np.ndarray,
                                  model_types: List[str] = None,
                                  feature_names: Optional[List[str]] = None,
                                  test_features: Optional[np.ndarray] = None,
                                  test_targets: Optional[np.ndarray] = None,
                                  optimize_hyperparameters: bool = False,
                                  create_visualizations: bool = True) -> Dict[str, Any]:
        """
        Run complete classification pipeline with model comparison and evaluation.
        
        Args:
            training_features: Training feature matrix
            target_values: Training target values
            model_types: List of model types to compare
            feature_names: Names of features
            test_features: Optional test feature matrix
            test_targets: Optional test target values
            optimize_hyperparameters: Whether to optimize hyperparameters
            create_visualizations: Whether to create result visualizations
            
        Returns:
            Dictionary containing comprehensive pipeline results
            
        Example:
            >>> results = pipeline.run_classification_pipeline(X, y, ['rf', 'xgboost'])
            >>> print(f"Best model: {results['best_model_name']}")
        """
        try:
            self.logger.info("Starting classification pipeline...")
            
            # Get model templates
            model_templates = self.model_factory.create_classifier_templates(target_values)
            
            # Filter models if specific types requested
            if model_types:
                available_models = {name: model for name, model in model_templates.items() 
                                  if any(model_type.lower() in name.lower() for model_type in model_types)}
                if not available_models:
                    available_models = model_templates
                    self.logger.warning(f"Requested models {model_types} not found, using all available models")
            else:
                available_models = model_templates
            
            # Compare models
            comparison_results = self._compare_models(
                available_models, training_features, target_values,
                test_features, test_targets
            )
            
            # Optimize hyperparameters for best model if requested
            best_model_info = comparison_results['best_model']
            if optimize_hyperparameters and self.hyperparameter_optimizer:
                self.logger.info(f"Optimizing hyperparameters for {best_model_info['name']}")
                optimization_results = self._optimize_best_model(
                    best_model_info, training_features, target_values, 'classification'
                )
                comparison_results['hyperparameter_optimization'] = optimization_results
            
            # Feature analysis
            feature_analysis_results = self._analyze_features(
                best_model_info['fitted_model'], training_features, target_values, feature_names
            )
            
            # Create visualizations
            visualization_results = {}
            if create_visualizations:
                visualization_results = self._create_classification_visualizations(
                    comparison_results, best_model_info, training_features, target_values,
                    test_features, test_targets
                )
            
            # Compile final results
            pipeline_results = {
                'pipeline_type': 'classification',
                'model_comparison': comparison_results,
                'best_model_name': best_model_info['name'],
                'best_model': best_model_info['fitted_model'],
                'best_model_scores': best_model_info['cv_scores'],
                'feature_analysis': feature_analysis_results,
                'visualizations': visualization_results,
                'training_config': self.training_config,
                'n_training_samples': len(training_features),
                'n_features': training_features.shape[1],
                'n_test_samples': len(test_features) if test_features is not None else None
            }
            
            self.logger.info("Classification pipeline completed successfully")
            return pipeline_results
            
        except Exception as e:
            raise ModelTrainingError(f"Classification pipeline failed: {str(e)}")
    
    def _compare_models(self,
                       model_templates: Dict[str, Any],
                       training_features: np.ndarray,
                       target_values: np.ndarray,
                       test_features: Optional[np.ndarray] = None,
                       test_targets: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compare multiple models using cross-validation and test set evaluation.
        
        Args:
            model_templates: Dictionary of model templates
            training_features: Training features
            target_values: Training targets
            test_features: Optional test features
            test_targets: Optional test targets
            
        Returns:
            Dictionary containing comparison results
        """
        comparison_results = {
            'model_scores': {},
            'cv_results': {},
            'test_results': {},
            'best_model': None
        }
        
        best_score = -np.inf
        
        for model_name, model_template in model_templates.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Cross-validation evaluation
                cv_results = self.performance_analyzer.cross_validate_model(
                    model_template, training_features, target_values,
                    scoring_metrics=self.training_config.scoring_metrics
                )
                
                # Fit model on full training set
                fitted_model = model_template.fit(training_features, target_values)
                
                # Calculate mean CV score for primary metric
                primary_metric = self.training_config.scoring_metrics[0]
                mean_cv_score = np.mean(cv_results[f'test_{primary_metric}'])
                
                # Test set evaluation if available
                test_scores = {}
                if test_features is not None and test_targets is not None:
                    test_predictions = fitted_model.predict(test_features)
                    test_scores = self.performance_analyzer.calculate_scores(
                        test_targets, test_predictions, self.training_config.scoring_metrics
                    )
                
                # Store results
                comparison_results['model_scores'][model_name] = mean_cv_score
                comparison_results['cv_results'][model_name] = cv_results
                comparison_results['test_results'][model_name] = test_scores
                
                # Track best model
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    comparison_results['best_model'] = {
                        'name': model_name,
                        'fitted_model': fitted_model,
                        'cv_scores': cv_results,
                        'test_scores': test_scores,
                        'mean_cv_score': mean_cv_score
                    }
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                comparison_results['model_scores'][model_name] = np.nan
        
        return comparison_results
    
    def _analyze_features(self,
                         fitted_model,
                         training_features: np.ndarray,
                         target_values: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive feature analysis on the best model.
        
        Args:
            fitted_model: Fitted model for analysis
            training_features: Training features
            target_values: Training targets
            feature_names: Names of features
            
        Returns:
            Dictionary containing feature analysis results
        """
        feature_analysis = {}
        
        try:
            # Feature importance
            importance_df = self.feature_toolkit.calculate_feature_importance(
                fitted_model, training_features, target_values, feature_names
            )
            feature_analysis['importance'] = importance_df
            
            # SHAP analysis if available
            if SHAP_AVAILABLE:
                shap_results = self.feature_toolkit.generate_shap_analysis(
                    fitted_model, training_features, feature_names
                )
                feature_analysis['shap_analysis'] = shap_results
            
        except Exception as e:
            self.logger.error(f"Feature analysis failed: {str(e)}")
            feature_analysis['error'] = str(e)
        
        return feature_analysis
    
    def _create_classification_visualizations(self,
                                            comparison_results: Dict[str, Any],
                                            best_model_info: Dict[str, Any],
                                            training_features: np.ndarray,
                                            target_values: np.ndarray,
                                            test_features: Optional[np.ndarray] = None,
                                            test_targets: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Create comprehensive visualizations for classification results.
        
        Args:
            comparison_results: Model comparison results
            best_model_info: Best model information
            training_features: Training features
            target_values: Training targets
            test_features: Optional test features
            test_targets: Optional test targets
            
        Returns:
            Dictionary containing visualization results
        """
        visualizations = {}
        
        try:
            # Model comparison plot
            scores_df = pd.DataFrame(comparison_results['model_scores'], index=[0]).T
            scores_df.columns = [self.training_config.scoring_metrics[0]]
            self.visualization_engine.plot_model_comparison(scores_df)
            visualizations['model_comparison'] = 'displayed'
            
            # ROC curve and PR curve for best model (if binary classification)
            if len(np.unique(target_values)) == 2 and test_features is not None:
                fitted_model = best_model_info['fitted_model']
                
                if hasattr(fitted_model, 'predict_proba'):
                    test_probabilities = fitted_model.predict_proba(test_features)[:, 1]
                    
                    # ROC curve
                    roc_results = self.visualization_engine.plot_roc_curve(
                        test_targets, test_probabilities
                    )
                    visualizations['roc_curve'] = roc_results
                    
                    # Precision-Recall curve
                    pr_results = self.visualization_engine.plot_precision_recall_curve(
                        test_targets, test_probabilities
                    )
                    visualizations['precision_recall_curve'] = pr_results
            
            # Confusion matrix
            if test_features is not None:
                test_predictions = best_model_info['fitted_model'].predict(test_features)
                confusion_matrix = self.visualization_engine.plot_confusion_matrix(
                    test_targets, test_predictions
                )
                visualizations['confusion_matrix'] = confusion_matrix
                
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {str(e)}")
            visualizations['error'] = str(e)
        
        return visualizations
    
    def _optimize_best_model(self,
                           best_model_info: Dict[str, Any],
                           training_features: np.ndarray,
                           target_values: np.ndarray,
                           task_type: str) -> Dict[str, Any]:
        """
        Optimize hyperparameters for the best performing model.
        
        Args:
            best_model_info: Information about the best model
            training_features: Training features
            target_values: Training targets
            task_type: Type of ML task ('classification' or 'regression')
            
        Returns:
            Dictionary containing optimization results
        """
        model_name = best_model_info['name'].lower()
        
        # Determine model type for optimization
        if 'xgboost' in model_name or 'xgb' in model_name:
            model_type = 'xgboost'
        elif 'lightgbm' in model_name or 'lgbm' in model_name:
            model_type = 'lightgbm'
        elif 'forest' in model_name or 'rf' in model_name:
            model_type = 'rf'
        elif 'svm' in model_name or 'svc' in model_name:
            model_type = 'svm'
        else:
            self.logger.warning(f"Hyperparameter optimization not supported for {model_name}")
            return {'error': f'Optimization not supported for {model_name}'}
        
        try:
            if task_type == 'classification':
                return self.hyperparameter_optimizer.optimize_classifier(
                    training_features, target_values, model_type
                )
            else:
                return self.hyperparameter_optimizer.optimize_regressor(
                    training_features, target_values, model_type
                )
                
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {str(e)}")
            return {'error': str(e)}


# ============================================================================
# BACKWARD COMPATIBILITY LAYER
# ============================================================================

# Global instances for backward compatibility
_performance_analyzer = None
_model_factory = None
_visualization_engine = None
_feature_toolkit = None
_data_processor = None
_hyperparameter_optimizer = None
_training_pipeline = None

def _get_performance_analyzer():
    """Get or create global performance analyzer instance."""
    global _performance_analyzer
    if _performance_analyzer is None:
        _performance_analyzer = ModelPerformanceAnalyzer()
    return _performance_analyzer

def _get_model_factory():
    """Get or create global model factory instance."""
    global _model_factory
    if _model_factory is None:
        _model_factory = ModelTemplateFactory()
    return _model_factory

def _get_visualization_engine():
    """Get or create global visualization engine instance."""
    global _visualization_engine
    if _visualization_engine is None and PLOTTING_AVAILABLE:
        _visualization_engine = MLVisualizationEngine()
    return _visualization_engine

def _get_feature_toolkit():
    """Get or create global feature toolkit instance."""
    global _feature_toolkit
    if _feature_toolkit is None:
        _feature_toolkit = FeatureAnalysisToolkit()
    return _feature_toolkit

def _get_data_processor():
    """Get or create global data processor instance."""
    global _data_processor
    if _data_processor is None:
        _data_processor = DataProcessingUtilities()
    return _data_processor

def _get_hyperparameter_optimizer():
    """Get or create global hyperparameter optimizer instance."""
    global _hyperparameter_optimizer
    if _hyperparameter_optimizer is None and OPTUNA_AVAILABLE:
        _hyperparameter_optimizer = HyperparameterOptimizer()
    return _hyperparameter_optimizer

def _get_training_pipeline():
    """Get or create global training pipeline instance."""
    global _training_pipeline
    if _training_pipeline is None:
        _training_pipeline = ModelTrainingPipeline()
    return _training_pipeline


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def classifiers_template(y, random_state=10, imputer_strategy='mean', pc_n_components=5):
    """
    BACKWARD COMPATIBILITY: Create pre-configured classifier templates.
    
    This function maintains compatibility with the original API while using
    the new ModelTemplateFactory class.
    
    Args:
        y: Target variable distribution (for balancing considerations)
        random_state: Random state for reproducibility
        imputer_strategy: Strategy for missing value imputation
        pc_n_components: Number of PCA components
        
    Returns:
        Dictionary of classifier templates
    """
    config = ModelTrainingConfiguration(
        random_state=random_state,
        imputer_strategy=imputer_strategy,
        pca_n_components=pc_n_components
    )
    
    factory = ModelTemplateFactory(config)
    return factory.create_classifier_templates(y)


def regressors_template(random_state=10):
    """
    BACKWARD COMPATIBILITY: Create pre-configured regressor templates.
    
    Args:
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary of regressor templates
    """
    config = ModelTrainingConfiguration(random_state=random_state)
    factory = ModelTemplateFactory(config)
    return factory.create_regressor_templates()


def ml_scores(y_model, scores_names, **kwargs):
    """
    BACKWARD COMPATIBILITY: Calculate multiple ML evaluation scores.
    
    Args:
        y_model: DataFrame with true and predicted values
        scores_names: List of score names to calculate
        **kwargs: Additional arguments for specific metrics
        
    Returns:
        Dictionary of calculated scores
    """
    analyzer = _get_performance_analyzer()
    
    # Extract true and predicted values from DataFrame
    if hasattr(y_model, 'columns'):
        if 'y_true' in y_model.columns and 'y_pred' in y_model.columns:
            y_true = y_model['y_true'].values
            y_pred = y_model['y_pred'].values
        else:
            # Assume first two columns are true and predicted
            y_true = y_model.iloc[:, 0].values
            y_pred = y_model.iloc[:, 1].values
    else:
        raise ValidationError("y_model must be a DataFrame with true and predicted values")
    
    return analyzer.calculate_scores(y_true, y_pred, scores_names, **kwargs)


def ml_scores_crossvalidate(**kwargs):
    """
    BACKWARD COMPATIBILITY: Perform cross-validation with multiple scoring metrics.
    
    Args:
        **kwargs: Keyword arguments for cross-validation
        
    Returns:
        Cross-validation results dictionary
    """
    analyzer = _get_performance_analyzer()
    
    # Extract required parameters
    estimator = kwargs.get('estimator')
    X = kwargs.get('X')
    y = kwargs.get('y')
    scoring = kwargs.get('scoring', ['accuracy'])
    cv = kwargs.get('cv', 5)
    
    if not all([estimator is not None, X is not None, y is not None]):
        raise ValidationError("estimator, X, and y are required parameters")
    
    return analyzer.cross_validate_model(estimator, X, y, scoring, cv)


def ml_prediction(ml_model, X, y, sk_fold, X_test=None, y_test=None, callbacks=None, verbose=False):
    """
    BACKWARD COMPATIBILITY: Perform ML prediction with cross-validation and optional early stopping.
    
    Args:
        ml_model: Machine learning model to use for prediction
        X: Feature matrix for training
        y: Target vector for training
        sk_fold: Cross-validation splitting strategy or list containing validation data
        X_test: Optional test feature matrix
        y_test: Optional test target vector
        callbacks: Optional list of callback functions for early stopping
        verbose: If True, print progress messages
        
    Returns:
        Tuple of (y_model, ml_models, df_epochs) where:
        - y_model: DataFrame with predictions and true values for each CV iteration
        - ml_models: List of fitted models for each CV iteration
        - df_epochs: DataFrame with epoch information or None if no early stopping
        
    Example:
        >>> from sklearn.model_selection import StratifiedKFold
        >>> cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        >>> y_model, models, epochs = ml_prediction(model, X, y, cv)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
    from sklearn.base import is_classifier
    import copy
    
    y_model = pd.DataFrame([])
    df_epochs = pd.DataFrame([])
    ml_models = []
    
    # Get the actual model from pipeline if necessary
    umodel = ml_model[-1] if isinstance(ml_model, Pipeline) else ml_model
    model_name = umodel.__class__.__name__
    
    # Check for early stopping capability (mainly XGBoost)
    early_stopping_rounds = getattr(umodel, 'early_stopping_rounds', None) if 'xgb' in model_name.lower() else None
    
    # Warning messages for awkward scenarios
    if (X_test is not None) and (isinstance(sk_fold, (StratifiedKFold, TimeSeriesSplit))) and (early_stopping_rounds is None):
        print("Warning! Awkward scenario. Cross validation is on and prediction on a separate test data set!")
    
    # Handle different types of fold inputs
    if sk_fold is None:
        print("Warning! Training and validation data sets are the same")
        cv = [(list(range(X.shape[0])), list(range(X.shape[0])))]
    elif isinstance(sk_fold, list):
        print("No cross validation")
        X_val = sk_fold[0]
        y_val = sk_fold[1]
        train_no = X.shape[0]
        X = pd.concat([X, X_val], axis=0)
        y = pd.concat([y, y_val], axis=0)
        cv = [(list(range(train_no)), list(range(train_no, X.shape[0])))]
    else:
        cv = list(sk_fold.split(X, y))
    
    # Perform cross-validation
    for cv_itr, (train_index, val_index) in enumerate(cv):
        if verbose:
            print(f"CV Iteration {cv_itr + 1}")
        
        # Create a deep copy of the model for this iteration
        current_model = copy.deepcopy(ml_model)
        
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Handle early stopping for XGBoost-like models
        if early_stopping_rounds is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            current_model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=callbacks,
                verbose=10 if verbose else 0
            )
            
            # Extract epoch information if available
            if hasattr(current_model, 'evals_result_'):
                try:
                    results = current_model.evals_result_
                    df_epochs_tmp = pd.DataFrame()
                    for metric_key in results['validation_0'].keys():
                        val0 = results['validation_0'][metric_key]
                        val1 = results['validation_1'][metric_key]
                        tmp = pd.DataFrame(
                            [val0, val1], 
                            index=[f'Train_{metric_key}', f'Validation_{metric_key}']
                        ).T
                        df_epochs_tmp = pd.concat([df_epochs_tmp, tmp], axis=1)
                    
                    df_epochs_tmp.index.name = 'epochs'
                    df_epochs_tmp = df_epochs_tmp.reset_index()
                    df_epochs_tmp['best_ntree'] = getattr(current_model, 'best_iteration', None)
                    df_epochs_tmp['CV_Iteration'] = cv_itr
                    df_epochs = pd.concat([df_epochs, df_epochs_tmp], axis=0)
                    
                    if verbose:
                        print(f"Best iteration: {getattr(current_model, 'best_iteration', 'N/A')}")
                        print(f"Best score: {getattr(current_model, 'best_score', 'N/A')}")
                except Exception as e:
                    print(f"Could not extract epoch information: {e}")
        else:
            current_model.fit(X_train, y_train)
        
        ml_models.append(current_model)
        
        # Use test data if provided, otherwise use validation data
        X_pred = X_test if X_test is not None else X_val
        y_true = y_test if y_test is not None else y_val
        
        # Make predictions based on model type
        if is_classifier(umodel):
            if hasattr(current_model, 'predict_proba'):
                y_proba = current_model.predict_proba(X_pred)
                y_model0 = pd.DataFrame(y_proba, index=y_true.index)
                y_pred = y_model0.idxmax(axis=1)
                y_model0 = pd.concat([y_model0, y_pred.rename('y_pred'), y_true.rename('y_true')], axis=1)
            else:
                y_pred = current_model.predict(X_pred)
                y_model0 = pd.DataFrame({
                    'y_pred': y_pred,
                    'y_true': y_true.values
                }, index=y_true.index)
        else:
            # Regression
            y_pred = current_model.predict(X_pred)
            y_model0 = pd.DataFrame({
                'y_pred': y_pred,
                'y_true': y_true.values
            }, index=y_true.index)
        
        y_model0['CV_Iteration'] = cv_itr
        y_model = pd.concat([y_model, y_model0], axis=0)
    
    return y_model, ml_models, df_epochs if not df_epochs.empty else None


def ml_prediction_sub_epochs(model):
    """
    BACKWARD COMPATIBILITY: Extract epoch information from XGBoost models.
    
    Args:
        model: Fitted XGBoost model with evaluation results
        
    Returns:
        DataFrame containing epoch information
    """
    try:
        results = model.evals_result_
        df_epochs = pd.DataFrame()
        
        for metric_key in results['validation_0'].keys():
            val0 = results['validation_0'][metric_key]
            val1 = results['validation_1'][metric_key]
            tmp = pd.DataFrame(
                [val0, val1], 
                index=[f'Train_{metric_key}', f'Validation_{metric_key}']
            ).T
            df_epochs = pd.concat([df_epochs, tmp], axis=1)
        
        df_epochs.index.name = 'epochs'
        df_epochs = df_epochs.reset_index()
        df_epochs['best_ntree'] = getattr(model, 'best_iteration', None)
        
        return df_epochs
    except Exception as e:
        print(f"Error extracting epoch information: {e}")
        return pd.DataFrame()


def ml_comparison(ml_models, X, y, sk_fold, X_test=None, y_test=None, **kwargs):
    """
    BACKWARD COMPATIBILITY: Compare multiple ML models.
    
    Args:
        ml_models: Dictionary of models to compare
        X: Training features
        y: Training targets
        sk_fold: Cross-validation splitter
        X_test: Optional test features
        y_test: Optional test targets
        **kwargs: Additional arguments
        
    Returns:
        Model comparison results
    """
    pipeline = _get_training_pipeline()
    
    # Convert to new API format
    if isinstance(ml_models, dict):
        model_types = list(ml_models.keys())
    else:
        model_types = None
    
    results = pipeline.run_classification_pipeline(
        training_features=X,
        target_values=y,
        model_types=model_types,
        test_features=X_test,
        test_targets=y_test
    )
    
    return results


def ml_comparison_plot(metrics_all, outputFile=None):
    """
    BACKWARD COMPATIBILITY: Create model comparison visualization.
    
    Args:
        metrics_all: DataFrame with model performance metrics
        outputFile: Optional output file path
    """
    viz_engine = _get_visualization_engine()
    if viz_engine:
        viz_engine.plot_model_comparison(metrics_all, outputFile)


def precision_recall_curve2(y, model_prob, pos_label, outputFile=None, **kwargs):
    """
    BACKWARD COMPATIBILITY: Create precision-recall curve.
    
    Args:
        y: True binary labels
        model_prob: Predicted probabilities
        pos_label: Positive class label
        outputFile: Optional output file path
        **kwargs: Additional plotting parameters
        
    Returns:
        Precision-recall curve results
    """
    viz_engine = _get_visualization_engine()
    if viz_engine:
        return viz_engine.plot_precision_recall_curve(
            y, model_prob, pos_label, outputFile, **kwargs
        )


def roc_curve2(y, model_prob, pos_label, outputFile=None, **kwargs):
    """
    BACKWARD COMPATIBILITY: Create ROC curve.
    
    Args:
        y: True binary labels
        model_prob: Predicted probabilities
        pos_label: Positive class label
        outputFile: Optional output file path
        **kwargs: Additional plotting parameters
        
    Returns:
        ROC curve results
    """
    viz_engine = _get_visualization_engine()
    if viz_engine:
        return viz_engine.plot_roc_curve(
            y, model_prob, pos_label, outputFile, **kwargs
        )


def plot_confusion_matrix2(y_model, map_lbls, outputFile=None, ncol=3, all_data_flag=True):
    """
    BACKWARD COMPATIBILITY: Create confusion matrix plot.
    
    Args:
        y_model: DataFrame with true and predicted values
        map_lbls: Class label mappings
        outputFile: Optional output file path
        ncol: Number of columns (unused)
        all_data_flag: Whether to use all data (unused)
        
    Returns:
        Confusion matrix array
    """
    viz_engine = _get_visualization_engine()
    if viz_engine:
        # Extract true and predicted values
        if hasattr(y_model, 'columns'):
            y_true = y_model.iloc[:, 0].values
            y_pred = y_model.iloc[:, 1].values
        else:
            raise ValidationError("y_model must be a DataFrame")
        
        class_names = list(map_lbls.keys()) if isinstance(map_lbls, dict) else None
        return viz_engine.plot_confusion_matrix(y_true, y_pred, class_names, output_file_path=outputFile)


def feature_importance_batch(umodel, X, y):
    """
    BACKWARD COMPATIBILITY: Calculate feature importance for a model.
    
    Args:
        umodel: Fitted model
        X: Feature matrix
        y: Target variable
        
    Returns:
        Feature importance DataFrame
    """
    toolkit = _get_feature_toolkit()
    return toolkit.calculate_feature_importance(umodel, X, y)


def shap_plots_batch(X, y, umodel, test_size=0.2, kmeans=None, random_state=100):
    """
    BACKWARD COMPATIBILITY: Generate SHAP analysis plots.
    
    Args:
        X: Feature matrix
        y: Target variable
        umodel: Fitted model
        test_size: Test set size (unused)
        kmeans: KMeans clustering (unused)
        random_state: Random state
        
    Returns:
        SHAP analysis results
    """
    toolkit = _get_feature_toolkit()
    if SHAP_AVAILABLE:
        return toolkit.generate_shap_analysis(umodel, X)
    else:
        raise DataProcessingError("SHAP library not available")


def hyperparameter_tuning(X_train, y_train, model_type, scoring='accuracy', n_trials=100, **kwargs):
    """
    BACKWARD COMPATIBILITY: Hyperparameter optimization with Optuna.
    
    Args:
        X_train: Training features
        y_train: Training targets
        model_type: Type of model to optimize
        scoring: Scoring metric
        n_trials: Number of optimization trials
        **kwargs: Additional arguments
        
    Returns:
        Optimization results
    """
    optimizer = _get_hyperparameter_optimizer()
    if optimizer:
        return optimizer.optimize_classifier(
            X_train, y_train, model_type, scoring, n_trials=n_trials
        )
    else:
        raise HyperparameterTuningError("Optuna not available for hyperparameter tuning")


def pca_plot(df):
    """
    BACKWARD COMPATIBILITY: Create PCA analysis plots.
    
    Args:
        df: DataFrame with features
        
    Returns:
        PCA analysis results
    """
    toolkit = _get_feature_toolkit()
    return toolkit.perform_pca_analysis(df.values)


def class_weight2(uclass_weight, y):
    """
    BACKWARD COMPATIBILITY: Calculate class weights.
    
    Args:
        uclass_weight: Weight calculation method
        y: Target variable
        
    Returns:
        Class weight dictionary
    """
    processor = _get_data_processor()
    return processor.calculate_class_weights(y, uclass_weight)


def binarizer(tags):
    """
    BACKWARD COMPATIBILITY: Binarize multi-label tags.
    
    Args:
        tags: Multi-label tags to binarize
        
    Returns:
        Binarized label matrix
    """
    from sklearn.preprocessing import MultiLabelBinarizer
    
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(tags)
    return binary_labels, mlb


def split_multiLabel_data(df_samples2, binarized_tags, random_state=None):
    """
    BACKWARD COMPATIBILITY: Split multi-label data.
    
    Args:
        df_samples2: Feature DataFrame
        binarized_tags: Binarized label matrix
        random_state: Random state for reproducibility
        
    Returns:
        Train-test split results
    """
    processor = _get_data_processor()
    return processor.split_multilabel_data(
        df_samples2.values, binarized_tags, stratify=True
    )


def gainNlift(y, model_prob, pos_label, outputFile, groupNo=25):
    """
    BACKWARD COMPATIBILITY: Create gain and lift charts.
    
    Args:
        y: True binary labels
        model_prob: Predicted probabilities
        pos_label: Positive class label
        outputFile: Output file path
        groupNo: Number of groups
        
    Returns:
        Gain and lift analysis results
    """
    viz_engine = _get_visualization_engine()
    if viz_engine:
        return viz_engine.create_gain_lift_chart(
            y, model_prob, pos_label, groupNo, outputFile
        )


def classifer_performance_batch(y_model, 
                               map_lbls={0:'Low Loss', 1:'High Loss'},
                               scores_names=['accuracy', 'recall', 'precision'],
                               multi_class='raise',
                               average='balanced'):
    """
    BACKWARD COMPATIBILITY: Evaluate classifier performance using various metrics.
    
    Args:
        y_model: Dictionary with 'y_true' and 'prob' keys
        map_lbls: Dictionary mapping class labels to descriptions
        scores_names: List of score names to evaluate
        multi_class: Multi-class handling strategy
        average: Averaging strategy for multi-class metrics
        
    Returns:
        Tuple of (scores, confusion_matrices)
    """
    # Extract data from y_model
    if isinstance(y_model, dict):
        y_true = y_model['y_true']
        model_prob = y_model['prob']
    else:
        # Assume DataFrame format
        y_true = y_model['y_true']
        model_prob = y_model['prob'] if 'prob' in y_model.columns else y_model.iloc[:, 1]
    
    # Calculate confusion matrices
    y_pred = (model_prob > 0.5).astype(int) if len(np.unique(y_true)) == 2 else np.round(model_prob).astype(int)
    confusion_matrices = plot_confusion_matrix2(
        pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}), 
        map_lbls, outputFile=None
    )
    
    # Calculate scores
    analyzer = _get_performance_analyzer()
    scores = analyzer.calculate_scores(y_true, y_pred, scores_names, average=average)
    
    return scores, confusion_matrices


def ml_prediction_xValNest(ml_model, X, y, outter_fold, inner_fold):
    """
    BACKWARD COMPATIBILITY: Perform nested cross-validation.
    
    Args:
        ml_model: Machine learning model
        X: Feature matrix
        y: Target variable
        outter_fold: Outer cross-validation fold
        inner_fold: Inner cross-validation fold
        
    Returns:
        Tuple of (y_model, df_epochs)
    """
    import copy
    
    y_model = pd.DataFrame([])
    df_epochs = pd.DataFrame([])
    
    for cv_outer, (trainVal_index, test_index) in enumerate(outter_fold.split(X, y)):
        X_trainVal, X_test = X.iloc[trainVal_index, :], X.iloc[test_index, :]
        y_trainVal, y_test = y.iloc[trainVal_index], y.iloc[test_index]
        
        best_iteration = None
        
        for cv_inner, (train_index, val_index) in enumerate(inner_fold.split(X_trainVal, y_trainVal)):
            X_train, X_val = X_trainVal.iloc[train_index, :], X_trainVal.iloc[val_index, :]
            y_train, y_val = y_trainVal.iloc[train_index], y_trainVal.iloc[val_index]
            
            # Create model copy for this iteration
            model_copy = copy.deepcopy(ml_model)
            
            # Handle XGBoost early stopping
            if 'xgb' in ml_model.__class__.__name__.lower():
                eval_set = [(X_train, y_train), (X_val, y_val)]
                model_copy.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                
                if hasattr(model_copy, 'best_iteration'):
                    best_iteration = model_copy.best_iteration
                    
                # Track epochs if available
                if hasattr(model_copy, 'evals_result_'):
                    df_epochs_tmp = ml_prediction_sub_epochs(model_copy)
                    df_epochs_tmp['CV_Iteration'] = f'{cv_outer}_{cv_inner}'
                    df_epochs = pd.concat([df_epochs, df_epochs_tmp], axis=0)
            else:
                model_copy.fit(X_train, y_train)
        
        # Train final model on full training+validation set
        if best_iteration and 'xgb' in ml_model.__class__.__name__.lower():
            final_model = copy.deepcopy(ml_model)
            final_model.set_params(n_estimators=best_iteration)
            final_model.fit(X_trainVal, y_trainVal)
        else:
            final_model = copy.deepcopy(ml_model)
            final_model.fit(X_trainVal, y_trainVal)
        
        # Make predictions on test set
        if hasattr(final_model, 'predict_proba'):
            y_proba = final_model.predict_proba(X_test)
            y_model0 = pd.DataFrame(y_proba, index=X_test.index)
            y_pred = y_model0.idxmax(axis=1)
            y_model0 = pd.concat([y_model0, y_pred.rename('y_pred')], axis=1)
        else:
            y_pred = final_model.predict(X_test)
            y_model0 = pd.DataFrame({'y_pred': y_pred}, index=X_test.index)
        
        y_model0['CV_Iteration'] = f'{cv_outer}'
        y_model0['y_true'] = y_test
        y_model = pd.concat([y_model, y_model0], axis=0)
    
    return y_model, df_epochs


def learning_curve_early_stopping(df_epochs, outputFile=None):
    """
    BACKWARD COMPATIBILITY: Create learning curves with early stopping visualization.
    
    Args:
        df_epochs: DataFrame containing epoch information
        outputFile: Optional output file path
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available")
        return
        
    plt.figure(figsize=(12, 8))
    
    # Group by CV iteration if available
    if 'CV_Iteration' in df_epochs.columns:
        for cv_iter in df_epochs['CV_Iteration'].unique():
            cv_data = df_epochs[df_epochs['CV_Iteration'] == cv_iter]
            
            # Plot training and validation curves
            for col in cv_data.columns:
                if col.startswith('Train_'):
                    metric_name = col.replace('Train_', '')
                    val_col = f'Validation_{metric_name}'
                    
                    if val_col in cv_data.columns:
                        plt.plot(cv_data['epochs'], cv_data[col], 
                               label=f'Train {metric_name} (CV {cv_iter})', alpha=0.7)
                        plt.plot(cv_data['epochs'], cv_data[val_col], 
                               label=f'Val {metric_name} (CV {cv_iter})', alpha=0.7)
                        
                        # Mark best iteration
                        if 'best_ntree' in cv_data.columns:
                            best_epoch = cv_data['best_ntree'].iloc[0]
                            if pd.notna(best_epoch) and best_epoch < len(cv_data):
                                plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
    
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Learning Curves with Early Stopping')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if outputFile:
        plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def evaluate_multiLabel(y_pred, y_true, **kwargs):
    """
    BACKWARD COMPATIBILITY: Evaluate multi-label classification performance.
    
    Args:
        y_pred: Predicted labels
        y_true: True labels
        **kwargs: Additional evaluation parameters
        
    Returns:
        Comprehensive evaluation results
    """
    from sklearn.metrics import classification_report, hamming_loss, jaccard_score
    
    # Calculate various multi-label metrics
    results = {
        'hamming_loss': hamming_loss(y_true, y_pred),
        'jaccard_score_micro': jaccard_score(y_true, y_pred, average='micro'),
        'jaccard_score_macro': jaccard_score(y_true, y_pred, average='macro'),
        'jaccard_score_samples': jaccard_score(y_true, y_pred, average='samples'),
    }
    
    # Classification report
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        results['classification_report'] = report
    except Exception as e:
        print(f"Could not generate classification report: {e}")
    
    # Subset accuracy (exact match)
    subset_accuracy = np.mean(np.all(y_pred == y_true, axis=1))
    results['subset_accuracy'] = subset_accuracy
    
    return results


def ml_tuner(trial, X_train, y_train, X_val, y_val, model_type='xgboost', scoring='accuracy'):
    """
    BACKWARD COMPATIBILITY: Optuna trial function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_type: Type of model to tune
        scoring: Scoring metric
        
    Returns:
        Score for this trial
    """
    optimizer = _get_hyperparameter_optimizer()
    if optimizer:
        # Create temporary model with suggested parameters
        if model_type.lower() == 'xgboost' and XGBOOST_AVAILABLE:
            model = optimizer._suggest_classification_model(trial, 'xgboost')
        elif model_type.lower() == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model = optimizer._suggest_classification_model(trial, 'lightgbm')
        else:
            raise ValueError(f"Unsupported model type for ml_tuner: {model_type}")
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        analyzer = _get_performance_analyzer()
        scores = analyzer.calculate_scores(y_val, y_pred, [scoring])
        return scores[scoring]
    else:
        raise HyperparameterTuningError("Optuna not available")


def xgb_tuner(X_train, y_train, X_val=None, y_val=None, n_trials=100, **kwargs):
    """
    BACKWARD COMPATIBILITY: XGBoost-specific hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Optional validation features
        y_val: Optional validation targets
        n_trials: Number of optimization trials
        **kwargs: Additional arguments
        
    Returns:
        Best parameters and optimization results
    """
    optimizer = _get_hyperparameter_optimizer()
    if optimizer and XGBOOST_AVAILABLE:
        return optimizer.optimize_classifier(
            X_train, y_train, 'xgboost', n_trials=n_trials, **kwargs
        )
    else:
        raise HyperparameterTuningError("XGBoost or Optuna not available")


def reliability_diagram(y, model_prob, pos_label, outputFile, **kwargs):
    """
    BACKWARD COMPATIBILITY: Create reliability diagram (calibration plot).
    
    Args:
        y: True binary labels
        model_prob: Predicted probabilities
        pos_label: Positive class label
        outputFile: Output file path
        **kwargs: Additional plotting parameters
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available")
        return
        
    from sklearn.calibration import calibration_curve
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y == pos_label, model_prob, n_bins=10
    )
    
    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Model')
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram (Calibration Plot)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if outputFile:
        plt.savefig(outputFile, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def pdp_plot_batch(X, umodel, sel_features):
    """
    BACKWARD COMPATIBILITY: Create partial dependence plots for selected features.
    
    Args:
        X: Feature matrix
        umodel: Fitted model
        sel_features: List of feature indices or names to plot
    """
    try:
        from sklearn.inspection import PartialDependenceDisplay
        
        if not PLOTTING_AVAILABLE:
            print("Plotting libraries not available")
            return
            
        # Create partial dependence plots
        fig, ax = plt.subplots(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(
            umodel, X, sel_features, ax=ax, grid_resolution=50
        )
        plt.suptitle('Partial Dependence Plots')
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("sklearn.inspection not available for partial dependence plots")


def pca_explainedVar(pcaML, output_Folder=None, plot_type='cumulative'):
    """
    BACKWARD COMPATIBILITY: Plot PCA explained variance.
    
    Args:
        pcaML: Fitted PCA model
        output_Folder: Optional output folder path
        plot_type: Type of plot ('cumulative' or 'individual')
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available")
        return
        
    explained_variance_ratio = pcaML.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'cumulative':
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Cumulative Explained Variance')
        plt.axhline(y=0.95, color='red', linestyle='--', label='95% Variance')
        plt.legend()
    else:
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Individual Component Variance')
    
    plt.grid(alpha=0.3)
    
    if output_Folder:
        output_file = f"{output_Folder}/pca_explained_variance_{plot_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def pca_ortho_rotation(lam, method='varimax'):
    """
    BACKWARD COMPATIBILITY: Perform orthogonal rotation of PCA loadings.
    
    Args:
        lam: Loading matrix
        method: Rotation method (only 'varimax' supported)
        
    Returns:
        Rotated loadings matrix
    """
    # Simple varimax rotation implementation
    def varimax_rotation(loadings, normalize=True, max_iter=1000, tol=1e-6):
        n_vars, n_factors = loadings.shape
        
        if normalize:
            # Kaiser normalization
            h2 = np.sum(loadings**2, axis=1)
            loadings = loadings / np.sqrt(h2[:, np.newaxis])
        
        rotation_matrix = np.eye(n_factors)
        
        for _ in range(max_iter):
            old_rotation = rotation_matrix.copy()
            
            # Perform rotation
            lambda_rot = loadings @ rotation_matrix
            
            for i in range(n_factors):
                for j in range(i + 1, n_factors):
                    # Calculate rotation angle
                    a = lambda_rot[:, i]
                    b = lambda_rot[:, j]
                    
                    u = a**2 - b**2
                    v = 2 * a * b
                    
                    A = np.sum(u)
                    B = np.sum(v)
                    C = np.sum(u**2) - np.sum(v**2)
                    D = 2 * np.sum(u * v)
                    
                    if C == 0 and D == 0:
                        continue
                        
                    angle = 0.25 * np.arctan2(D, C)
                    
                    # Apply rotation
                    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
                    rotation_2d = np.array([[cos_angle, -sin_angle],
                                          [sin_angle, cos_angle]])
                    
                    full_rotation = np.eye(n_factors)
                    full_rotation[np.ix_([i, j], [i, j])] = rotation_2d
                    rotation_matrix = rotation_matrix @ full_rotation
            
            # Check convergence
            if np.allclose(rotation_matrix, old_rotation, atol=tol):
                break
        
        rotated_loadings = loadings @ rotation_matrix
        
        if normalize:
            # Reverse Kaiser normalization
            rotated_loadings = rotated_loadings * np.sqrt(h2[:, np.newaxis])
        
        return rotated_loadings
    
    if method.lower() == 'varimax':
        return varimax_rotation(lam)
    else:
        raise ValueError(f"Rotation method '{method}' not supported")


def pca_important_features(transformed_features, components_, columns):
    """
    BACKWARD COMPATIBILITY: Identify important features for each principal component.
    
    Args:
        transformed_features: PCA transformed data
        components_: PCA components matrix
        columns: Original feature names
        
    Returns:
        Dictionary mapping PC names to important features
    """
    important_features = {}
    
    for i, component in enumerate(components_):
        # Get absolute loadings
        abs_loadings = np.abs(component)
        
        # Get top features (top 20% or at least top 3)
        n_top = max(3, int(len(abs_loadings) * 0.2))
        top_indices = np.argsort(abs_loadings)[-n_top:][::-1]
        
        important_features[f'PC{i+1}'] = [
            (columns[idx], component[idx]) for idx in top_indices
        ]
    
    return important_features


def pc_draw_vectors(transformed_features, components_, columns):
    """
    BACKWARD COMPATIBILITY: Draw loading vectors on PCA biplot.
    
    Args:
        transformed_features: PCA transformed data
        components_: PCA components matrix
        columns: Original feature names
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available")
        return
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot transformed data points
    ax.scatter(transformed_features[:, 0], transformed_features[:, 1], alpha=0.6)
    
    # Draw loading vectors
    for i, (feature_name, pc1_loading, pc2_loading) in enumerate(
        zip(columns, components_[0], components_[1])
    ):
        ax.arrow(0, 0, pc1_loading*3, pc2_loading*3, 
                head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax.text(pc1_loading*3.2, pc2_loading*3.2, feature_name, 
               fontsize=10, ha='center', va='center')
    
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('PCA Biplot with Loading Vectors')
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def cca_batch(X1_sub, X2_sub, n_comp=2):
    """
    BACKWARD COMPATIBILITY: Perform Canonical Correlation Analysis.
    
    Args:
        X1_sub: First dataset
        X2_sub: Second dataset
        n_comp: Number of components
        
    Returns:
        CCA results dictionary
    """
    try:
        from sklearn.cross_decomposition import CCA
        
        cca = CCA(n_components=n_comp)
        X1_c, X2_c = cca.fit_transform(X1_sub, X2_sub)
        
        # Calculate canonical correlations
        correlations = []
        for i in range(n_comp):
            corr = np.corrcoef(X1_c[:, i], X2_c[:, i])[0, 1]
            correlations.append(corr)
        
        return {
            'X1_canonical': X1_c,
            'X2_canonical': X2_c,
            'canonical_correlations': correlations,
            'cca_model': cca
        }
    except ImportError:
        raise DataProcessingError("sklearn.cross_decomposition not available")


def canonical_correlation_analysis(first_dataset, second_dataset, num_components=2):
    """
    BACKWARD COMPATIBILITY: Extended Canonical Correlation Analysis.
    
    Args:
        first_dataset: First dataset
        second_dataset: Second dataset
        num_components: Number of components
        
    Returns:
        Comprehensive CCA results
    """
    return cca_batch(first_dataset, second_dataset, num_components)


def unify_cols(df1, df2, df1_name, df2_name):
    """
    BACKWARD COMPATIBILITY: Unify columns between two DataFrames.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        df1_name: Name for first DataFrame
        df2_name: Name for second DataFrame
        
    Returns:
        Tuple of unified DataFrames
    """
    # Get common columns
    common_cols = list(set(df1.columns) & set(df2.columns))
    
    if not common_cols:
        print(f"Warning: No common columns between {df1_name} and {df2_name}")
        return df1, df2
    
    # Select only common columns
    df1_unified = df1[common_cols].copy()
    df2_unified = df2[common_cols].copy()
    
    # Add source identifier
    df1_unified['source'] = df1_name
    df2_unified['source'] = df2_name
    
    print(f"Unified to {len(common_cols)} common columns")
    return df1_unified, df2_unified


def split_multiLabel_data__index(X, y, test_size, random_state=None):
    """
    BACKWARD COMPATIBILITY: Split multi-label data returning indices.
    
    Args:
        X: Feature matrix
        y: Multi-label target matrix
        test_size: Test set proportion
        random_state: Random state for reproducibility
        
    Returns:
        Train and test indices
    """
    try:
        from skmultilearn.model_selection import iterative_train_test_split
        
        # Get indices instead of actual data
        indices = np.arange(len(X))
        train_indices, test_indices, _, _ = iterative_train_test_split(
            indices.reshape(-1, 1), y, test_size=test_size
        )
        
        return train_indices.flatten(), test_indices.flatten()
    except ImportError:
        # Fallback to regular train_test_split
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(X))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        return train_indices, test_indices


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def ridge_lasso_notes():
    """
    BACKWARD COMPATIBILITY: Return notes about Ridge and Lasso regression.
    
    Returns:
        String with Ridge/Lasso regression notes
    """
    return """
    Ridge and Lasso Regression Notes:
    
    Ridge Regression (L2 regularization):
    - Adds penalty term: λ * sum(β²)
    - Shrinks coefficients towards zero but doesn't make them exactly zero
    - Good for handling multicollinearity
    - All features remain in the model
    
    Lasso Regression (L1 regularization):
    - Adds penalty term: λ * sum(|β|)
    - Can make coefficients exactly zero (feature selection)
    - Produces sparse models
    - Good for feature selection
    
    ElasticNet:
    - Combines L1 and L2 penalties
    - Balances feature selection and coefficient shrinkage
    - Good when you have many correlated features
    
    Key differences for regularized models in assumption checking:
    - Multicollinearity is less problematic due to regularization
    - VIF values > 10 are more acceptable
    - Feature selection happens automatically with Lasso
    """


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Module-level documentation
__version__ = "2.0.0"
__author__ = "Refactored by Claude Code"
__description__ = """
Machine Learning Toolbox - Object-Oriented Refactored Version

A comprehensive machine learning utilities library providing:
- Model performance evaluation and comparison
- Automated model template generation  
- Advanced visualization capabilities
- Feature analysis and interpretation tools
- Hyperparameter optimization with Optuna
- Data preprocessing and utilities
- Complete training pipelines

Key Classes:
- ModelPerformanceAnalyzer: Comprehensive model evaluation
- ModelTemplateFactory: Pre-configured model templates
- MLVisualizationEngine: ML-specific visualizations
- FeatureAnalysisToolkit: Feature importance and SHAP analysis
- HyperparameterOptimizer: Optuna-based optimization
- DataProcessingUtilities: Data preprocessing tools
- ModelTrainingPipeline: End-to-end training orchestration
- LinearRegressionAssumptionsChecker: Regression diagnostics

The module maintains backward compatibility with the original procedural API
while providing a clean, object-oriented interface for new development.

Example Usage:
    # New OOP API
    >>> pipeline = ModelTrainingPipeline()
    >>> results = pipeline.run_classification_pipeline(X, y, ['rf', 'xgboost'])
    
    # Original API (backward compatible)
    >>> classifiers = classifiers_template(y)
    >>> scores = ml_scores(y_model, ['accuracy', 'f1'])
"""

if __name__ == "__main__":
    print(f"ML Toolbox v{__version__}")
    print(__description__)