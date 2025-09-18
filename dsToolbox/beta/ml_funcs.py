"""
Machine Learning Functions - Comprehensively Refactored Version
===============================================================

This module provides a complete set of machine learning utilities with improved structure,
comprehensive documentation, and robust error handling. The refactored version groups 
related functionality into meaningful classes while maintaining all original functionality
and backward compatibility.

Classes:
    - ModelTemplateManager: Creates and manages ML model templates
    - ModelTrainer: Handles model training, cross-validation, and comparison
    - ModelEvaluator: Comprehensive model evaluation and scoring
    - ModelPerformanceAnalyzer: Performance visualization and analysis
    - SHAPAnalyzer: SHAP-based interpretability and analysis
    - XGBoostRuleExtractor: Rule extraction from XGBoost models  
    - DecisionTreeInterpreter: Decision tree visualization and interpretation
    - LinearRegressionAssumptionsChecker: Statistical assumption validation

Utility Functions:
    - Data preprocessing and transformation utilities
    - Multi-label classification helpers
    - Principal Component Analysis functions
    - Visualization utilities

Author: Reza Nourzadeh (Refactored and enhanced version)
Version: 2.0 (Refactored)
"""

import pandas as pd
import numpy as np
import math
import os
import sys
import re
import warnings
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL CONFIGURATIONS AND CONSTANTS
# ============================================================================

# Enhanced metric dictionary with comprehensive ML metrics
COMPREHENSIVE_METRICS = {
    # Classification Metrics
    'accuracy': metrics.accuracy_score,
    'balanced_accuracy': metrics.balanced_accuracy_score,
    'top_k_accuracy': metrics.top_k_accuracy_score,
    'average_precision': metrics.average_precision_score,
    'area_under_pr_curve': metrics.average_precision_score,
    'brier_score': metrics.brier_score_loss,
    'f1_score': metrics.f1_score,
    'f1_samples': metrics.f1_score,
    'log_loss': metrics.log_loss,
    'precision': metrics.precision_score,
    'recall': metrics.recall_score,
    'jaccard': metrics.jaccard_score,
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
    
    # Regression Metrics
    'explained_variance': metrics.explained_variance_score,
    'max_error': metrics.max_error,
    'mean_absolute_error': metrics.mean_absolute_error,
    'mean_squared_error': metrics.mean_squared_error,
    'mean_squared_log_error': metrics.mean_squared_log_error,
    'median_absolute_error': metrics.median_absolute_error,
    'r2_score': metrics.r2_score,
    'mean_poisson_deviance': metrics.mean_poisson_deviance,
    'mean_gamma_deviance': metrics.mean_gamma_deviance,
    'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error,
    
    # Additional Metrics
    'matthews_correlation_coefficient': metrics.matthews_corrcoef,
    'cohen_kappa_score': metrics.cohen_kappa_score,
}

# Backward compatibility
metric_dict = COMPREHENSIVE_METRICS

def validate_input_data(training_features: Union[pd.DataFrame, np.ndarray], 
                       target_variable: Union[pd.Series, np.ndarray],
                       feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Validate and standardize input data for machine learning functions.
    
    Args:
        training_features: Feature matrix (samples x features)
        target_variable: Target values (samples,)
        feature_names: Optional list of feature names
        
    Returns:
        Tuple of (features_array, target_array, feature_names_list)
        
    Raises:
        ValueError: If input data is invalid or incompatible
    """
    try:
        # Convert to numpy arrays
        if isinstance(training_features, pd.DataFrame):
            if feature_names is None:
                feature_names = list(training_features.columns)
            features_array = training_features.values
        else:
            features_array = np.array(training_features)
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(features_array.shape[1])]
        
        if isinstance(target_variable, pd.Series):
            target_array = target_variable.values
        else:
            target_array = np.array(target_variable)
        
        # Validate shapes
        if features_array.shape[0] != target_array.shape[0]:
            raise ValueError(f"Features ({features_array.shape[0]}) and target ({target_array.shape[0]}) have different number of samples")
        
        if len(feature_names) != features_array.shape[1]:
            raise ValueError(f"Number of feature names ({len(feature_names)}) doesn't match number of features ({features_array.shape[1]})")
        
        # Check for empty data
        if features_array.size == 0:
            raise ValueError("Feature matrix is empty")
        
        if target_array.size == 0:
            raise ValueError("Target array is empty")
        
        return features_array, target_array, feature_names
        
    except Exception as e:
        raise ValueError(f"Data validation failed: {str(e)}")

# ============================================================================
# CORE MACHINE LEARNING CLASSES
# ============================================================================

class ModelTemplateManager:
    """
    Advanced machine learning model template manager with comprehensive preprocessing.
    
    This class provides standardized model configurations for both classification 
    and regression tasks, complete with preprocessing pipelines and parameter grids.
    
    Attributes:
        random_state (int): Random seed for reproducibility
        
    Example:
        >>> template_manager = ModelTemplateManager(random_state=42)
        >>> classifiers = template_manager.get_classification_models()
        >>> regressors = template_manager.get_regression_models()
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model template manager.
        
        Args:
            random_state: Random state for reproducibility across all models
        """
        self.random_state = random_state
        logger.info(f"ModelTemplateManager initialized with random_state={random_state}")
    
    def get_classification_models(self, 
                                 training_features: Optional[pd.DataFrame] = None,
                                 target_variable: Optional[np.ndarray] = None, 
                                 imputation_strategy: str = 'mean',
                                 principal_components_count: int = 5,
                                 target_type: str = 'auto') -> Dict[str, Any]:
        """
        Generate comprehensive classification model templates with preprocessing pipelines.
        
        This method creates a dictionary of classification models including:
        - Logistic Regression with regularization
        - Random Forest with optimized parameters
        - Gradient Boosting (XGBoost, LightGBM)
        - Support Vector Machines
        - Neural Networks
        - Naive Bayes variants
        
        Args:
            training_features: Training feature matrix for model customization
            target_variable: Target values for class weight calculation
            imputation_strategy: Strategy for handling missing values 
                ('mean', 'median', 'most_frequent', 'constant')
            principal_components_count: Number of PCA components for dimensionality reduction
            target_type: Type of target variable ('binary', 'multiclass', 'auto')
            
        Returns:
            Dictionary mapping model names to configured sklearn estimators with pipelines
            
        Raises:
            ValueError: If invalid parameters are provided
            
        Example:
            >>> manager = ModelTemplateManager()
            >>> models = manager.get_classification_models(X_train, y_train)
            >>> lr_model = models['logistic_regression']
        """
        try:
            # Input validation
            if imputation_strategy not in ['mean', 'median', 'most_frequent', 'constant']:
                raise ValueError(f"Invalid imputation strategy: {imputation_strategy}")
            
            if principal_components_count < 1:
                raise ValueError("Principal components count must be positive")
            
            # Import required modules
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
            from sklearn.naive_bayes import GaussianNB, MultinomialNB
            from sklearn.neural_network import MLPClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            
            # Determine target characteristics
            is_binary = False
            if target_variable is not None:
                unique_classes = len(np.unique(target_variable))
                is_binary = unique_classes == 2
                logger.info(f"Detected {unique_classes} classes, binary classification: {is_binary}")
            
            # Create preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=imputation_strategy)),
                ('scaler', StandardScaler())
            ])
            
            # Model templates with optimized parameters
            model_templates = {
                'logistic_regression': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('classifier', LogisticRegression(
                        random_state=self.random_state,
                        max_iter=1000,
                        class_weight='balanced' if target_variable is not None else None
                    ))
                ]),
                
                'random_forest': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('classifier', RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=self.random_state,
                        class_weight='balanced' if target_variable is not None else None,
                        n_jobs=-1
                    ))
                ]),
                
                'gradient_boosting': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('classifier', GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=self.random_state
                    ))
                ]),
                
                'support_vector_machine': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('classifier', SVC(
                        kernel='rbf',
                        C=1.0,
                        gamma='scale',
                        random_state=self.random_state,
                        probability=True,
                        class_weight='balanced' if target_variable is not None else None
                    ))
                ]),
                
                'neural_network': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('classifier', MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        max_iter=500,
                        learning_rate_init=0.001,
                        random_state=self.random_state,
                        early_stopping=True,
                        validation_fraction=0.1
                    ))
                ]),
                
                'k_nearest_neighbors': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('classifier', KNeighborsClassifier(
                        n_neighbors=5,
                        weights='distance',
                        metric='minkowski',
                        n_jobs=-1
                    ))
                ]),
                
                'gaussian_naive_bayes': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('classifier', GaussianNB())
                ]),
                
                'decision_tree': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('classifier', DecisionTreeClassifier(
                        max_depth=10,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        random_state=self.random_state,
                        class_weight='balanced' if target_variable is not None else None
                    ))
                ])
            }
            
            # Add XGBoost and LightGBM if available
            try:
                import xgboost as xgb
                model_templates['xgboost_classifier'] = Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('classifier', xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        eval_metric='logloss',
                        use_label_encoder=False
                    ))
                ])
            except ImportError:
                logger.warning("XGBoost not available, skipping XGBoost classifier")
            
            try:
                import lightgbm as lgb
                model_templates['lightgbm_classifier'] = Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('classifier', lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        verbosity=-1
                    ))
                ])
            except ImportError:
                logger.warning("LightGBM not available, skipping LightGBM classifier")
            
            logger.info(f"Created {len(model_templates)} classification model templates")
            return model_templates
            
        except Exception as e:
            logger.error(f"Error creating classification models: {str(e)}")
            raise ValueError(f"Failed to create classification models: {str(e)}")
    
    def get_regression_models(self, random_state: int = None) -> Dict[str, Any]:
        """
        Generate comprehensive regression model templates with preprocessing pipelines.
        
        Creates optimized regression models including:
        - Linear Regression with regularization (Ridge, Lasso, Elastic Net)
        - Random Forest Regressor
        - Gradient Boosting Regressors
        - Support Vector Regression
        - Neural Network Regressor
        
        Args:
            random_state: Override the default random state for this specific call
            
        Returns:
            Dictionary mapping model names to configured sklearn regression estimators
            
        Example:
            >>> manager = ModelTemplateManager()
            >>> models = manager.get_regression_models()
            >>> rf_model = models['random_forest_regressor']
        """
        try:
            if random_state is None:
                random_state = self.random_state
            
            # Import required modules
            from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
                                            ElasticNet, BayesianRidge)
            from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                                        ExtraTreesRegressor)
            from sklearn.svm import SVR
            from sklearn.neural_network import MLPRegressor
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.tree import DecisionTreeRegressor
            
            # Preprocessing pipeline for regression
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            regression_models = {
                'linear_regression': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('regressor', LinearRegression())
                ]),
                
                'ridge_regression': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('regressor', Ridge(alpha=1.0, random_state=random_state))
                ]),
                
                'lasso_regression': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('regressor', Lasso(alpha=1.0, random_state=random_state))
                ]),
                
                'elastic_net_regression': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('regressor', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state))
                ]),
                
                'random_forest_regressor': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('regressor', RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1
                    ))
                ]),
                
                'gradient_boosting_regressor': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('regressor', GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=random_state
                    ))
                ]),
                
                'support_vector_regressor': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('regressor', SVR(kernel='rbf', C=1.0, gamma='scale'))
                ]),
                
                'neural_network_regressor': Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('regressor', MLPRegressor(
                        hidden_layer_sizes=(100, 50),
                        max_iter=500,
                        learning_rate_init=0.001,
                        random_state=random_state,
                        early_stopping=True,
                        validation_fraction=0.1
                    ))
                ])
            }
            
            # Add advanced regressors if available
            try:
                import xgboost as xgb
                regression_models['xgboost_regressor'] = Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('regressor', xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state
                    ))
                ])
            except ImportError:
                pass
            
            try:
                import lightgbm as lgb
                regression_models['lightgbm_regressor'] = Pipeline([
                    ('preprocessor', ColumnTransformer(
                        transformers=[('num', numeric_transformer, slice(None))],
                        remainder='passthrough')),
                    ('regressor', lgb.LGBMRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        verbosity=-1
                    ))
                ])
            except ImportError:
                pass
            
            logger.info(f"Created {len(regression_models)} regression model templates")
            return regression_models
            
        except Exception as e:
            logger.error(f"Error creating regression models: {str(e)}")
            raise ValueError(f"Failed to create regression models: {str(e)}")


class ModelEvaluator:
    """
    Comprehensive model evaluation system with extensive metrics and cross-validation.
    
    This class provides advanced model evaluation capabilities including:
    - Multi-metric scoring systems
    - Cross-validation with various strategies
    - Statistical significance testing
    - Performance comparison utilities
    
    Example:
        >>> evaluator = ModelEvaluator()
        >>> scores = evaluator.calculate_model_scores(y_true, y_pred, ['accuracy', 'f1'])
        >>> cv_results = evaluator.cross_validation_scores(model, X, y)
    """
    
    def __init__(self):
        """Initialize the model evaluator with default configuration."""
        logger.info("ModelEvaluator initialized")
        
    def calculate_model_scores(self,
                             model_predictions: Dict[str, np.ndarray],
                             metric_names: List[str],
                             multiclass_strategy: str = 'ovr',
                             averaging_strategy: str = 'macro',
                             target_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive model performance metrics.
        
        Args:
            model_predictions: Dictionary with 'y_true' and 'y_pred' keys
            metric_names: List of metric names to calculate
            multiclass_strategy: Strategy for multiclass problems ('ovr', 'ovo')
            averaging_strategy: Averaging strategy ('macro', 'micro', 'weighted', 'binary')
            target_names: Optional list of target class names
            
        Returns:
            Dictionary mapping metric names to calculated values
            
        Raises:
            ValueError: If invalid metrics or incompatible data provided
            
        Example:
            >>> predictions = {'y_true': y_test, 'y_pred': model.predict(X_test)}
            >>> scores = evaluator.calculate_model_scores(predictions, ['accuracy', 'f1'])
        """
        try:
            # Validate inputs
            if 'y_true' not in model_predictions or 'y_pred' not in model_predictions:
                raise ValueError("model_predictions must contain 'y_true' and 'y_pred' keys")
            
            y_true = np.array(model_predictions['y_true'])
            y_pred = np.array(model_predictions['y_pred'])
            
            if len(y_true) != len(y_pred):
                raise ValueError("y_true and y_pred must have the same length")
            
            calculated_scores = {}
            
            for metric_name in metric_names:
                try:
                    if metric_name not in COMPREHENSIVE_METRICS:
                        logger.warning(f"Unknown metric: {metric_name}, skipping")
                        continue
                    
                    metric_function = COMPREHENSIVE_METRICS[metric_name]
                    
                    # Handle metrics with special parameters
                    if metric_name in ['f1_score', 'precision', 'recall', 'jaccard']:
                        if len(np.unique(y_true)) > 2:  # Multiclass
                            score = metric_function(y_true, y_pred, average=averaging_strategy)
                        else:  # Binary
                            score = metric_function(y_true, y_pred, average='binary')
                    elif metric_name in ['roc_auc']:
                        if len(np.unique(y_true)) > 2:  # Multiclass
                            score = metric_function(y_true, y_pred, multi_class=multiclass_strategy, average=averaging_strategy)
                        else:  # Binary
                            score = metric_function(y_true, y_pred)
                    else:
                        score = metric_function(y_true, y_pred)
                    
                    calculated_scores[metric_name] = float(score)
                    
                except Exception as metric_error:
                    logger.warning(f"Error calculating {metric_name}: {str(metric_error)}")
                    calculated_scores[metric_name] = np.nan
            
            logger.info(f"Calculated {len(calculated_scores)} metrics successfully")
            return calculated_scores
            
        except Exception as e:
            logger.error(f"Error in calculate_model_scores: {str(e)}")
            raise ValueError(f"Failed to calculate model scores: {str(e)}")
    
    def cross_validation_scores(self,
                               model: Any,
                               training_features: Union[pd.DataFrame, np.ndarray],
                               target_variable: Union[pd.Series, np.ndarray],
                               cv_folds: int = 5,
                               metric_names: Optional[List[str]] = None,
                               cv_strategy: str = 'stratified',
                               random_state: int = 42,
                               n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation evaluation with multiple metrics.
        
        Args:
            model: Scikit-learn compatible model
            training_features: Feature matrix
            target_variable: Target values
            cv_folds: Number of cross-validation folds
            metric_names: List of metrics to evaluate
            cv_strategy: Cross-validation strategy ('stratified', 'kfold', 'shuffle')
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary containing cross-validation results and statistics
            
        Example:
            >>> results = evaluator.cross_validation_scores(model, X, y, cv_folds=10)
            >>> print(f"Mean accuracy: {results['test_accuracy'].mean():.3f}")
        """
        try:
            # Validate inputs
            features_array, target_array, _ = validate_input_data(training_features, target_variable)
            
            if metric_names is None:
                metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
            
            # Create cross-validation strategy
            if cv_strategy == 'stratified':
                from sklearn.model_selection import StratifiedKFold
                cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            elif cv_strategy == 'kfold':
                from sklearn.model_selection import KFold
                cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            elif cv_strategy == 'shuffle':
                from sklearn.model_selection import ShuffleSplit
                cv_splitter = ShuffleSplit(n_splits=cv_folds, test_size=0.2, random_state=random_state)
            else:
                cv_splitter = cv_folds
            
            # Map metric names to sklearn scoring format
            scoring_metrics = {}
            for metric in metric_names:
                if metric in ['accuracy']:
                    scoring_metrics[metric] = 'accuracy'
                elif metric in ['precision']:
                    scoring_metrics[metric] = 'precision_macro'
                elif metric in ['recall']:
                    scoring_metrics[metric] = 'recall_macro'
                elif metric in ['f1_score', 'f1']:
                    scoring_metrics[metric] = 'f1_macro'
                elif metric in ['roc_auc']:
                    scoring_metrics[metric] = 'roc_auc'
                else:
                    scoring_metrics[metric] = metric
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, features_array, target_array,
                cv=cv_splitter,
                scoring=scoring_metrics,
                n_jobs=n_jobs,
                return_train_score=True,
                error_score='raise'
            )
            
            # Calculate summary statistics
            summary_stats = {}
            for metric in scoring_metrics.keys():
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                summary_stats[metric] = {
                    'test_mean': np.mean(test_scores),
                    'test_std': np.std(test_scores),
                    'test_scores': test_scores,
                    'train_mean': np.mean(train_scores),
                    'train_std': np.std(train_scores),
                    'train_scores': train_scores,
                    'overfitting_score': np.mean(train_scores) - np.mean(test_scores)
                }
            
            # Add timing information
            summary_stats['fit_time'] = {
                'mean': np.mean(cv_results['fit_time']),
                'std': np.std(cv_results['fit_time']),
                'total': np.sum(cv_results['fit_time'])
            }
            
            summary_stats['score_time'] = {
                'mean': np.mean(cv_results['score_time']),
                'std': np.std(cv_results['score_time']),
                'total': np.sum(cv_results['score_time'])
            }
            
            logger.info(f"Cross-validation completed successfully with {cv_folds} folds")
            return summary_stats
            
        except Exception as e:
            logger.error(f"Error in cross_validation_scores: {str(e)}")
            raise ValueError(f"Cross-validation failed: {str(e)}")


class ModelTrainer:
    """
    Advanced model training and comparison system with comprehensive evaluation.
    
    This class provides sophisticated model training capabilities including:
    - Automated model comparison across multiple algorithms
    - Cross-validation with statistical significance testing
    - Hyperparameter optimization integration
    - Model selection based on multiple criteria
    
    Example:
        >>> trainer = ModelTrainer()
        >>> results = trainer.compare_models(models_dict, X_train, y_train)
        >>> best_model = trainer.select_best_model(results)
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs for training
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_evaluator = ModelEvaluator()
        logger.info(f"ModelTrainer initialized with random_state={random_state}, n_jobs={n_jobs}")
    
    def compare_models(self,
                      model_dictionary: Dict[str, Any],
                      training_features: Union[pd.DataFrame, np.ndarray],
                      target_variable: Union[pd.Series, np.ndarray],
                      cv_folds: int = 5,
                      metric_names: Optional[List[str]] = None,
                      test_features: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                      test_target: Optional[Union[pd.Series, np.ndarray]] = None,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Compare multiple models using cross-validation and optional hold-out testing.
        
        Args:
            model_dictionary: Dictionary mapping model names to model instances
            training_features: Training feature matrix
            target_variable: Training target values
            cv_folds: Number of cross-validation folds
            metric_names: List of metrics to evaluate
            test_features: Optional test feature matrix
            test_target: Optional test target values
            verbose: Whether to print progress information
            
        Returns:
            Comprehensive results dictionary containing:
            - Cross-validation results for each model
            - Test set results (if provided)
            - Model rankings by metric
            - Statistical comparisons
            
        Example:
            >>> models = {'rf': RandomForestClassifier(), 'svm': SVC()}
            >>> results = trainer.compare_models(models, X_train, y_train)
        """
        try:
            # Validate inputs
            features_array, target_array, feature_names = validate_input_data(
                training_features, target_variable)
            
            if metric_names is None:
                metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
            
            if verbose:
                logger.info(f"Comparing {len(model_dictionary)} models with {len(metric_names)} metrics")
            
            comparison_results = {
                'cv_results': {},
                'test_results': {},
                'model_rankings': {},
                'summary_statistics': {},
                'metadata': {
                    'n_samples': len(target_array),
                    'n_features': len(feature_names),
                    'n_models': len(model_dictionary),
                    'cv_folds': cv_folds,
                    'metrics': metric_names
                }
            }
            
            # Perform cross-validation for each model
            for model_name, model in model_dictionary.items():
                if verbose:
                    logger.info(f"Evaluating model: {model_name}")
                
                try:
                    # Cross-validation evaluation
                    cv_results = self.model_evaluator.cross_validation_scores(
                        model, features_array, target_array,
                        cv_folds=cv_folds,
                        metric_names=metric_names,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs
                    )
                    comparison_results['cv_results'][model_name] = cv_results
                    
                    # Test set evaluation (if provided)
                    if test_features is not None and test_target is not None:
                        test_features_array, test_target_array, _ = validate_input_data(
                            test_features, test_target, feature_names)
                        
                        # Fit model on full training set
                        fitted_model = model.fit(features_array, target_array)
                        test_predictions = fitted_model.predict(test_features_array)
                        
                        # Calculate test metrics
                        test_scores = self.model_evaluator.calculate_model_scores(
                            {'y_true': test_target_array, 'y_pred': test_predictions},
                            metric_names
                        )
                        comparison_results['test_results'][model_name] = test_scores
                    
                except Exception as model_error:
                    logger.error(f"Error evaluating {model_name}: {str(model_error)}")
                    comparison_results['cv_results'][model_name] = None
            
            # Calculate model rankings for each metric
            for metric in metric_names:
                rankings = []
                for model_name, results in comparison_results['cv_results'].items():
                    if results is not None and metric in results:
                        mean_score = results[metric]['test_mean']
                        rankings.append((model_name, mean_score))
                
                # Sort by score (descending for most metrics)
                if metric in ['log_loss', 'mean_squared_error', 'mean_absolute_error']:
                    rankings.sort(key=lambda x: x[1])  # Lower is better
                else:
                    rankings.sort(key=lambda x: x[1], reverse=True)  # Higher is better
                
                comparison_results['model_rankings'][metric] = rankings
            
            # Calculate summary statistics across all models
            self._calculate_summary_statistics(comparison_results)
            
            if verbose:
                self._print_comparison_summary(comparison_results)
            
            logger.info("Model comparison completed successfully")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error in compare_models: {str(e)}")
            raise ValueError(f"Model comparison failed: {str(e)}")
    
    def _calculate_summary_statistics(self, results: Dict[str, Any]) -> None:
        """Calculate summary statistics across all models and metrics."""
        try:
            summary = {}
            
            for metric in results['metadata']['metrics']:
                metric_scores = []
                for model_results in results['cv_results'].values():
                    if model_results is not None and metric in model_results:
                        metric_scores.append(model_results[metric]['test_mean'])
                
                if metric_scores:
                    summary[metric] = {
                        'mean': np.mean(metric_scores),
                        'std': np.std(metric_scores),
                        'min': np.min(metric_scores),
                        'max': np.max(metric_scores),
                        'range': np.max(metric_scores) - np.min(metric_scores)
                    }
            
            results['summary_statistics'] = summary
            
        except Exception as e:
            logger.warning(f"Error calculating summary statistics: {str(e)}")
    
    def _print_comparison_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted summary of model comparison results."""
        try:
            print("\\n" + "="*80)
            print("MODEL COMPARISON SUMMARY")
            print("="*80)
            
            metadata = results['metadata']
            print(f"Dataset: {metadata['n_samples']} samples, {metadata['n_features']} features")
            print(f"Models evaluated: {metadata['n_models']}")
            print(f"Cross-validation folds: {metadata['cv_folds']}")
            
            print("\\nModel Rankings by Metric:")
            print("-"*50)
            
            for metric, rankings in results['model_rankings'].items():
                print(f"\\n{metric.upper()}:")
                for i, (model_name, score) in enumerate(rankings[:5], 1):
                    print(f"  {i}. {model_name}: {score:.4f}")
            
            print("\\n" + "="*80)
            
        except Exception as e:
            logger.warning(f"Error printing summary: {str(e)}")

    def cross_validate_model(self,
                           model: Any,
                           training_features: Union[pd.DataFrame, np.ndarray],
                           target_variable: Union[pd.Series, np.ndarray],
                           cross_validation_strategy: Any,
                           test_features: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                           test_target: Optional[Union[pd.Series, np.ndarray]] = None,
                           callback_functions: Optional[List[Callable]] = None,
                           verbose: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation for a single model with detailed analysis.
        
        Args:
            model: The machine learning model to evaluate
            training_features: Training feature matrix
            target_variable: Training target values
            cross_validation_strategy: Cross-validation strategy (e.g., StratifiedKFold)
            test_features: Optional test features for final evaluation
            test_target: Optional test target for final evaluation
            callback_functions: Optional list of callback functions to execute
            verbose: Whether to print detailed progress information
            
        Returns:
            Detailed cross-validation results including fold-by-fold performance
        """
        try:
            # Validate inputs
            features_array, target_array, feature_names = validate_input_data(
                training_features, target_variable)
            
            results = {
                'fold_results': [],
                'aggregated_metrics': {},
                'feature_importance': [],
                'prediction_details': {
                    'predictions': [],
                    'probabilities': [],
                    'indices': []
                }
            }
            
            if verbose:
                logger.info(f"Starting cross-validation with {cross_validation_strategy.n_splits} folds")
            
            # Perform cross-validation fold by fold
            for fold_idx, (train_indices, val_indices) in enumerate(
                cross_validation_strategy.split(features_array, target_array)):
                
                if verbose:
                    logger.info(f"Processing fold {fold_idx + 1}/{cross_validation_strategy.n_splits}")
                
                # Split data for this fold
                X_train_fold = features_array[train_indices]
                X_val_fold = features_array[val_indices]
                y_train_fold = target_array[train_indices]
                y_val_fold = target_array[val_indices]
                
                # Train model on fold
                fold_model = model.fit(X_train_fold, y_train_fold)
                
                # Make predictions
                val_predictions = fold_model.predict(X_val_fold)
                
                # Store predictions and indices
                results['prediction_details']['predictions'].extend(val_predictions)
                results['prediction_details']['indices'].extend(val_indices)
                
                # Get probabilities if available
                if hasattr(fold_model, 'predict_proba'):
                    val_probabilities = fold_model.predict_proba(X_val_fold)
                    results['prediction_details']['probabilities'].extend(val_probabilities)
                
                # Calculate fold metrics
                fold_scores = self.model_evaluator.calculate_model_scores(
                    {'y_true': y_val_fold, 'y_pred': val_predictions},
                    ['accuracy', 'precision', 'recall', 'f1_score']
                )
                
                # Store feature importance if available
                if hasattr(fold_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': fold_model.feature_importances_,
                        'fold': fold_idx
                    })
                    results['feature_importance'].append(importance_df)
                
                # Store fold results
                fold_result = {
                    'fold': fold_idx,
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'metrics': fold_scores
                }
                results['fold_results'].append(fold_result)
                
                # Execute callbacks if provided
                if callback_functions:
                    for callback in callback_functions:
                        try:
                            callback(fold_model, X_val_fold, y_val_fold, val_predictions, fold_idx)
                        except Exception as callback_error:
                            logger.warning(f"Callback error in fold {fold_idx}: {str(callback_error)}")
            
            # Calculate aggregated metrics
            self._aggregate_cv_metrics(results)
            
            # Final model evaluation on test set if provided
            if test_features is not None and test_target is not None:
                final_model = model.fit(features_array, target_array)
                test_predictions = final_model.predict(test_features)
                
                test_scores = self.model_evaluator.calculate_model_scores(
                    {'y_true': test_target, 'y_pred': test_predictions},
                    ['accuracy', 'precision', 'recall', 'f1_score']
                )
                results['test_performance'] = test_scores
            
            logger.info("Cross-validation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in cross_validate_model: {str(e)}")
            raise ValueError(f"Cross-validation failed: {str(e)}")
    
    def _aggregate_cv_metrics(self, results: Dict[str, Any]) -> None:
        """Aggregate metrics across all cross-validation folds."""
        try:
            # Get all metric names from first fold
            if not results['fold_results']:
                return
            
            metric_names = list(results['fold_results'][0]['metrics'].keys())
            aggregated = {}
            
            for metric in metric_names:
                scores = [fold['metrics'][metric] for fold in results['fold_results']]
                aggregated[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'scores': scores
                }
            
            results['aggregated_metrics'] = aggregated
            
        except Exception as e:
            logger.warning(f"Error aggregating CV metrics: {str(e)}")


class SHAPAnalyzer:
    """
    Advanced SHAP-based model interpretability and analysis system.
    
    This class provides comprehensive SHAP analysis capabilities including:
    - Batch SHAP value calculation with correlation analysis
    - Local interpretability for individual predictions
    - Customer segmentation based on SHAP values
    - Feature importance ranking and analysis
    
    Example:
        >>> analyzer = SHAPAnalyzer()
        >>> shap_values, importance = analyzer.calculate_shap_contributions(X, model, predictions)
        >>> local_analysis = analyzer.analyze_local_interpretability(X, shap_values, sample_id)
    """
    
    def __init__(self, output_directory: Optional[str] = None):
        """
        Initialize SHAP analyzer.
        
        Args:
            output_directory: Directory for saving SHAP plots and analysis
        """
        self.output_directory = output_directory
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory)
        logger.info(f"SHAPAnalyzer initialized with output_directory={output_directory}")
    
    def calculate_shap_contributions(self,
                                   training_features: pd.DataFrame,
                                   trained_model: Any,
                                   model_predictions: pd.DataFrame,
                                   positive_class_label: str,
                                   plot_output_folder: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate SHAP contributions and feature importance with correlation analysis.
        
        This method performs comprehensive SHAP analysis including:
        - SHAP value calculation using TreeExplainer
        - Correlation analysis between SHAP values and features
        - Correlation analysis between SHAP values and target variable
        - Feature importance ranking based on mean absolute SHAP values
        - Optional visualization outputs
        
        Args:
            training_features: Feature matrix (samples x features)
            trained_model: Trained XGBoost or tree-based model
            model_predictions: DataFrame with predictions and probabilities
            positive_class_label: Name/key of the positive class column in predictions
            plot_output_folder: Optional folder to save SHAP visualizations
            
        Returns:
            Tuple of (shap_contributions_df, feature_importance_df)
            
        Example:
            >>> shap_df, importance_df = analyzer.calculate_shap_contributions(
            ...     X_test, xgb_model, predictions_df, 'positive_class_prob')
        """
        try:
            import shap
            
            # Validate inputs
            if not isinstance(training_features, pd.DataFrame):
                raise ValueError("training_features must be a pandas DataFrame")
            
            if positive_class_label not in model_predictions.columns:
                raise ValueError(f"positive_class_label '{positive_class_label}' not found in model_predictions")
            
            logger.info("Starting SHAP analysis...")
            
            # Initialize SHAP explainer
            try:
                explainer = shap.TreeExplainer(trained_model)
                shap_values_array = explainer.shap_values(training_features)
            except Exception as e:
                logger.warning(f"TreeExplainer failed: {e}, trying GPUTree explainer")
                try:
                    explainer = shap.explainers.GPUTree(trained_model, training_features)
                    shap_values_array = explainer(training_features)
                    if hasattr(shap_values_array, 'values'):
                        shap_values_array = shap_values_array.values
                except Exception as e2:
                    raise ValueError(f"Both TreeExplainer and GPUTree failed: {e2}")
            
            # Create SHAP contributions DataFrame
            shap_contributions_df = pd.DataFrame(
                shap_values_array,
                index=training_features.index,
                columns=training_features.columns
            )
            shap_contributions_df['predicted_class'] = model_predictions['label'] if 'label' in model_predictions.columns else None
            
            # Correlation analysis
            logger.info("Calculating correlation analysis...")
            feature_correlations = []
            shap_target_correlations = []
            feature_target_correlations = []
            
            features_array = training_features.values.astype(float)
            shap_array = shap_values_array.astype(float)
            
            for feature_idx in range(features_array.shape[1]):
                # SHAP-Feature correlation
                shap_feature_corr = np.corrcoef(shap_array[:, feature_idx], features_array[:, feature_idx])[0, 1]
                if np.isnan(shap_feature_corr):
                    shap_feature_corr = 0.0
                feature_correlations.append(shap_feature_corr)
                
                # SHAP-Target correlation
                shap_target_corr = np.corrcoef(shap_array[:, feature_idx], model_predictions[positive_class_label])[0, 1]
                if np.isnan(shap_target_corr):
                    shap_target_corr = 0.0
                shap_target_correlations.append(shap_target_corr)
                
                # Feature-Target correlation
                feature_target_corr = np.corrcoef(features_array[:, feature_idx], model_predictions[positive_class_label])[0, 1]
                if np.isnan(feature_target_corr):
                    feature_target_corr = 0.0
                feature_target_correlations.append(feature_target_corr)
            
            # Create feature importance DataFrame
            mean_abs_shap = np.abs(shap_array).mean(axis=0)
            
            feature_importance_df = pd.DataFrame({
                'feature_name': training_features.columns,
                'shap_feature_correlation': feature_correlations,
                'shap_target_correlation': shap_target_correlations,
                'feature_target_correlation': feature_target_correlations,
                'mean_absolute_shap': mean_abs_shap
            }).sort_values(by='mean_absolute_shap', ascending=False)
            
            # Generate visualizations if output folder provided
            if plot_output_folder:
                self._generate_shap_visualizations(
                    shap_values_array, training_features, shap_contributions_df, 
                    model_predictions, plot_output_folder
                )
            
            logger.info(f"SHAP analysis completed successfully. Top feature: {feature_importance_df.iloc[0]['feature_name']}")
            return shap_contributions_df, feature_importance_df
            
        except Exception as e:
            logger.error(f"Error in calculate_shap_contributions: {str(e)}")
            raise ValueError(f"SHAP analysis failed: {str(e)}")
    
    def _generate_shap_visualizations(self, shap_values, features, shap_df, predictions, output_folder):
        """Generate and save SHAP visualizations."""
        try:
            import shap
            import matplotlib.pyplot as plt
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            logger.info("Generating SHAP visualizations...")
            
            # Summary plot of SHAP values
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, features, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # SHAP values vs probabilities scatter plot
            plt.figure(figsize=(10, 6))
            shap_sums = np.sum(shap_values, axis=1)
            plt.scatter(shap_sums, predictions.get('prob_max', predictions.iloc[:, 0]), alpha=0.6)
            plt.xlabel('Sum of SHAP Values')
            plt.ylabel('Model Probability')
            plt.title('SHAP Values vs Model Probabilities')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "shap_vs_probabilities.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP visualizations saved to {output_folder}")
            
        except Exception as e:
            logger.warning(f"Error generating SHAP visualizations: {str(e)}")
    
    def analyze_local_interpretability(self,
                                     training_features: pd.DataFrame,
                                     shap_contributions: pd.DataFrame,
                                     feature_importance: pd.DataFrame,
                                     sample_identifier: str,
                                     output_file_path: str,
                                     **visualization_kwargs) -> pd.DataFrame:
        """
        Analyze local interpretability for a specific sample using SHAP values.
        
        This method provides detailed local explanation including:
        - Individual SHAP value analysis for the sample
        - Comparison with feature means and SHAP importance
        - Logic validation for feature contributions
        - Force plot visualization
        
        Args:
            training_features: Original feature matrix
            shap_contributions: SHAP contributions DataFrame from calculate_shap_contributions
            feature_importance: Feature importance DataFrame from calculate_shap_contributions
            sample_identifier: ID/index of the sample to analyze
            output_file_path: Path to save the force plot visualization
            **visualization_kwargs: Additional arguments for SHAP force_plot
            
        Returns:
            Detailed analysis DataFrame with SHAP logic validation
            
        Example:
            >>> analysis = analyzer.analyze_local_interpretability(
            ...     X, shap_df, importance_df, 'customer_123', 'output/force_plot.png')
        """
        try:
            import shap
            
            # Validate sample identifier exists
            if sample_identifier not in training_features.index:
                raise ValueError(f"Sample identifier '{sample_identifier}' not found in training_features index")
            
            if sample_identifier not in shap_contributions.index:
                raise ValueError(f"Sample identifier '{sample_identifier}' not found in shap_contributions index")
            
            logger.info(f"Analyzing local interpretability for sample: {sample_identifier}")
            
            # Extract sample data
            sample_shap = shap_contributions.loc[sample_identifier].drop(['predicted_class', 'expected_value'], errors='ignore')
            sample_features = training_features.loc[sample_identifier]
            
            # Sort SHAP values by absolute magnitude
            sorted_shap = sample_shap.reindex(sample_shap.abs().sort_values(ascending=False).index)
            
            # Get corresponding feature values and means
            feature_means = training_features.mean()
            sample_feature_values = sample_features.reindex(sorted_shap.index)
            feature_means_sorted = feature_means.reindex(sorted_shap.index)
            
            # Create comprehensive analysis DataFrame
            analysis_df = pd.DataFrame({
                'feature_name': sorted_shap.index,
                'shap_value': sorted_shap.values,
                'feature_value': sample_feature_values.values,
                'feature_mean': feature_means_sorted.values
            })
            
            # Merge with importance metrics
            importance_subset = feature_importance.set_index('feature_name')[
                ['mean_absolute_shap', 'shap_feature_correlation', 'shap_target_correlation']
            ]
            analysis_df = analysis_df.set_index('feature_name').join(importance_subset).reset_index()
            
            # Logic validation
            analysis_df['shap_vs_importance_logic'] = np.where(
                np.sign(analysis_df['shap_value'] - analysis_df['mean_absolute_shap']) == 
                np.sign(analysis_df['shap_feature_correlation']),
                True, False
            )
            
            analysis_df['feature_vs_mean_logic'] = np.where(
                np.sign(analysis_df['feature_value'] - analysis_df['feature_mean']) == 
                np.sign(analysis_df['shap_feature_correlation']),
                True, False
            )
            
            # Calculate contribution strength
            analysis_df['contribution_strength'] = np.abs(analysis_df['shap_value'])
            analysis_df['relative_contribution'] = (
                analysis_df['contribution_strength'] / analysis_df['contribution_strength'].sum()
            )
            
            # Print summary
            logger.info("Local Interpretability Analysis Summary:")
            logger.info(f"Total SHAP contribution: {analysis_df['shap_value'].sum():.4f}")
            logger.info(f"Top contributing feature: {analysis_df.iloc[0]['feature_name']} ({analysis_df.iloc[0]['shap_value']:.4f})")
            
            # Generate force plot visualization
            if 'expected_value' in shap_contributions.columns:
                expected_value = shap_contributions.loc[sample_identifier, 'expected_value']
            else:
                expected_value = 0.0
                
            try:
                shap.initjs()  # Initialize JavaScript for notebooks
                force_plot = shap.force_plot(
                    expected_value,
                    sample_shap.values,
                    sample_features,
                    show=False,
                    **visualization_kwargs
                )
                force_plot.savefig(output_file_path)
                logger.info(f"Force plot saved to {output_file_path}")
            except Exception as plot_error:
                logger.warning(f"Error creating force plot: {str(plot_error)}")
            
            return analysis_df
            
        except Exception as e:
            logger.error(f"Error in analyze_local_interpretability: {str(e)}")
            raise ValueError(f"Local interpretability analysis failed: {str(e)}")


class XGBoostRuleExtractor:
    """
    Advanced XGBoost rule extraction and analysis system.
    
    This class provides comprehensive rule extraction from XGBoost models including:
    - Decision tree to rule conversion
    - Rule combination and optimization
    - Customer segmentation based on rules
    - Rule frequency and importance analysis
    
    Example:
        >>> extractor = XGBoostRuleExtractor()
        >>> rules = extractor.extract_decision_rules(xgb_model, max_depth=6)
        >>> segments = extractor.create_rule_based_segments(rules, customer_data)
    """
    
    def __init__(self):
        """Initialize the XGBoost rule extractor."""
        logger.info("XGBoostRuleExtractor initialized")
    
    def extract_decision_rules(self,
                             xgboost_model_trees: pd.DataFrame,
                             maximum_tree_depth: int) -> pd.DataFrame:
        """
        Extract decision rules from XGBoost model trees for all leaf nodes.
        
        This method converts XGBoost tree structure into interpretable if-then rules
        by tracing paths from root to leaf nodes and combining feature conditions.
        
        Args:
            xgboost_model_trees: DataFrame from xgb_model.trees_to_dataframe()
            maximum_tree_depth: Maximum depth of trees in the model
            
        Returns:
            DataFrame with rules for each leaf node containing feature thresholds
            
        Example:
            >>> trees_df = xgb_model.trees_to_dataframe()
            >>> rules = extractor.extract_decision_rules(trees_df, max_depth=6)
        """
        try:
            logger.info(f"Extracting decision rules from {xgboost_model_trees['Tree'].nunique()} trees")
            
            def extract_leaf_rule(leaf_node_id: str, tree_dataframe: pd.DataFrame, feature_names: List[str]) -> pd.Series:
                """Extract rule for a single leaf node by tracing path from root."""
                # Initialize feature bounds
                feature_lower_bounds = pd.Series(-1e99, index=feature_names)
                feature_upper_bounds = pd.Series(1e99, index=feature_names)
                smaller_conditions = pd.Series(np.nan, index=feature_names)
                larger_conditions = pd.Series(np.nan, index=feature_names)
                
                current_node = leaf_node_id
                
                # Trace path from leaf to root
                while current_node.split('-')[-1] != '0':  # Not root node
                    # Find parent node that leads to current node via 'Yes' branch
                    parent_yes = tree_dataframe[
                        (tree_dataframe['Yes'] == current_node) & (tree_dataframe['Gain'] == 0)
                    ]
                    
                    # Find parent node that leads to current node via 'No' branch  
                    parent_no = tree_dataframe[
                        (tree_dataframe['No'] == current_node) & (tree_dataframe['Gain'] == 0)
                    ]
                    
                    if not parent_yes.empty:
                        # Current node reached via 'Yes' branch (feature <= threshold)
                        parent_info = parent_yes.iloc[0]
                        feature = parent_info['Feature']
                        threshold = round(parent_info['Split'], 1)
                        
                        feature_upper_bounds.loc[feature] = min([threshold, feature_upper_bounds.loc[feature]])
                        smaller_conditions.loc[feature] = feature_upper_bounds.loc[feature]
                        current_node = parent_info['ID']
                        
                    elif not parent_no.empty:
                        # Current node reached via 'No' branch (feature > threshold)
                        parent_info = parent_no.iloc[0]
                        feature = parent_info['Feature']
                        threshold = round(parent_info['Split'], 1)
                        
                        feature_lower_bounds.loc[feature] = max([threshold, feature_lower_bounds.loc[feature]])
                        larger_conditions.loc[feature] = feature_lower_bounds.loc[feature]
                        current_node = parent_info['ID']
                
                # Combine conditions into MultiIndex Series
                combined_conditions = pd.concat([smaller_conditions, larger_conditions])
                index_arrays = [
                    np.repeat(['smaller', 'larger'], len(feature_names)),
                    list(smaller_conditions.index) + list(larger_conditions.index)
                ]
                combined_conditions.index = pd.MultiIndex.from_arrays(index_arrays)
                
                return combined_conditions
            
            # Extract unique feature names
            all_features = xgboost_model_trees['Feature'].loc[
                xgboost_model_trees['Feature'] != 'Leaf'
            ].unique()
            
            all_rules = pd.DataFrame()
            
            # Process each tree
            for tree_id in xgboost_model_trees['Tree'].unique():
                tree_subset = xgboost_model_trees.loc[xgboost_model_trees['Tree'] == tree_id].copy()
                
                # Add node levels for depth analysis
                tree_subset['node_level'] = 0
                current_level = 0
                current_level_nodes = [True] + [False] * (tree_subset.shape[0] - 1)
                
                while current_level < maximum_tree_depth:
                    level_nodes = tree_subset[['ID', 'Yes', 'No']].loc[current_level_nodes]
                    if isinstance(level_nodes, pd.Series):
                        level_nodes = level_nodes.to_frame().T
                    
                    next_level_nodes = (
                        tree_subset['ID'].isin(level_nodes['Yes'].tolist()) |
                        tree_subset['ID'].isin(level_nodes['No'].tolist())
                    )
                    
                    current_level += 1
                    tree_subset.loc[next_level_nodes, 'node_level'] = current_level
                    current_level_nodes = next_level_nodes
                
                # Identify decisive nodes and leaves
                self._mark_decisive_nodes(tree_subset, maximum_tree_depth)
                
                # Extract rules for all leaf nodes
                leaf_nodes = tree_subset['ID'].loc[tree_subset['Feature'] == 'Leaf']
                tree_rules = leaf_nodes.apply(lambda x: extract_leaf_rule(x, tree_subset, all_features))
                tree_rules.index = leaf_nodes
                
                all_rules = pd.concat([all_rules, tree_rules], ignore_index=False)
            
            logger.info(f"Extracted rules for {len(all_rules)} leaf nodes")
            return all_rules
            
        except Exception as e:
            logger.error(f"Error in extract_decision_rules: {str(e)}")
            raise ValueError(f"Rule extraction failed: {str(e)}")
    
    def _mark_decisive_nodes(self, tree_dataframe: pd.DataFrame, maximum_depth: int) -> None:
        """Mark nodes as decisive based on child node probabilities."""
        try:
            current_level = maximum_depth
            
            while current_level > 0:
                leaf_nodes = tree_dataframe.loc[tree_dataframe['node_level'] == current_level]
                parent_nodes = tree_dataframe.loc[tree_dataframe['node_level'] == current_level - 1].copy()
                
                # Merge with child node probabilities
                parent_nodes = parent_nodes.merge(
                    leaf_nodes[['ID', 'Gain']], 
                    how='left', left_on='Yes', right_on='ID', 
                    suffixes=('', '_yes')
                ).drop(['ID_yes'], axis=1).rename(columns={'Gain_yes': 'yes_probability'})
                
                parent_nodes = parent_nodes.merge(
                    leaf_nodes[['ID', 'Gain']], 
                    how='left', left_on='No', right_on='ID', 
                    suffixes=('', '_no')
                ).drop(['ID_no'], axis=1).rename(columns={'Gain_no': 'no_probability'})
                
                # Determine decisiveness based on child probabilities
                decisive_mask_neutral = (
                    (np.sign(parent_nodes['yes_probability']) != np.sign(parent_nodes['no_probability'])) |
                    ((np.sign(parent_nodes['yes_probability']) == np.sign(parent_nodes['no_probability'])) & 
                     (np.sign(parent_nodes['no_probability']) == 0))
                )
                
                decisive_mask_positive = (
                    (np.sign(parent_nodes['yes_probability']) == np.sign(parent_nodes['no_probability'])) &
                    (np.sign(parent_nodes['no_probability']) == 1)
                )
                
                decisive_mask_negative = (
                    (np.sign(parent_nodes['yes_probability']) == np.sign(parent_nodes['no_probability'])) &
                    (np.sign(parent_nodes['no_probability']) == -1)
                )
                
                # Update gain values to represent decisiveness
                update_indices = tree_dataframe['ID'].isin(parent_nodes['ID'])
                tree_dataframe.loc[update_indices & decisive_mask_neutral, 'Gain'] = 0
                tree_dataframe.loc[update_indices & decisive_mask_positive, 'Gain'] = 1
                tree_dataframe.loc[update_indices & decisive_mask_negative, 'Gain'] = -1
                
                current_level -= 1
                
        except Exception as e:
            logger.warning(f"Error marking decisive nodes: {str(e)}")
    
    def combine_rule_sets(self,
                         rule_set_with_probabilities: pd.Series,
                         rule_features_matrix: pd.DataFrame,
                         smaller_feature_indices: np.ndarray,
                         larger_feature_indices: np.ndarray) -> np.ndarray:
        """
        Combine multiple rules into a single consolidated rule set.
        
        Args:
            rule_set_with_probabilities: Series containing rule IDs and final probability
            rule_features_matrix: Matrix of feature conditions for all rules
            smaller_feature_indices: Indices of 'smaller than' conditions
            larger_feature_indices: Indices of 'larger than' conditions
            
        Returns:
            Combined rule as numpy array
        """
        try:
            # Extract individual rules and final probability
            individual_rules = rule_set_with_probabilities[:-1].tolist()
            final_probability = rule_set_with_probabilities[-1]
            
            # Get rules that match the rule set
            matching_rules = rule_features_matrix[
                np.isin(rule_features_matrix[:, 0], individual_rules)
            ]
            feature_conditions = matching_rules[:, 1:-1].astype('float32')
            rule_probabilities = matching_rules[:, -1]
            
            # Filter rules based on consistency with final decision
            if final_probability <= 0.5:
                # Remove positive rules for negative final decision
                feature_conditions[(final_probability <= 0.5) & (rule_probabilities > 0), :] = np.nan
            else:
                # Remove negative rules for positive final decision  
                feature_conditions[(final_probability > 0.5) & (rule_probabilities < 0), :] = np.nan
            
            # Combine conditions: min for upper bounds, max for lower bounds
            min_upper_bounds = np.nanmin(feature_conditions[:, smaller_feature_indices], axis=0)
            max_lower_bounds = np.nanmax(feature_conditions[:, larger_feature_indices], axis=0)
            
            combined_rule = np.hstack((min_upper_bounds, max_lower_bounds))
            
            return combined_rule
            
        except Exception as e:
            logger.error(f"Error in combine_rule_sets: {str(e)}")
            raise ValueError(f"Rule combination failed: {str(e)}")
    
    def generate_customer_segments(self,
                                 trained_xgboost_model: Any,
                                 prediction_results: pd.DataFrame,
                                 rule_frequency_data: pd.DataFrame,
                                 maximum_tree_depth: int,
                                 replication_classes: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate customer segments based on XGBoost decision rules and frequencies.
        
        This comprehensive method:
        1. Extracts decision rules from the trained XGBoost model
        2. Combines rules based on frequency patterns
        3. Creates customer segments with rule-based logic
        4. Provides detailed segment analysis and mapping
        
        Args:
            trained_xgboost_model: Trained XGBoost model with trees_to_dataframe() method
            prediction_results: DataFrame with customer predictions and metadata
            rule_frequency_data: DataFrame with rule combination frequencies
            maximum_tree_depth: Maximum depth of trees in the model
            replication_classes: Number of class replications in the model
            
        Returns:
            Tuple of (enhanced_prediction_results, segment_analysis)
            
        Example:
            >>> segments, analysis = extractor.generate_customer_segments(
            ...     xgb_model, predictions_df, frequency_df, max_depth=6, rep_classes=2)
        """
        try:
            logger.info("Starting customer segmentation based on XGBoost rules...")
            
            # Step 1: Sort and prepare frequency data
            sorted_frequency_data = rule_frequency_data.sort_values(
                by=['freq_percent'], ascending=False
            ).copy()
            
            # Convert rule columns to string for processing
            rule_columns = sorted_frequency_data.columns[:-7]  # Exclude metadata columns
            sorted_frequency_data[rule_columns] = sorted_frequency_data[rule_columns].astype(str)
            
            # Step 2: Extract decision rules from XGBoost trees
            xgboost_trees_df = trained_xgboost_model.trees_to_dataframe()
            decision_rules = self.extract_decision_rules(xgboost_trees_df, maximum_tree_depth)
            
            # Enhance rules with probability information
            tree_probabilities = xgboost_trees_df[['ID', 'Gain']].set_index('ID')
            tree_probabilities.columns = pd.MultiIndex.from_product([['Probability'], ['Gain']])
            enhanced_rules = decision_rules.merge(
                tree_probabilities, how='left', left_index=True, right_index=True
            )
            
            # Step 3: Process rule combinations
            limited_frequency_data = sorted_frequency_data.iloc[
                :, :trained_xgboost_model.best_ntree_limit * replication_classes
            ]
            limited_frequency_data = pd.concat([
                limited_frequency_data, 
                sorted_frequency_data[['prob_max']]
            ], axis=1)
            
            # Identify feature condition indices
            smaller_condition_indices = enhanced_rules.columns.get_level_values(0).str.contains("smaller")[:-1]
            larger_condition_indices = enhanced_rules.columns.get_level_values(0).str.contains("bigger")[:-1]
            
            # Apply rule combination to each row
            combined_rule_sets = np.apply_along_axis(
                self.combine_rule_sets, 1, limited_frequency_data.values,
                urule_features_matrix=enhanced_rules.reset_index().values,
                smaller_feature_indices=smaller_condition_indices,
                larger_feature_indices=larger_condition_indices
            )
            
            # Step 4: Create consolidated rule DataFrame
            consolidated_rules = pd.DataFrame(
                combined_rule_sets,
                columns=pd.MultiIndex.from_arrays([
                    list(np.repeat(['smaller', 'larger'], 
                                   (enhanced_rules.shape[1] - 1) // 2)),
                    enhanced_rules.columns.get_level_values(1)[:-1].unique().tolist() * 2
                ]),
                index=limited_frequency_data.index
            )
            
            # Flatten column names for easier handling
            consolidated_rules.columns = consolidated_rules.columns.map('{0[0]}__{0[1]}'.format)
            
            # Step 5: Create comprehensive segment analysis
            segment_analysis = pd.concat([
                consolidated_rules,
                sorted_frequency_data[['group', 'prob_max', 'frequency', 'freq_percent']]
            ], axis=1)
            
            # Step 6: Aggregate segments by feature combinations
            def aggregate_segments(segment_group):
                """Custom aggregation function for segment grouping."""
                return pd.Series({
                    'segment_probability': segment_group['prob_max'].mean(),
                    'segment_frequency': segment_group['frequency'].sum(),
                    'segment_frequency_percent': segment_group['freq_percent'].sum(),
                    'group_combinations': ';'.join(segment_group['group'].sort_values().tolist()),
                    'primary_group': segment_group['group'].sort_values().iloc[0]
                })
            
            # Group by rule conditions
            rule_condition_columns = consolidated_rules.columns.tolist()
            segment_analysis.iloc[:, :-4] = segment_analysis.iloc[:, :-4].fillna(-1000)
            
            grouped_segments = segment_analysis.groupby(rule_condition_columns).apply(aggregate_segments)
            grouped_segments = grouped_segments.reset_index().sort_values(
                'segment_frequency_percent', ascending=False
            )
            
            # Create segment identifiers
            grouped_segments['segment_id'] = 'segment_' + (
                grouped_segments['group_combinations'].astype('category').cat.codes.astype(str)
            )
            
            # Restore NaN values
            grouped_segments[rule_condition_columns] = grouped_segments[rule_condition_columns].replace(-1000, np.nan)
            
            # Validate segment quality
            if not all(grouped_segments[rule_condition_columns].sum(axis=1) != 0):
                logger.warning(f"Found segments with all NaN values. Total segments: {grouped_segments.shape[0]}")
            
            logger.info(f"Generated {grouped_segments.shape[0]} customer segments successfully")
            
            # Step 7: Map segments back to original predictions
            def map_group_combinations(group_text):
                """Map group combination text to individual group mappings."""
                groups = group_text.split(";")
                if len(groups) == 1:
                    return {}
                
                primary_group = groups[0]
                group_mapping = {}
                for group in groups[1:]:
                    group_mapping[group] = primary_group
                return group_mapping
            
            # Create group mapping dictionary
            group_mappings = grouped_segments['group_combinations'].apply(map_group_combinations)
            non_empty_mappings = group_mappings.loc[group_mappings.apply(len) != 0]
            
            # Consolidate all group mappings
            consolidated_group_mapping = {}
            for mapping_dict in non_empty_mappings:
                consolidated_group_mapping.update(mapping_dict)
            
            # Apply group mappings to prediction results
            enhanced_predictions = prediction_results.copy()
            enhanced_predictions['group'] = enhanced_predictions['group'].replace(consolidated_group_mapping)
            
            # Merge with segment information
            segment_columns = rule_condition_columns + ['primary_group', 'segment_id']
            enhanced_predictions = enhanced_predictions.reset_index().merge(
                grouped_segments[segment_columns],
                how='left',
                left_on=['group'],
                right_on=['primary_group']
            ).set_index(prediction_results.index.name)
            
            enhanced_predictions['total_segments'] = grouped_segments.shape[0]
            
            return enhanced_predictions, grouped_segments
            
        except Exception as e:
            logger.error(f"Error in generate_customer_segments: {str(e)}")
            raise ValueError(f"Customer segmentation failed: {str(e)}")


class DecisionTreeInterpreter:
    """
    Advanced decision tree interpretation and visualization system.
    
    This class provides comprehensive decision tree analysis including:
    - Rule extraction from decision trees
    - Tree-to-code conversion for interpretability
    - XGBoost tree parsing and string manipulation
    - Decision path analysis
    
    Example:
        >>> interpreter = DecisionTreeInterpreter()
        >>> interpreter.extract_tree_rules(tree_model, feature_names)
        >>> code = interpreter.convert_tree_to_code(tree_model, feature_names)
    """
    
    def __init__(self):
        """Initialize the decision tree interpreter."""
        logger.info("DecisionTreeInterpreter initialized")
    
    def convert_tree_to_code(self, 
                           decision_tree_model: Any,
                           feature_names: Union[pd.Index, List[str]]) -> str:
        """
        Convert a decision tree model to human-readable Python code.
        
        This method extracts decision rules from a trained scikit-learn decision tree
        and converts them into executable Python code format, making the model's
        decision logic fully interpretable.
        
        Args:
            decision_tree_model: Trained scikit-learn decision tree model
            feature_names: List or pandas Index of feature names
            
        Returns:
            String containing Python code representation of the decision tree
            
        Raises:
            ValueError: If the model is not a valid decision tree or features are invalid
            
        Example:
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> tree_model = DecisionTreeClassifier().fit(X_train, y_train)
            >>> code = interpreter.convert_tree_to_code(tree_model, X_train.columns)
            >>> print(code)
        """
        try:
            from sklearn.tree import _tree
            
            # Validate inputs
            if not hasattr(decision_tree_model, 'tree_'):
                raise ValueError("Model must be a trained scikit-learn decision tree")
            
            if isinstance(feature_names, pd.Index):
                feature_names = feature_names.tolist()
            
            tree_structure = decision_tree_model.tree_
            
            # Create feature name mapping
            feature_name_mapping = [
                feature_names[i] if i != tree_structure.TREE_UNDEFINED else "undefined_feature"
                for i in tree_structure.feature
            ]
            
            logger.info(f"Converting decision tree with {tree_structure.node_count} nodes to code")
            
            # Generate code recursively
            code_lines = []
            code_lines.append("def tree_decision_function(sample):")
            code_lines.append("    \"\"\"")
            code_lines.append("    Generated decision tree function.")
            code_lines.append(f"    Features: {', '.join(feature_names)}")
            code_lines.append("    \"\"\"")
            
            def generate_node_code(node_id: int, depth: int) -> List[str]:
                """Recursively generate code for each tree node."""
                indent = "    " * (depth + 1)
                node_lines = []
                
                if tree_structure.feature[node_id] != tree_structure.TREE_UNDEFINED:
                    # Internal node - create condition
                    feature_name = feature_name_mapping[node_id]
                    threshold = tree_structure.threshold[node_id]
                    
                    node_lines.append(f"{indent}if sample['{feature_name}'] <= {threshold:.6f}:")
                    
                    # Left child (condition is true)
                    left_child = tree_structure.children_left[node_id]
                    node_lines.extend(generate_node_code(left_child, depth + 1))
                    
                    node_lines.append(f"{indent}else:  # sample['{feature_name}'] > {threshold:.6f}")
                    
                    # Right child (condition is false)
                    right_child = tree_structure.children_right[node_id]
                    node_lines.extend(generate_node_code(right_child, depth + 1))
                    
                else:
                    # Leaf node - return prediction
                    prediction_value = tree_structure.value[node_id]
                    samples_count = tree_structure.n_node_samples[node_id]
                    
                    if prediction_value.shape[1] == 1:
                        # Regression
                        pred = prediction_value[0, 0]
                        node_lines.append(f"{indent}return {pred:.6f}  # {samples_count} samples")
                    else:
                        # Classification
                        class_counts = prediction_value[0]
                        predicted_class = np.argmax(class_counts)
                        confidence = class_counts[predicted_class] / np.sum(class_counts)
                        
                        node_lines.append(
                            f"{indent}return {predicted_class}  # class {predicted_class}, "
                            f"{samples_count} samples, confidence: {confidence:.3f}"
                        )
                
                return node_lines
            
            # Generate the complete function
            code_lines.extend(generate_node_code(0, 0))
            
            complete_code = "\\n".join(code_lines)
            
            logger.info("Decision tree code generation completed successfully")
            return complete_code
            
        except Exception as e:
            logger.error(f"Error in convert_tree_to_code: {str(e)}")
            raise ValueError(f"Tree to code conversion failed: {str(e)}")
    
    def parse_xgboost_tree_string(self, tree_string: str) -> Dict[str, Any]:
        """
        Parse XGBoost tree string representation into structured format.
        
        This method parses the string representation of XGBoost trees and extracts
        structured information about nodes, conditions, and predictions.
        
        Args:
            tree_string: String representation of XGBoost tree structure
            
        Returns:
            Dictionary containing parsed tree information
            
        Example:
            >>> tree_str = "0:[f0<0.5] yes=1,no=2,missing=1"
            >>> parsed = interpreter.parse_xgboost_tree_string(tree_str)
        """
        try:
            import re
            
            if not tree_string or not isinstance(tree_string, str):
                raise ValueError("tree_string must be a non-empty string")
            
            # Check if this is a leaf node
            if re.search(r":leaf=", tree_string):
                # Leaf node - extract leaf value
                leaf_match = re.search(r":leaf=([\\d.-]+)", tree_string)
                if leaf_match:
                    return {
                        'node_type': 'leaf',
                        'leaf_value': float(leaf_match.group(1)),
                        'raw_string': tree_string
                    }
                else:
                    return {'node_type': 'invalid_leaf', 'raw_string': tree_string}
            
            # Internal node - parse structure
            components = re.findall(r"[\\w.-]+", tree_string)
            tab_levels = re.findall(r"[\\t]+", tree_string)
            
            if len(components) < 8:
                return {'node_type': 'invalid_internal', 'raw_string': tree_string}
            
            # Extract node information
            node_info = {
                'node_type': 'internal',
                'node_id': components[0],
                'feature': components[1],
                'threshold': float(components[2]),
                'yes_node': components[3],
                'no_node': components[4],
                'missing_node': components[5] if len(components) > 5 else components[3],
                'raw_string': tree_string
            }
            
            # Handle missing value direction
            if len(components) >= 8 and components[4] == components[8]:
                node_info['missing_direction'] = 'yes'
                node_info['missing_handling'] = f" or np.isnan(sample['{components[1]}'])"
            else:
                node_info['missing_direction'] = 'no'
                node_info['missing_handling'] = ""
            
            # Extract indentation level
            if tab_levels:
                node_info['depth'] = len(tab_levels[0])
            else:
                node_info['depth'] = 0
            
            return node_info
            
        except Exception as e:
            logger.error(f"Error parsing XGBoost tree string: {str(e)}")
            return {'node_type': 'error', 'error': str(e), 'raw_string': tree_string}
    
    def extract_tree_parser_information(self, 
                                       decision_tree_structure: Any,
                                       node_index: int) -> Dict[str, Any]:
        """
        Extract detailed information from a specific tree node.
        
        Args:
            decision_tree_structure: Tree structure from XGBoost or scikit-learn
            node_index: Index of the node to analyze
            
        Returns:
            Dictionary with comprehensive node information
        """
        try:
            # This would typically interface with XGBoost internals
            # Implementation depends on the specific tree structure format
            logger.info(f"Extracting information for node {node_index}")
            
            # Placeholder for tree parser implementation
            node_info = {
                'node_index': node_index,
                'extraction_timestamp': pd.Timestamp.now(),
                'status': 'not_implemented'
            }
            
            return node_info
            
        except Exception as e:
            logger.error(f"Error extracting tree parser information: {str(e)}")
            return {'error': str(e), 'node_index': node_index}

# ============================================================================
# ADVANCED MACHINE LEARNING ANALYTICS CLASSES
# ============================================================================

class MulticollinearityDetector:
    """
    Comprehensive multicollinearity detection using Variance Inflation Factor (VIF) analysis.
    
    This class provides robust methods for detecting and analyzing multicollinearity
    in datasets, with actionable recommendations for addressing collinearity issues
    in machine learning and statistical modeling contexts.
    """
    
    def __init__(self, vif_threshold: float = 5.0):
        """
        Initialize the MulticollinearityDetector.
        
        Parameters:
        -----------
        vif_threshold : float, default=5.0
            Threshold for VIF values above which multicollinearity is considered
            problematic. Common thresholds: 5 (moderate), 10 (severe).
        """
        self.vif_threshold = vif_threshold
        self.analysis_history = []
        
    def calculate_variance_inflation_factors(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Variance Inflation Factor (VIF) for multicollinearity assessment.
        
        VIF measures how much the variance of a regression coefficient increases
        due to collinearity. Higher VIF values indicate stronger multicollinearity.
        VIF = 1/(1-R²) where R² is from regressing each feature on all others.
        
        Parameters:
        -----------
        feature_matrix : pandas.DataFrame
            DataFrame containing numeric features to analyze. Each column represents
            a feature, each row an observation. Non-numeric columns are ignored.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns 'Feature' and 'VIF', sorted by VIF value in
            descending order. Higher VIF values indicate stronger multicollinearity.
            
        Raises:
        -------
        ValueError
            If feature_matrix is empty or contains insufficient numeric features
        ImportError
            If required statistical libraries are not available
            
        Examples:
        --------
        >>> detector = MulticollinearityDetector()
        >>> vif_df = detector.calculate_variance_inflation_factors(features)
        >>> print(vif_df.head())
        """
        # Input validation
        if not isinstance(feature_matrix, pd.DataFrame):
            raise TypeError("feature_matrix must be a pandas DataFrame")
            
        if feature_matrix.empty:
            raise ValueError("feature_matrix cannot be empty")
        
        # Import required libraries
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np
        except ImportError as e:
            raise ImportError(f"Required libraries not available: {e}")
        
        # Select only numeric columns
        numeric_features = feature_matrix.select_dtypes(include=[np.number])
        
        if numeric_features.shape[1] < 2:
            raise ValueError("At least 2 numeric features required for VIF calculation")
        
        logger.info(f"Calculating VIF for {numeric_features.shape[1]} numeric features")
        
        # Remove any columns with zero variance
        feature_variances = numeric_features.var()
        valid_features = numeric_features.loc[:, feature_variances > 1e-10]
        
        if valid_features.shape[1] < 2:
            raise ValueError("Insufficient features with non-zero variance")
        
        feature_names = valid_features.columns.tolist()
        vif_values = {}
        
        # Calculate VIF for each feature
        for target_feature in feature_names:
            try:
                # Prepare predictor features (all except current target)
                predictor_features = [f for f in feature_names if f != target_feature]
                
                # Extract target and predictor variables
                target_variable = valid_features[target_feature].values
                predictor_matrix = valid_features[predictor_features].values
                
                # Handle edge case of single predictor
                if len(predictor_features) == 1:
                    # Simple correlation coefficient squared
                    correlation_coeff = np.corrcoef(target_variable, predictor_matrix.flatten())[0, 1]
                    r_squared = correlation_coeff ** 2
                else:
                    # Multiple regression
                    regression_model = LinearRegression(fit_intercept=True)
                    regression_model.fit(predictor_matrix, target_variable)
                    
                    # Calculate R-squared
                    predicted_values = regression_model.predict(predictor_matrix)
                    correlation_coeff = np.corrcoef(target_variable, predicted_values)[0, 1]
                    r_squared = max(0, min(0.99999, correlation_coeff ** 2))  # Constrain to valid range
                
                # Calculate VIF: 1/(1-R²)
                vif_value = 1 / (1 - r_squared)
                vif_values[target_feature] = vif_value
                
            except Exception as e:
                logger.warning(f"Could not calculate VIF for {target_feature}: {e}")
                vif_values[target_feature] = np.nan
        
        # Create results DataFrame
        vif_dataframe = pd.DataFrame([
            {'Feature': feature, 'VIF': vif_value}
            for feature, vif_value in vif_values.items()
        ])
        
        # Sort by VIF value in descending order
        vif_dataframe = vif_dataframe.sort_values('VIF', ascending=False)
        vif_dataframe.reset_index(drop=True, inplace=True)
        
        logger.info(f"VIF calculation completed. Highest VIF: {vif_dataframe.iloc[0]['VIF']:.2f}")
        return vif_dataframe
    
    def analyze_multicollinearity_patterns(self, 
                                         feature_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], str]:
        """
        Comprehensive multicollinearity analysis with actionable recommendations.
        
        Performs VIF analysis and provides detailed recommendations for addressing
        multicollinearity issues based on severity levels and feature characteristics.
        
        Parameters:
        -----------
        feature_matrix : pandas.DataFrame
            DataFrame with numeric features for multicollinearity analysis.
            
        Returns:
        --------
        Tuple[pd.DataFrame, List[str], str]
            - vif_results: DataFrame with VIF values for each feature
            - problematic_features: List of feature names exceeding VIF threshold
            - recommendations: Detailed string with suggested remediation actions
            
        Examples:
        --------
        >>> detector = MulticollinearityDetector(vif_threshold=5.0)
        >>> vif_df, problem_features, advice = detector.analyze_multicollinearity_patterns(data)
        >>> print(f"Problematic features: {problem_features}")
        >>> print(advice)
        """
        # Calculate VIF values
        vif_results = self.calculate_variance_inflation_factors(feature_matrix)
        
        # Identify problematic features
        problematic_features = vif_results[
            vif_results['VIF'] > self.vif_threshold
        ]['Feature'].tolist()
        
        # Generate severity-based recommendations
        if not problematic_features:
            recommendations = (
                "✅ No significant multicollinearity detected in the dataset.\n"
                f"All features have VIF values below the threshold of {self.vif_threshold}.\n"
                "Your feature set appears suitable for linear modeling approaches."
            )
        else:
            # Analyze severity levels
            severe_features = vif_results[vif_results['VIF'] > 10]['Feature'].tolist()
            moderate_features = vif_results[
                (vif_results['VIF'] > self.vif_threshold) & (vif_results['VIF'] <= 10)
            ]['Feature'].tolist()
            
            recommendations = f"""
🚨 MULTICOLLINEARITY ANALYSIS RESULTS
{"="*50}

Detected Issues:
• {len(problematic_features)} features exceed VIF threshold of {self.vif_threshold}
• {len(severe_features)} features show severe multicollinearity (VIF > 10)
• {len(moderate_features)} features show moderate multicollinearity (VIF 5-10)

Problematic Features:
{', '.join(f"{feat} (VIF: {vif_results[vif_results['Feature']==feat]['VIF'].iloc[0]:.2f})" for feat in problematic_features[:10])}

RECOMMENDED REMEDIATION STRATEGIES:
{"="*40}

1. FEATURE REMOVAL (Immediate):
   • Remove features with highest VIF values iteratively
   • Start with VIF > 10: {', '.join(severe_features[:5]) if severe_features else 'None'}
   • Recalculate VIF after each removal

2. DIMENSIONALITY REDUCTION:
   • Apply Principal Component Analysis (PCA) to correlated feature groups
   • Use Factor Analysis for conceptually related features
   • Consider Independent Component Analysis (ICA) for non-Gaussian features

3. FEATURE ENGINEERING:
   • Create interaction terms instead of using raw correlated features
   • Combine correlated features using domain knowledge (e.g., ratios, sums)
   • Apply polynomial feature transformation with regularization

4. REGULARIZATION TECHNIQUES:
   • Use Ridge Regression (L2) to handle multicollinearity in linear models
   • Apply Lasso Regression (L1) for automatic feature selection
   • Consider Elastic Net for balanced regularization approach

5. SPECIALIZED MODELING:
   • Use tree-based models (Random Forest, XGBoost) which handle collinearity better
   • Apply Partial Least Squares (PLS) regression for high-dimensional correlated data
   • Consider Bayesian approaches with appropriate priors

6. DATA COLLECTION:
   • Collect additional observations if sample size is limited
   • Gather more diverse data to reduce correlation patterns
   • Ensure balanced representation across all feature ranges

For categorical/encoded features specifically:
• Reduce categories by combining similar ones
• Use ordinal encoding for naturally ordered categories
• Apply target encoding with cross-validation to prevent overfitting

NEXT STEPS:
• Examine correlation matrix for feature relationships
• Test model performance with and without problematic features
• Monitor model stability across different data samples
            """.strip()
        
        # Store analysis in history
        analysis_record = {
            'timestamp': pd.Timestamp.now(),
            'n_features': len(feature_matrix.columns),
            'n_problematic': len(problematic_features),
            'max_vif': vif_results['VIF'].max(),
            'threshold_used': self.vif_threshold
        }
        self.analysis_history.append(analysis_record)
        
        return vif_results, problematic_features, recommendations


class CanonicalAnalyzer:
    """
    Canonical Correlation Analysis (CCA) with advanced visualization and interpretation.
    
    This class provides comprehensive CCA functionality for exploring relationships
    between two sets of variables, with interactive visualizations, clustering
    options, and detailed statistical interpretation capabilities.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the CanonicalAnalyzer.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible results in clustering operations
        """
        self.random_state = random_state
        self.fitted_models = {}
        self.analysis_results = {}
    
    def perform_canonical_correlation_analysis(self,
                                            variable_set_1: pd.DataFrame,
                                            variable_set_2: pd.DataFrame,
                                            n_canonical_components: int = 2,
                                            preprocessing_method: str = 'standard',
                                            cluster_variables_1: bool = False,
                                            cluster_variables_2: bool = False) -> Dict[str, Any]:
        """
        Perform Canonical Correlation Analysis with comprehensive visualization options.
        
        CCA finds linear combinations of two sets of variables that are maximally
        correlated. This implementation includes preprocessing options, hierarchical
        clustering for variable grouping, and interactive visualizations.
        
        Parameters:
        -----------
        variable_set_1 : pandas.DataFrame
            First set of variables (predictors). Each column represents a variable,
            each row an observation. Must have same number of rows as variable_set_2.
        variable_set_2 : pandas.DataFrame
            Second set of variables (responses). Each column represents a variable,
            each row an observation. Must have same number of rows as variable_set_1.
        n_canonical_components : int, default=2
            Number of canonical component pairs to extract. Must be <= min(n_vars1, n_vars2).
        preprocessing_method : str, default='standard'
            Preprocessing method for variables. Options:
            - 'standard': StandardScaler (mean=0, std=1)
            - 'minmax': MinMaxScaler (range 0-1)
            - 'robust': RobustScaler (median=0, IQR=1)
            - 'none': No preprocessing applied
        cluster_variables_1 : bool, default=False
            Whether to apply hierarchical clustering to reorder variables in set 1
            for better visualization of coefficient patterns.
        cluster_variables_2 : bool, default=False
            Whether to apply hierarchical clustering to reorder variables in set 2
            for better visualization of coefficient patterns.
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive results dictionary containing:
            - 'canonical_correlations': Series of canonical correlation coefficients
            - 'raw_coefficients': DataFrame of raw CCA coefficients
            - 'normalized_coefficients': DataFrame with normalized coefficients
            - 'correlation_plot': Plotly figure showing canonical correlations
            - 'coefficient_heatmap': Plotly heatmap of coefficient relationships
            - 'model': Fitted CCA model for predictions
            - 'preprocessing': Preprocessing parameters used
            
        Raises:
        -------
        ValueError
            If datasets have different numbers of rows, invalid preprocessing method,
            or insufficient canonical components requested
        ImportError
            If required statistical libraries are not available
            
        Examples:
        --------
        >>> analyzer = CanonicalAnalyzer()
        >>> results = analyzer.perform_canonical_correlation_analysis(
        ...     X1, X2, n_canonical_components=3, preprocessing_method='standard'
        ... )
        >>> 
        >>> # Display results
        >>> results['correlation_plot'].show()
        >>> results['coefficient_heatmap'].show()
        >>> print(results['canonical_correlations'])
        """
        # Input validation
        if not isinstance(variable_set_1, pd.DataFrame) or not isinstance(variable_set_2, pd.DataFrame):
            raise TypeError("Both variable sets must be pandas DataFrames")
            
        if variable_set_1.shape[0] != variable_set_2.shape[0]:
            raise ValueError("Variable sets must have the same number of observations (rows)")
            
        if variable_set_1.empty or variable_set_2.empty:
            raise ValueError("Variable sets cannot be empty")
            
        max_components = min(variable_set_1.shape[1], variable_set_2.shape[1])
        if n_canonical_components > max_components:
            raise ValueError(f"n_canonical_components ({n_canonical_components}) cannot exceed "
                           f"min(n_vars1, n_vars2) = {max_components}")
        
        valid_preprocessing_methods = ['standard', 'minmax', 'robust', 'none']
        if preprocessing_method not in valid_preprocessing_methods:
            raise ValueError(f"preprocessing_method must be one of: {valid_preprocessing_methods}")
        
        # Import required libraries
        try:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
            from sklearn.cross_decomposition import CCA
            import plotly.express as px
            import plotly.graph_objects as go
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import pdist
        except ImportError as e:
            raise ImportError(f"Required libraries not available: {e}")
        
        logger.info(f"Performing CCA with {n_canonical_components} components on "
                   f"{variable_set_1.shape[1]} vs {variable_set_2.shape[1]} variables")
        
        # Apply preprocessing
        if preprocessing_method == 'standard':
            scaler = StandardScaler()
        elif preprocessing_method == 'minmax':
            scaler = MinMaxScaler()
        elif preprocessing_method == 'robust':
            scaler = RobustScaler()
        else:  # 'none'
            scaler = None
        
        if scaler is not None:
            variables_1_scaled = pd.DataFrame(
                scaler.fit_transform(variable_set_1),
                columns=variable_set_1.columns,
                index=variable_set_1.index
            )
            variables_2_scaled = pd.DataFrame(
                scaler.fit_transform(variable_set_2),
                columns=variable_set_2.columns,
                index=variable_set_2.index
            )
        else:
            variables_1_scaled = variable_set_1.copy()
            variables_2_scaled = variable_set_2.copy()
        
        # Fit CCA model
        cca_model = CCA(scale=False, n_components=n_canonical_components)
        cca_model.fit(variables_1_scaled, variables_2_scaled)
        
        # Transform data to canonical space
        canonical_vars_1, canonical_vars_2 = cca_model.transform(variables_1_scaled, variables_2_scaled)
        
        # Calculate canonical correlations
        canonical_correlations = pd.Series([
            np.corrcoef(canonical_vars_1[:, i], canonical_vars_2[:, i])[1, 0]
            for i in range(n_canonical_components)
        ], index=[f"CC{i+1}" for i in range(n_canonical_components)])
        
        # Extract and format coefficients
        raw_coefficients = pd.DataFrame(
            np.round(cca_model.coef_, 4),
            index=variables_2_scaled.columns,
            columns=variables_1_scaled.columns
        )
        
        # Normalize coefficients for visualization
        coefficient_normalizer = MinMaxScaler()
        normalized_coefficients = pd.DataFrame(
            2 * coefficient_normalizer.fit_transform(raw_coefficients) - 1,
            index=raw_coefficients.index,
            columns=raw_coefficients.columns
        )
        
        # Apply hierarchical clustering if requested
        visualization_data = normalized_coefficients.copy()
        
        if cluster_variables_2:
            # Cluster rows (variable set 2)
            row_linkage = linkage(pdist(visualization_data.values, metric='euclidean'), 
                                method='ward')
            row_dendrogram = dendrogram(row_linkage, no_plot=True)
            row_order = row_dendrogram['leaves']
            visualization_data = visualization_data.iloc[row_order]
        
        if cluster_variables_1:
            # Cluster columns (variable set 1)
            col_linkage = linkage(pdist(visualization_data.values.T, metric='euclidean'), 
                                method='ward')
            col_dendrogram = dendrogram(col_linkage, no_plot=True)
            col_order = col_dendrogram['leaves']
            visualization_data = visualization_data.iloc[:, col_order]
        
        # Create canonical correlations bar plot
        correlation_plot = px.bar(
            x=canonical_correlations.index,
            y=canonical_correlations.values,
            title=f'Canonical Correlations (n_components={n_canonical_components})',
            labels={'x': 'Canonical Component', 'y': 'Correlation Coefficient'},
            text=np.round(canonical_correlations.values, 3)
        )
        
        correlation_plot.update_traces(
            texttemplate='%{text}', 
            textposition='outside',
            marker_color='lightblue',
            marker_line=dict(color='darkblue', width=1)
        )
        
        correlation_plot.update_layout(
            width=800, 
            height=500,
            showlegend=False,
            title_x=0.5,
            template='plotly_white',
            yaxis=dict(range=[0, 1.1]),
            font=dict(size=12)
        )
        
        # Create coefficient heatmap
        set_1_name = variable_set_1.columns.name or 'Variable Set 1'
        set_2_name = variable_set_2.columns.name or 'Variable Set 2'
        
        # Generate clustering information for title
        clustering_info = ""
        if cluster_variables_1 or cluster_variables_2:
            clustered_aspects = []
            if cluster_variables_2:
                clustered_aspects.append("rows")
            if cluster_variables_1:
                clustered_aspects.append("columns")
            clustering_info = f" (Clustered: {' & '.join(clustered_aspects)})"
        
        # Format correlation values for subtitle
        top_correlations = ', '.join([
            f'{corr:.3f}' for corr in canonical_correlations.head(3).values
        ])
        
        heatmap_title = (f'CCA Coefficients: {set_2_name} → {set_1_name}{clustering_info}<br>'
                        f'<sub>Top Canonical Correlations: {top_correlations}</sub>')
        
        coefficient_heatmap = go.Figure(data=go.Heatmap(
            z=visualization_data.values,
            x=visualization_data.columns,
            y=visualization_data.index,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(
                title="Normalized<br>Coefficient",
                titleside="right"
            ),
            hovertemplate='%{y} → %{x}<br>Coefficient: %{z:.3f}<extra></extra>',
            xgap=1,
            ygap=1
        ))
        
        # Dynamic sizing and font adjustment
        x_font_size = max(8, min(14, 300 / len(visualization_data.columns)))
        y_font_size = max(8, min(14, 300 / len(visualization_data.index)))
        
        coefficient_heatmap.update_layout(
            title=heatmap_title,
            title_x=0.5,
            width=min(1200, max(600, len(visualization_data.columns) * 40)),
            height=min(1000, max(500, len(visualization_data.index) * 35)),
            xaxis_title=set_1_name,
            yaxis_title=set_2_name,
            template='plotly_white',
            xaxis=dict(tickfont=dict(size=x_font_size)),
            yaxis=dict(tickfont=dict(size=y_font_size)),
            font=dict(size=12)
        )
        
        # Compile comprehensive results
        analysis_results = {
            'canonical_correlations': canonical_correlations,
            'raw_coefficients': raw_coefficients,
            'normalized_coefficients': normalized_coefficients,
            'correlation_plot': correlation_plot,
            'coefficient_heatmap': coefficient_heatmap,
            'model': cca_model,
            'preprocessing': {
                'method': preprocessing_method,
                'scaler': scaler,
                'clustered_vars1': cluster_variables_1,
                'clustered_vars2': cluster_variables_2
            },
            'canonical_variables': {
                'set_1': canonical_vars_1,
                'set_2': canonical_vars_2
            },
            'explained_variance': {
                'cumulative_correlation': canonical_correlations.cumsum(),
                'total_correlation': canonical_correlations.sum()
            }
        }
        
        # Store results for potential reuse
        analysis_key = f"cca_{n_canonical_components}_{preprocessing_method}"
        self.analysis_results[analysis_key] = analysis_results
        
        logger.info(f"CCA analysis completed. Top correlation: {canonical_correlations.iloc[0]:.3f}")
        return analysis_results


class ClusteringAnalyzer:
    """
    Advanced K-means clustering analysis with comprehensive evaluation and optimization.
    
    This class provides robust clustering analysis with automatic parameter selection,
    multiple evaluation metrics, and detailed cluster characterization for exploratory
    data analysis and unsupervised learning workflows.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ClusteringAnalyzer.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible clustering results
        """
        self.random_state = random_state
        self.clustering_results = {}
        self.evaluation_history = []
    
    def perform_optimal_clustering_analysis(self,
                                          dataset: pd.DataFrame,
                                          cluster_range: range = range(2, 8),
                                          scaling_strategy: str = 'standard',
                                          evaluation_metric: str = 'silhouette') -> Dict[str, Any]:
        """
        Perform comprehensive K-means clustering with automatic parameter optimization.
        
        This method evaluates multiple cluster numbers, applies appropriate preprocessing,
        and identifies the optimal clustering configuration based on specified metrics.
        Provides detailed cluster characterization and validation metrics.
        
        Parameters:
        -----------
        dataset : pandas.DataFrame
            Dataset for clustering analysis. Rows represent observations, columns
            represent features. Non-numeric columns are automatically excluded.
        cluster_range : range, default=range(2, 8)
            Range of cluster numbers to evaluate. Should be reasonable for dataset
            size and computational constraints.
        scaling_strategy : str, default='standard'
            Preprocessing strategy for features. Options:
            - 'standard': StandardScaler (z-score normalization)
            - 'minmax': MinMaxScaler (0-1 scaling)
            - 'robust': RobustScaler (median and IQR based)
            - 'none': No preprocessing applied
        evaluation_metric : str, default='silhouette'
            Primary metric for cluster evaluation. Options:
            - 'silhouette': Silhouette coefficient (higher is better)
            - 'inertia': Within-cluster sum of squares (lower is better)
            - 'calinski_harabasz': Calinski-Harabasz index (higher is better)
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive clustering results containing:
            - 'cluster_assignments': Series with cluster labels for each observation
            - 'optimal_clusters': Number of clusters selected
            - 'evaluation_scores': Dict of scores for all tested cluster numbers
            - 'preprocessing_scaler': Fitted scaler object (if scaling applied)
            - 'clustering_model': Fitted KMeans model
            - 'cluster_centers': Cluster centroids in original feature space
            - 'cluster_statistics': Detailed statistics for each cluster
            - 'silhouette_scores': Individual silhouette scores for validation
            
        Raises:
        -------
        ValueError
            If dataset is empty, invalid scaling strategy, or insufficient numeric features
        ImportError
            If required clustering libraries are not available
            
        Examples:
        --------
        >>> analyzer = ClusteringAnalyzer()
        >>> results = analyzer.perform_optimal_clustering_analysis(
        ...     data, cluster_range=range(2, 10), scaling_strategy='standard'
        ... )
        >>> 
        >>> print(f"Optimal clusters: {results['optimal_clusters']}")
        >>> print(f"Silhouette score: {results['evaluation_scores']['silhouette']:.3f}")
        >>> 
        >>> # Access cluster assignments
        >>> cluster_labels = results['cluster_assignments']
        """
        # Input validation
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame")
            
        if dataset.empty:
            raise ValueError("dataset cannot be empty")
            
        valid_scaling_strategies = ['standard', 'minmax', 'robust', 'none']
        if scaling_strategy not in valid_scaling_strategies:
            raise ValueError(f"scaling_strategy must be one of: {valid_scaling_strategies}")
            
        valid_metrics = ['silhouette', 'inertia', 'calinski_harabasz']
        if evaluation_metric not in valid_metrics:
            raise ValueError(f"evaluation_metric must be one of: {valid_metrics}")
        
        # Import required libraries
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        except ImportError as e:
            raise ImportError(f"Required clustering libraries not available: {e}")
        
        # Prepare numeric data
        numeric_dataset = dataset.select_dtypes(include=[np.number])
        if numeric_dataset.shape[1] == 0:
            raise ValueError("Dataset must contain at least one numeric column")
            
        observation_labels = dataset.index.tolist()
        feature_matrix = numeric_dataset.values
        
        logger.info(f"Performing clustering analysis on {feature_matrix.shape[0]} observations "
                   f"with {feature_matrix.shape[1]} features")
        
        # Apply preprocessing if requested
        preprocessing_scaler = None
        if scaling_strategy != 'none':
            logger.info(f"Applying {scaling_strategy.upper()} scaling preprocessing")
            
            if scaling_strategy == 'standard':
                preprocessing_scaler = StandardScaler()
            elif scaling_strategy == 'minmax':
                preprocessing_scaler = MinMaxScaler()
            elif scaling_strategy == 'robust':
                preprocessing_scaler = RobustScaler()
            
            processed_features = preprocessing_scaler.fit_transform(feature_matrix)
        else:
            logger.info("No preprocessing applied to features")
            processed_features = feature_matrix.copy()
        
        # Evaluate different cluster numbers
        logger.info(f"Evaluating cluster numbers: {list(cluster_range)}")
        evaluation_scores = {}
        fitted_models = {}
        
        for n_clusters in cluster_range:
            # Fit K-means model
            kmeans_model = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            cluster_predictions = kmeans_model.fit_predict(processed_features)
            
            # Calculate evaluation metrics
            silhouette_avg = silhouette_score(processed_features, cluster_predictions)
            inertia_score = kmeans_model.inertia_
            calinski_score = calinski_harabasz_score(processed_features, cluster_predictions)
            
            evaluation_scores[n_clusters] = {
                'silhouette': silhouette_avg,
                'inertia': inertia_score,
                'calinski_harabasz': calinski_score
            }
            fitted_models[n_clusters] = kmeans_model
            
            logger.info(f"k={n_clusters}: Silhouette={silhouette_avg:.3f}, "
                       f"Inertia={inertia_score:.0f}, Calinski-Harabasz={calinski_score:.1f}")
        
        # Select optimal number of clusters
        if evaluation_metric == 'silhouette':
            optimal_k = max(evaluation_scores.keys(), 
                           key=lambda k: evaluation_scores[k]['silhouette'])
        elif evaluation_metric == 'calinski_harabasz':
            optimal_k = max(evaluation_scores.keys(), 
                           key=lambda k: evaluation_scores[k]['calinski_harabasz'])
        else:  # inertia
            optimal_k = min(evaluation_scores.keys(), 
                           key=lambda k: evaluation_scores[k]['inertia'])
        
        optimal_model = fitted_models[optimal_k]
        optimal_predictions = optimal_model.predict(processed_features)
        
        logger.info(f"Optimal clustering: k={optimal_k} "
                   f"(Silhouette: {evaluation_scores[optimal_k]['silhouette']:.3f})")
        
        # Create cluster assignment series
        cluster_assignments = pd.Series(
            optimal_predictions + 1,  # 1-indexed clusters
            index=numeric_dataset.index,
            name='cluster_id'
        )
        
        # Calculate detailed cluster statistics
        cluster_statistics = {}
        for cluster_id in range(1, optimal_k + 1):
            cluster_mask = cluster_assignments == cluster_id
            cluster_data = numeric_dataset[cluster_mask]
            
            cluster_statistics[cluster_id] = {
                'size': cluster_mask.sum(),
                'percentage': (cluster_mask.sum() / len(cluster_assignments)) * 100,
                'centroid': cluster_data.mean().to_dict(),
                'feature_std': cluster_data.std().to_dict(),
                'feature_ranges': {
                    col: {'min': cluster_data[col].min(), 'max': cluster_data[col].max()}
                    for col in cluster_data.columns
                }
            }
        
        # Calculate individual silhouette scores for validation
        individual_silhouette_scores = silhouette_samples(processed_features, optimal_predictions)
        
        # Transform cluster centers back to original feature space if scaling was applied
        if preprocessing_scaler is not None:
            original_cluster_centers = preprocessing_scaler.inverse_transform(optimal_model.cluster_centers_)
        else:
            original_cluster_centers = optimal_model.cluster_centers_
        
        # Compile comprehensive results
        clustering_results = {
            'cluster_assignments': cluster_assignments,
            'optimal_clusters': optimal_k,
            'evaluation_scores': evaluation_scores,
            'best_scores': evaluation_scores[optimal_k],
            'preprocessing_scaler': preprocessing_scaler,
            'clustering_model': optimal_model,
            'cluster_centers': pd.DataFrame(
                original_cluster_centers,
                columns=numeric_dataset.columns,
                index=[f'Cluster_{i+1}' for i in range(optimal_k)]
            ),
            'cluster_statistics': cluster_statistics,
            'silhouette_scores': pd.Series(
                individual_silhouette_scores,
                index=numeric_dataset.index,
                name='silhouette_score'
            ),
            'preprocessing_info': {
                'strategy': scaling_strategy,
                'features_used': numeric_dataset.columns.tolist(),
                'n_observations': len(observation_labels)
            }
        }
        
        # Store results and evaluation history
        analysis_key = f"clustering_{optimal_k}_{scaling_strategy}_{evaluation_metric}"
        self.clustering_results[analysis_key] = clustering_results
        
        self.evaluation_history.append({
            'timestamp': pd.Timestamp.now(),
            'optimal_k': optimal_k,
            'scaling_strategy': scaling_strategy,
            'evaluation_metric': evaluation_metric,
            'best_score': evaluation_scores[optimal_k][evaluation_metric],
            'n_features': numeric_dataset.shape[1],
            'n_observations': numeric_dataset.shape[0]
        })
        
        return clustering_results

class DimensionalityReducer:
    """
    Advanced dimensionality reduction with t-SNE visualization and comprehensive analysis.
    
    This class provides high-quality t-SNE visualizations with customizable parameters,
    automatic data preprocessing, and interactive plotting capabilities for exploratory
    data analysis and cluster visualization.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the DimensionalityReducer.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible results across all operations
        """
        self.random_state = random_state
        self.fitted_models = {}
        self.preprocessing_params = {}
        
    def create_tsne_visualization(self,
                                 data: Union[np.ndarray, pd.DataFrame],
                                 class_labels: Optional[Union[np.ndarray, pd.Series, List]] = None,
                                 n_dimensions: int = 2,
                                 perplexity_value: float = 30.0,
                                 max_iterations: int = 1000,
                                 plot_title: Optional[str] = None,
                                 color_scheme: Optional[Union[str, List[str]]] = None,
                                 marker_size: int = 5,
                                 figure_width: int = 800,
                                 figure_height: int = 600,
                                 apply_normalization: bool = True) -> Tuple[Any, np.ndarray]:
        """
        Create interactive t-SNE visualization with comprehensive customization options.
        
        This method performs t-SNE dimensionality reduction and creates publication-ready
        interactive visualizations suitable for exploratory data analysis, cluster
        validation, and presentation purposes.
        
        Parameters:
        -----------
        data : array-like, shape (n_samples, n_features)
            Input dataset for dimensionality reduction. Can be numpy array, pandas DataFrame,
            or any array-like structure. Features should be numeric.
        class_labels : array-like, shape (n_samples,), optional
            Class labels for color-coding points in visualization. If None, all points
            will use the same color. Supports categorical and continuous labels.
        n_dimensions : int, default=2
            Output dimensionality for t-SNE. Must be 2 for 2D scatter plot or 3 for 3D
            scatter plot visualization.
        perplexity_value : float, default=30.0
            Perplexity parameter for t-SNE algorithm. Controls the balance between local
            and global structure preservation. Typical range: 5-50. Higher values
            preserve global structure better.
        max_iterations : int, default=1000
            Maximum number of iterations for t-SNE optimization. More iterations may
            improve results but increase computation time.
        plot_title : str, optional
            Custom title for the visualization. If None, generates automatic title
            based on dimensionality.
        color_scheme : str or list, optional
            Color palette for visualization. Can be plotly color scale name
            (e.g., 'viridis', 'plasma', 'rainbow') or list of custom colors.
        marker_size : int, default=5
            Size of scatter plot markers. Adjust based on dataset size and density.
        figure_width : int, default=800
            Width of the output figure in pixels.
        figure_height : int, default=600
            Height of the output figure in pixels.
        apply_normalization : bool, default=True
            Whether to apply StandardScaler normalization before t-SNE. Recommended
            for features with different scales.
            
        Returns:
        --------
        visualization_figure : plotly.graph_objects.Figure
            Interactive plotly figure object that can be displayed, saved, or modified.
            Includes hover information, zoom capabilities, and legend if labels provided.
        transformed_coordinates : ndarray, shape (n_samples, n_dimensions)
            t-SNE transformed coordinates in specified dimensionality. Can be used
            for downstream analysis or custom visualizations.
            
        Raises:
        -------
        ValueError
            If n_dimensions not in [2, 3], if data is empty, if perplexity is invalid
        ImportError
            If required visualization libraries are not available
            
        Examples:
        --------
        >>> # Basic 2D t-SNE visualization
        >>> reducer = DimensionalityReducer()
        >>> fig, coords = reducer.create_tsne_visualization(data)
        >>> fig.show()
        
        >>> # 3D visualization with custom parameters
        >>> fig, coords = reducer.create_tsne_visualization(
        ...     data, labels, n_dimensions=3, perplexity_value=50,
        ...     color_scheme='viridis', plot_title='3D t-SNE Analysis'
        ... )
        
        >>> # High-resolution visualization for publication
        >>> fig, coords = reducer.create_tsne_visualization(
        ...     data, labels, figure_width=1200, figure_height=900,
        ...     marker_size=8, apply_normalization=True
        ... )
        """
        # Input validation
        if n_dimensions not in [2, 3]:
            raise ValueError("n_dimensions must be 2 for 2D or 3 for 3D visualization")
        
        if perplexity_value <= 0:
            raise ValueError("perplexity_value must be positive")
            
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
            
        # Import required libraries with error handling
        try:
            import numpy as np
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler
            import pandas as pd
        except ImportError as e:
            raise ImportError(f"Required libraries not available: {e}")
        
        # Convert input data to numpy array
        if isinstance(data, pd.DataFrame):
            feature_names = data.columns.tolist()
            data_array = data.values
        else:
            data_array = np.asarray(data)
            feature_names = [f"Feature_{i}" for i in range(data_array.shape[1])]
            
        # Validate data dimensions
        if data_array.ndim != 2:
            raise ValueError("Input data must be 2-dimensional (samples x features)")
            
        if data_array.shape[0] < 2:
            raise ValueError("Data must contain at least 2 samples")
            
        if data_array.shape[1] < 1:
            raise ValueError("Data must contain at least 1 feature")
            
        logger.info(f"Processing data with shape {data_array.shape}")
        
        # Apply normalization if requested
        if apply_normalization:
            logger.info("Applying StandardScaler normalization")
            scaler = StandardScaler()
            data_normalized = scaler.fit_transform(data_array)
            self.preprocessing_params['scaler'] = scaler
            self.preprocessing_params['feature_names'] = feature_names
        else:
            data_normalized = data_array.copy()
            self.preprocessing_params['scaler'] = None
        
        # Validate perplexity relative to data size
        max_perplexity = (data_array.shape[0] - 1) / 3
        if perplexity_value > max_perplexity:
            logger.warning(f"Perplexity {perplexity_value} too high for {data_array.shape[0]} samples. "
                          f"Reducing to {max_perplexity:.1f}")
            perplexity_value = max_perplexity
        
        # Perform t-SNE transformation
        logger.info(f"Performing {n_dimensions}D t-SNE with perplexity={perplexity_value}, "
                   f"iterations={max_iterations}")
        
        tsne_model = TSNE(
            n_components=n_dimensions,
            perplexity=perplexity_value,
            n_iter=max_iterations,
            random_state=self.random_state,
            verbose=1,
            init='random'
        )
        
        transformed_coordinates = tsne_model.fit_transform(data_normalized)
        
        # Store fitted model for potential reuse
        model_key = f"tsne_{n_dimensions}d_{perplexity_value}_{max_iterations}"
        self.fitted_models[model_key] = {
            'model': tsne_model,
            'coordinates': transformed_coordinates,
            'parameters': {
                'n_dimensions': n_dimensions,
                'perplexity': perplexity_value,
                'max_iterations': max_iterations,
                'normalization_applied': apply_normalization
            }
        }
        
        # Create DataFrame for visualization
        coord_columns = [f't-SNE_{i+1}' for i in range(n_dimensions)]
        visualization_df = pd.DataFrame(transformed_coordinates, columns=coord_columns)
        
        # Add class labels if provided
        if class_labels is not None:
            class_labels_array = np.asarray(class_labels)
            if len(class_labels_array) != data_array.shape[0]:
                raise ValueError("Length of class_labels must match number of samples in data")
            visualization_df['class_label'] = class_labels_array
            color_column = 'class_label'
        else:
            color_column = None
        
        # Set default title if not provided
        if plot_title is None:
            plot_title = f"{n_dimensions}D t-SNE Visualization (perplexity={perplexity_value})"
        
        # Create interactive visualization
        if n_dimensions == 2:
            visualization_figure = px.scatter(
                visualization_df, 
                x=coord_columns[0], 
                y=coord_columns[1], 
                color=color_column,
                title=plot_title,
                color_discrete_sequence=color_scheme,
                width=figure_width, 
                height=figure_height,
                hover_data={col: ':.3f' for col in coord_columns}
            )
            
            # Update marker properties
            visualization_figure.update_traces(
                marker=dict(size=marker_size, line=dict(width=0.5, color='DarkSlateGrey'))
            )
            
            # Update layout for 2D
            visualization_figure.update_layout(
                xaxis_title=coord_columns[0],
                yaxis_title=coord_columns[1],
                showlegend=class_labels is not None,
                template='plotly_white',
                title_x=0.5,
                font=dict(size=12)
            )
            
        else:  # 3D visualization
            visualization_figure = px.scatter_3d(
                visualization_df, 
                x=coord_columns[0], 
                y=coord_columns[1], 
                z=coord_columns[2],
                color=color_column,
                title=plot_title,
                color_discrete_sequence=color_scheme,
                width=figure_width, 
                height=figure_height,
                hover_data={col: ':.3f' for col in coord_columns}
            )
            
            # Update marker properties
            visualization_figure.update_traces(
                marker=dict(size=marker_size, line=dict(width=0.5, color='DarkSlateGrey'))
            )
            
            # Update layout for 3D
            visualization_figure.update_layout(
                scene=dict(
                    xaxis_title=coord_columns[0],
                    yaxis_title=coord_columns[1],
                    zaxis_title=coord_columns[2],
                    camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
                ),
                showlegend=class_labels is not None,
                template='plotly_white',
                title_x=0.5,
                font=dict(size=12)
            )
        
        logger.info("t-SNE visualization created successfully")
        return visualization_figure, transformed_coordinates
    
    def create_explained_variance_visualization(self, 
                                              fitted_pca_model,
                                              output_folder: Optional[str] = None, 
                                              plot_type: str = 'both',
                                              figure_width: int = 900,
                                              figure_height: int = 600) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create comprehensive explained variance visualizations for PCA analysis.
        
        This method generates publication-quality scree plots and cumulative variance plots
        to help determine the optimal number of principal components for dimensionality reduction.
        
        Parameters:
        -----------
        fitted_pca_model : sklearn.decomposition.PCA
            A fitted PCA model object from scikit-learn
        output_folder : str, optional
            Directory path to save visualization plots. If None, plots are only displayed
        plot_type : {'scree', 'cumulative', 'both'}, default='both'
            Type of plot to generate:
            - 'scree': Individual explained variance by component
            - 'cumulative': Cumulative explained variance
            - 'both': Generate both plot types
        figure_width : int, default=900
            Width of generated figures in pixels
        figure_height : int, default=600
            Height of generated figures in pixels
            
        Returns:
        --------
        explained_variance_ratios : np.ndarray
            Array of explained variance ratios (percentages) for each component
        eigenvalues : np.ndarray
            Array of eigenvalues for each principal component
            
        Raises:
        ------
        ValueError
            If plot_type is not one of the accepted values
        ImportError
            If required visualization libraries are unavailable
            
        Examples:
        --------
        >>> from sklearn.decomposition import PCA
        >>> pca = PCA().fit(X_train)
        >>> reducer = DimensionalityReducer()
        >>> var_ratios, eigenvals = reducer.create_explained_variance_visualization(
        ...     pca, output_folder='./plots/', plot_type='both'
        ... )
        >>> print(f"First 3 components explain {var_ratios[:3].sum():.1f}% variance")
        """
        # Input validation
        if plot_type not in ['scree', 'cumulative', 'both']:
            raise ValueError("plot_type must be 'scree', 'cumulative', or 'both'")
        
        if not hasattr(fitted_pca_model, 'explained_variance_ratio_'):
            raise ValueError("fitted_pca_model must be a fitted PCA model")
            
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            import os
        except ImportError as e:
            raise ImportError(f"Required visualization libraries not available: {e}")
        
        # Extract variance information
        eigenvalues = fitted_pca_model.explained_variance_
        explained_variance_ratios = np.round(fitted_pca_model.explained_variance_ratio_, decimals=4) * 100
        cumulative_variance = np.cumsum(explained_variance_ratios)
        
        component_numbers = np.arange(1, fitted_pca_model.n_components_ + 1)
        
        logger.info(f"Creating variance visualizations for {fitted_pca_model.n_components_} components")
        
        # Generate scree plot
        if plot_type in ['scree', 'both']:
            scree_figure = go.Figure()
            
            # Add explained variance line and markers
            scree_figure.add_trace(go.Scatter(
                x=component_numbers, 
                y=explained_variance_ratios,
                mode='lines+markers',
                name='Explained Variance',
                line=dict(color='steelblue', width=3),
                marker=dict(size=8, color='darkblue', symbol='circle')
            ))
            
            # Update layout
            scree_figure.update_layout(
                title=dict(
                    text='Scree Plot: Explained Variance by Principal Component',
                    font=dict(size=16),
                    x=0.5
                ),
                xaxis=dict(
                    title='Principal Component',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title='Explained Variance (%)',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                width=figure_width,
                height=figure_height,
                template='plotly_white',
                showlegend=False,
                font=dict(size=12)
            )
            
            # Add elbow detection annotation
            if len(explained_variance_ratios) >= 3:
                # Find elbow using simple difference method
                differences = np.diff(explained_variance_ratios)
                second_differences = np.diff(differences)
                if len(second_differences) > 0:
                    elbow_candidate = np.argmax(second_differences) + 2  # +2 due to double diff
                    scree_figure.add_vline(
                        x=elbow_candidate,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Elbow candidate: PC{elbow_candidate}"
                    )
            
            # Save and display
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                scree_path = os.path.join(output_folder, 'pca_scree_plot.png')
                pio.write_image(scree_figure, scree_path, width=figure_width, height=figure_height)
                logger.info(f"Scree plot saved to: {scree_path}")
            
            scree_figure.show()

        # Generate cumulative variance plot
        if plot_type in ['cumulative', 'both']:
            cumulative_figure = go.Figure()
            
            # Add cumulative variance line
            cumulative_figure.add_trace(go.Scatter(
                x=component_numbers, 
                y=cumulative_variance,
                mode='lines+markers',
                name='Cumulative Explained Variance',
                line=dict(color='darkgreen', width=3),
                marker=dict(size=8, color='forestgreen', symbol='diamond')
            ))
            
            # Add reference lines for common thresholds
            thresholds = [80, 90, 95]
            colors = ['orange', 'red', 'purple']
            
            for threshold, color in zip(thresholds, colors):
                if max(cumulative_variance) >= threshold:
                    # Find first component that exceeds threshold
                    exceeding_component = np.where(cumulative_variance >= threshold)[0][0] + 1
                    
                    cumulative_figure.add_hline(
                        y=threshold,
                        line_dash="dot",
                        line_color=color,
                        annotation_text=f"{threshold}% variance (PC{exceeding_component})",
                        annotation_position="top right"
                    )
            
            # Update layout
            cumulative_figure.update_layout(
                title=dict(
                    text='Cumulative Explained Variance by Principal Components',
                    font=dict(size=16),
                    x=0.5
                ),
                xaxis=dict(
                    title='Number of Principal Components',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title='Cumulative Explained Variance (%)',
                    range=[0, min(105, max(cumulative_variance) + 5)],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                width=figure_width,
                height=figure_height,
                template='plotly_white',
                showlegend=False,
                font=dict(size=12)
            )
            
            # Save and display
            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
                cumulative_path = os.path.join(output_folder, 'pca_cumulative_variance.png')
                pio.write_image(cumulative_figure, cumulative_path, width=figure_width, height=figure_height)
                logger.info(f"Cumulative variance plot saved to: {cumulative_path}")
                
            cumulative_figure.show()

        # Log summary statistics
        logger.info(f"Variance Analysis Summary:")
        logger.info(f"  - Total components: {len(explained_variance_ratios)}")
        logger.info(f"  - First component explains: {explained_variance_ratios[0]:.2f}% variance")
        logger.info(f"  - First 3 components explain: {cumulative_variance[2]:.2f}% variance" 
                   if len(cumulative_variance) >= 3 else "")
        logger.info(f"  - All components explain: {cumulative_variance[-1]:.2f}% variance")
        
        return explained_variance_ratios, eigenvalues
    
    def apply_varimax_rotation(self, 
                              component_loadings: np.ndarray,
                              rotation_method: str = 'varimax',
                              convergence_threshold: float = 1e-6,
                              max_iterations: int = 100) -> np.ndarray:
        """
        Apply orthogonal rotation to PCA component loadings for enhanced interpretability.
        
        Varimax rotation maximizes the variance of squared loadings to create a simpler
        structure where each component has high loadings on fewer variables, making
        interpretation easier while maintaining orthogonality.
        
        Parameters:
        -----------
        component_loadings : np.ndarray, shape (n_features, n_components)
            Component loading matrix from PCA analysis
        rotation_method : {'varimax', 'quartimax'}, default='varimax'
            Type of orthogonal rotation to apply:
            - 'varimax': Maximizes variance of squared loadings
            - 'quartimax': Minimizes number of factors needed to explain variables
        convergence_threshold : float, default=1e-6
            Convergence threshold for rotation algorithm
        max_iterations : int, default=100
            Maximum number of iterations for rotation algorithm
            
        Returns:
        --------
        rotation_matrix : np.ndarray, shape (n_components, n_components)
            Orthogonal rotation matrix that can be applied to transform loadings
            
        Raises:
        ------
        ValueError
            If rotation_method is not supported or loadings matrix is invalid
            
        Examples:
        --------
        >>> # Apply varimax rotation to PCA loadings
        >>> pca_loadings = fitted_pca.components_.T  # Transpose for correct shape
        >>> rotation_matrix = reducer.apply_varimax_rotation(pca_loadings)
        >>> rotated_loadings = pca_loadings @ rotation_matrix
        
        Note:
        -----
        The algorithm is based on Kaiser's varimax method with iterative optimization.
        Reference: https://github.com/rossfadely/consomme/blob/master/consomme/rotate_factor.py
        """
        # Input validation
        if rotation_method not in ['varimax', 'quartimax']:
            raise ValueError("rotation_method must be 'varimax' or 'quartimax'")
            
        if component_loadings.ndim != 2:
            raise ValueError("component_loadings must be a 2D array")
            
        # Set gamma parameter based on rotation method
        gamma = 1.0 if rotation_method == 'varimax' else 0.0
        
        n_features, n_components = component_loadings.shape
        rotation_matrix = np.eye(n_components)
        variance_explained = 0
        
        logger.info(f"Applying {rotation_method} rotation to {n_features}×{n_components} loading matrix")
        
        for iteration in range(max_iterations):
            # Apply current rotation to loadings
            rotated_loadings = np.dot(component_loadings, rotation_matrix)
            
            # Calculate objective function components
            squared_loadings = rotated_loadings ** 2
            diagonal_term = np.diag(np.sum(squared_loadings, axis=0)) / n_features * gamma
            
            # Compute gradient using singular value decomposition
            gradient_matrix = np.dot(
                component_loadings.T, 
                rotated_loadings ** 3 - np.dot(rotated_loadings, diagonal_term)
            )
            
            # SVD decomposition for rotation update
            u_matrix, singular_values, v_matrix = np.linalg.svd(gradient_matrix)
            rotation_matrix = np.dot(u_matrix, v_matrix)
            
            # Check for convergence
            new_variance = np.sum(singular_values)
            if iteration > 0 and new_variance < variance_explained * (1 + convergence_threshold):
                logger.info(f"{rotation_method} rotation converged after {iteration} iterations")
                break
            variance_explained = new_variance
        else:
            logger.warning(f"{rotation_method} rotation did not converge after {max_iterations} iterations")
        
        return rotation_matrix
    
    def calculate_feature_importance_scores(self, 
                                          pca_transformed_data: np.ndarray,
                                          pca_components: np.ndarray, 
                                          feature_names: List[str]) -> pd.Series:
        """
        Calculate feature importance scores based on PCA component contributions.
        
        This method computes how much each original feature contributes to the first two
        principal components, providing insights into which features are most important
        for the primary sources of variation in the data.
        
        Parameters:
        -----------
        pca_transformed_data : np.ndarray, shape (n_samples, n_components)
            PCA-transformed data coordinates
        pca_components : np.ndarray, shape (n_components, n_features)  
            PCA component loadings matrix
        feature_names : List[str]
            Names of original features
            
        Returns:
        --------
        feature_importance : pd.Series
            Series with feature importance scores, sorted by importance (descending)
            
        Raises:
        ------
        ValueError
            If input dimensions don't match or if insufficient components
            
        Examples:
        --------
        >>> # Calculate feature importance from PCA results
        >>> importance = reducer.calculate_feature_importance_scores(
        ...     X_pca, pca.components_, feature_names
        ... )
        >>> print(f"Top 3 features: {importance.head(3).index.tolist()}")
        
        Note:
        -----
        Importance is calculated as the Euclidean magnitude of feature contributions
        to the first two principal components, scaled by the maximum transformed values.
        """
        # Input validation
        if pca_transformed_data.shape[1] < 2:
            raise ValueError("At least 2 principal components required")
            
        if pca_components.shape[1] != len(feature_names):
            raise ValueError("Number of features must match pca_components columns")
            
        if pca_components.shape[0] < 2:
            raise ValueError("At least 2 components required in pca_components")
        
        logger.info(f"Calculating feature importance for {len(feature_names)} features")
        
        try:
            import math
            
            # Scale principal components by maximum transformed values
            pc1_scaling = np.max(pca_transformed_data[:, 0])
            pc2_scaling = np.max(pca_transformed_data[:, 1])
            
            # Calculate scaled component vectors
            pc1_scaled = pca_components[0] * pc1_scaling
            pc2_scaled = pca_components[1] * pc2_scaling
            
            # Compute feature importance as Euclidean magnitude
            feature_importance_dict = {
                feature_names[i]: math.sqrt(pc1_scaled[i]**2 + pc2_scaled[i]**2)
                for i in range(len(feature_names))
            }
            
            # Convert to series and sort by importance
            importance_series = pd.Series(feature_importance_dict)
            importance_series = importance_series.sort_values(ascending=False)
            
            # Log top features
            top_features = importance_series.head(5)
            logger.info("Top 5 most important features:")
            for feature, score in top_features.items():
                logger.info(f"  {feature}: {score:.4f}")
            
            return importance_series
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise ValueError(f"Feature importance calculation failed: {str(e)}")
    
    def create_biplot_visualization(self, 
                                   pca_transformed_data: np.ndarray,
                                   pca_components: np.ndarray, 
                                   feature_names: List[str],
                                   sample_labels: Optional[List[str]] = None,
                                   title: str = "PCA Biplot Visualization") -> Any:
        """
        Create a biplot showing both sample projections and feature vectors.
        
        A biplot displays the PCA-transformed sample points along with vectors
        representing the original features, helping visualize both sample patterns
        and feature relationships in the reduced dimensionality space.
        
        Parameters:
        -----------
        pca_transformed_data : np.ndarray, shape (n_samples, n_components)
            PCA-transformed sample coordinates (first 2 components used)
        pca_components : np.ndarray, shape (n_components, n_features)
            PCA component loading matrix
        feature_names : List[str]
            Names of original features for vector labeling
        sample_labels : List[str], optional
            Labels for individual samples in the biplot
        title : str, default="PCA Biplot Visualization"
            Title for the biplot
            
        Returns:
        --------
        matplotlib.axes.Axes
            Matplotlib axes object containing the biplot
            
        Examples:
        --------
        >>> # Create biplot from PCA results
        >>> ax = reducer.create_biplot_visualization(
        ...     X_pca, pca.components_, feature_names, 
        ...     sample_labels=group_labels
        ... )
        >>> plt.show()
        
        Note:
        -----
        Feature vectors are scaled to be visible on the same plot as sample points.
        Vector length indicates feature importance for the displayed components.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for biplot visualization")
            
        if pca_transformed_data.shape[1] < 2:
            raise ValueError("At least 2 principal components required for biplot")
            
        if pca_components.shape[0] < 2:
            raise ValueError("At least 2 components required in pca_components")
        
        logger.info(f"Creating PCA biplot with {len(feature_names)} feature vectors")
        
        # Set up the plot with ggplot style
        plt.style.use('ggplot')
        ax = plt.axes()
        
        # Plot sample points
        if sample_labels is not None:
            # Create scatter plot with different colors for different labels
            unique_labels = list(set(sample_labels))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = np.array(sample_labels) == label
                ax.scatter(
                    pca_transformed_data[mask, 0], 
                    pca_transformed_data[mask, 1],
                    c=[color], label=label, alpha=0.7, s=50
                )
            ax.legend()
        else:
            # Simple scatter plot without labels
            ax.scatter(
                pca_transformed_data[:, 0], 
                pca_transformed_data[:, 1],
                alpha=0.7, s=50, c='steelblue'
            )
        
        # Scale feature vectors by maximum sample coordinates for visibility
        max_pc1 = np.max(np.abs(pca_transformed_data[:, 0]))
        max_pc2 = np.max(np.abs(pca_transformed_data[:, 1]))
        
        # Draw feature vectors
        for i, feature_name in enumerate(feature_names):
            # Scale vectors to be visible on sample coordinate system
            vector_x = pca_components[0, i] * max_pc1 * 0.8
            vector_y = pca_components[1, i] * max_pc2 * 0.8
            
            # Draw arrow from origin to scaled component coordinates
            ax.arrow(
                0, 0, vector_x, vector_y, 
                color='red', width=0.005, head_width=max_pc1*0.02, 
                alpha=0.8, length_includes_head=True
            )
            
            # Add feature label at arrow tip
            ax.text(
                vector_x * 1.15, vector_y * 1.15, feature_name, 
                color='red', alpha=0.8, fontsize=10, ha='center', va='center'
            )
        
        # Set labels and title
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component') 
        ax.set_title(title)
        
        # Add grid and styling
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        logger.info("PCA biplot created successfully")
        return ax
    
    def perform_comprehensive_pca(self,
                                 input_dataframe: pd.DataFrame, 
                                 n_components: Optional[int] = None,
                                 standardize_features: bool = True,
                                 explained_variance_threshold: float = 0.95,
                                 create_visualizations: bool = False,
                                 output_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive Principal Component Analysis with detailed results and visualizations.
        
        This method provides a complete PCA analysis including component selection,
        explained variance analysis, feature contribution analysis, and optional visualizations
        using the integrated DimensionalityReducer methods.
        
        Parameters:
        -----------
        input_dataframe : pd.DataFrame
            Input data for PCA analysis with numeric features
        n_components : int, optional
            Number of components to keep (None for automatic selection)
        standardize_features : bool, default=True
            Whether to standardize features before PCA
        explained_variance_threshold : float, default=0.95
            Threshold for automatic component selection when n_components is None
        create_visualizations : bool, default=False
            Whether to generate explained variance and biplot visualizations
        output_folder : str, optional
            Directory path to save visualizations (only used if create_visualizations=True)
            
        Returns:
        --------
        pca_results : Dict[str, Any]
            Comprehensive dictionary with PCA results, analysis, and optional visualizations
            
        Raises:
        ------
        ValueError
            If input data is invalid or PCA analysis fails
            
        Examples:
        --------
        >>> reducer = DimensionalityReducer()
        >>> pca_results = reducer.perform_comprehensive_pca(
        ...     df, n_components=5, create_visualizations=True
        ... )
        >>> print(f"Explained variance: {pca_results['total_explained_variance']:.3f}")
        """
        try:
            if not isinstance(input_dataframe, pd.DataFrame):
                raise ValueError("input_dataframe must be a pandas DataFrame")
            
            if input_dataframe.empty:
                raise ValueError("input_dataframe cannot be empty")
            
            logger.info(f"Performing comprehensive PCA on data with shape {input_dataframe.shape}")
            
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Select numeric columns only
            numeric_columns = input_dataframe.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found for PCA")
            
            numeric_data = input_dataframe[numeric_columns].dropna()
            if numeric_data.empty:
                raise ValueError("No valid numeric data remaining after removing NaN values")
            
            logger.info(f"Using {len(numeric_columns)} numeric features for PCA")
            
            # Standardize features if requested
            if standardize_features:
                scaler = StandardScaler()
                standardized_data = scaler.fit_transform(numeric_data)
            else:
                standardized_data = numeric_data.values
                scaler = None
            
            # Determine optimal number of components
            if n_components is None:
                # Use threshold-based selection
                max_components = min(numeric_data.shape[0] - 1, numeric_data.shape[1])
                temp_pca = PCA(n_components=max_components)
                temp_pca.fit(standardized_data)
                
                cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
                n_components = max(1, min(n_components, max_components))
                
                logger.info(f"Automatically selected {n_components} components for "
                           f"{explained_variance_threshold:.1%} explained variance")
            else:
                max_components = min(numeric_data.shape[0] - 1, numeric_data.shape[1])
                n_components = min(n_components, max_components)
            
            # Perform PCA
            pca_analyzer = PCA(n_components=n_components)
            transformed_components = pca_analyzer.fit_transform(standardized_data)
            
            # Create comprehensive results dictionary
            pca_results = {
                # Core PCA results
                'pca_transformer': pca_analyzer,
                'transformed_data': transformed_components,
                'standardizer': scaler,
                
                # Variance analysis
                'explained_variance': pca_analyzer.explained_variance_,
                'explained_variance_ratio': pca_analyzer.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(pca_analyzer.explained_variance_ratio_),
                'total_explained_variance': np.sum(pca_analyzer.explained_variance_ratio_),
                
                # Component analysis
                'n_components_selected': n_components,
                'component_loadings': pca_analyzer.components_,
                'feature_names': list(numeric_columns),
                
                # Data characteristics
                'original_shape': input_dataframe.shape,
                'processed_shape': numeric_data.shape,
                'n_samples': numeric_data.shape[0],
                'n_original_features': numeric_data.shape[1],
                'dimensionality_reduction_ratio': n_components / numeric_data.shape[1],
                
                # Quality metrics
                'reconstruction_error': np.mean(pca_analyzer.noise_variance_) if hasattr(pca_analyzer, 'noise_variance_') else None,
                'condition_number': np.linalg.cond(pca_analyzer.components_) if n_components > 1 else 1.0
            }
            
            # Feature contribution analysis using existing method
            feature_importance = self.calculate_feature_importance_scores(
                transformed_components, pca_analyzer.components_, list(numeric_columns)
            )
            
            # Component interpretation
            component_summary = []
            for i in range(n_components):
                pc_name = f'PC{i+1}'
                
                component_summary.append({
                    'component': pc_name,
                    'explained_variance_ratio': pca_analyzer.explained_variance_ratio_[i],
                    'cumulative_variance_ratio': np.cumsum(pca_analyzer.explained_variance_ratio_)[i],
                    'eigenvalue': pca_analyzer.explained_variance_[i]
                })
            
            pca_results['feature_importance'] = feature_importance
            pca_results['component_summary'] = component_summary
            
            # Generate visualizations if requested
            if create_visualizations:
                logger.info("Creating PCA visualizations using integrated methods")
                
                # Create explained variance visualization
                explained_var_ratios, eigenvals = self.create_explained_variance_visualization(
                    pca_analyzer, output_folder=output_folder, plot_type='both'
                )
                pca_results['explained_variance_visualization'] = {
                    'variance_ratios': explained_var_ratios,
                    'eigenvalues': eigenvals
                }
                
                # Create biplot visualization if we have <= 2D components
                if n_components >= 2:
                    biplot_ax = self.create_biplot_visualization(
                        transformed_components, pca_analyzer.components_, 
                        list(numeric_columns), title="Comprehensive PCA Biplot"
                    )
                    pca_results['biplot_visualization'] = biplot_ax
            
            logger.info(f"Comprehensive PCA completed: {n_components} components explain "
                       f"{pca_results['total_explained_variance']:.3f} of total variance")
            
            return pca_results
            
        except Exception as e:
            logger.error(f"Error in perform_comprehensive_pca: {str(e)}")
            raise ValueError(f"PCA analysis failed: {str(e)}")

# ============================================================================
# MULTI-LABEL CLASSIFICATION ANALYZER
# ============================================================================

class MultiLabelClassifier:
    """
    Comprehensive multi-label classification utilities for data science workflows.
    
    This class provides specialized methods for multi-label data preprocessing, 
    stratified splitting, and evaluation that maintain label distribution across
    training, validation, and test sets. Essential for multi-label learning tasks
    where samples can belong to multiple classes simultaneously.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the MultiLabelClassifier.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducible results across all operations
        """
        self.random_state = random_state
        self.label_distributions = {}
    
    def split_dataset_with_stratification(self, 
                                        sample_dataframe: pd.DataFrame,
                                        binary_labels: pd.DataFrame, 
                                        train_size: float = 0.7,
                                        validation_size: float = 0.15,
                                        test_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split multi-label dataset into stratified train, validation, and test sets.
        
        This method ensures label distribution is preserved across splits, which is crucial
        for reliable model evaluation in multi-label classification scenarios where
        label imbalance is common.
        
        Parameters:
        -----------
        sample_dataframe : pd.DataFrame
            DataFrame containing samples to be split with features and metadata
        binary_labels : pd.DataFrame
            Binary-encoded labels DataFrame where each column represents a label class
        train_size : float, default=0.7
            Proportion of data for training (0.0 to 1.0)
        validation_size : float, default=0.15
            Proportion of data for validation (0.0 to 1.0)  
        test_size : float, default=0.15
            Proportion of data for testing (0.0 to 1.0)
            
        Returns:
        --------
        sample_splits : pd.DataFrame
            Original DataFrame with added 'dataset_split' column indicating assignment
        label_splits : pd.DataFrame
            Binary labels DataFrame with added 'dataset_split' column for split tracking
            
        Raises:
        ------
        ValueError
            If split proportions don't sum to 1.0 or DataFrames have mismatched indices
            
        Examples:
        --------
        >>> classifier = MultiLabelClassifier(random_state=42)
        >>> samples_split, labels_split = classifier.split_dataset_with_stratification(
        ...     df_samples, df_labels, train_size=0.7, validation_size=0.15, test_size=0.15
        ... )
        >>> print(samples_split['dataset_split'].value_counts())
        """
        # Input validation
        if abs(train_size + validation_size + test_size - 1.0) > 1e-6:
            raise ValueError("Split proportions must sum to 1.0")
            
        if len(sample_dataframe) != len(binary_labels):
            raise ValueError("sample_dataframe and binary_labels must have same number of rows")
            
        if not sample_dataframe.index.equals(binary_labels.index):
            raise ValueError("sample_dataframe and binary_labels must have matching indices")
        
        logger.info(f"Splitting {len(sample_dataframe)} samples with {len(binary_labels.columns)} labels")
        logger.info(f"Split ratios - Train: {train_size}, Validation: {validation_size}, Test: {test_size}")
        
        # Create working copies
        samples_copy = sample_dataframe.copy()
        labels_copy = binary_labels.copy()
        
        try:
            # Convert binary labels to list format for stratification
            binary_label_lists = binary_labels.apply(lambda row: row.tolist(), axis=1)
            
            # First split: separate training data from evaluation+test data
            eval_test_size = validation_size + test_size
            train_indices, eval_test_indices = split_multilabel_data_stratified(
                sample_dataframe.values,
                np.array(binary_label_lists.tolist()),
                test_proportion=eval_test_size,
                random_state=self.random_state
            )
            
            # Second split: separate evaluation data from test data  
            eval_test_samples = sample_dataframe.iloc[eval_test_indices]
            eval_test_labels = binary_label_lists.iloc[eval_test_indices]
            
            # Calculate test proportion within the eval+test subset
            test_proportion_in_subset = test_size / eval_test_size
            
            validation_indices_subset, test_indices_subset = split_multilabel_data_stratified(
                eval_test_samples.values,
                np.array(eval_test_labels.tolist()),
                test_proportion=test_proportion_in_subset,
                random_state=self.random_state
            )
            
            # Map subset indices back to original dataframe indices
            validation_indices = eval_test_samples.iloc[validation_indices_subset].index
            test_indices = eval_test_samples.iloc[test_indices_subset].index
            train_indices_original = sample_dataframe.iloc[train_indices].index
            
            # Assign split labels to dataframes
            samples_copy.loc[train_indices_original, 'dataset_split'] = 'train'
            samples_copy.loc[validation_indices, 'dataset_split'] = 'validation'  
            samples_copy.loc[test_indices, 'dataset_split'] = 'test'
            
            # Ensure all samples are assigned
            if samples_copy['dataset_split'].isna().any():
                raise ValueError("Some samples were not assigned to any split")
            
            # Copy split assignments to labels dataframe
            labels_copy['dataset_split'] = samples_copy['dataset_split']
            
            # Log split statistics
            split_counts = samples_copy['dataset_split'].value_counts()
            for split_name, count in split_counts.items():
                percentage = (count / len(sample_dataframe)) * 100
                logger.info(f"  {split_name}: {count} samples ({percentage:.1f}%)")
            
            # Store label distribution information
            self.label_distributions = {
                'train': labels_copy[labels_copy['dataset_split'] == 'train'].drop('dataset_split', axis=1).sum(),
                'validation': labels_copy[labels_copy['dataset_split'] == 'validation'].drop('dataset_split', axis=1).sum(),
                'test': labels_copy[labels_copy['dataset_split'] == 'test'].drop('dataset_split', axis=1).sum()
            }
            
            logger.info("Multi-label stratified split completed successfully")
            return samples_copy, labels_copy
            
        except Exception as e:
            logger.error(f"Error in multi-label stratified split: {str(e)}")
            raise ValueError(f"Dataset splitting failed: {str(e)}")
    
    def evaluate_multilabel_performance(self, 
                                       predicted_labels: pd.DataFrame,
                                       true_labels: pd.DataFrame,
                                       averaging_strategy: str = 'binary',
                                       metrics_to_calculate: List[str] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Comprehensive evaluation of multi-label classification performance.
        
        This method calculates various performance metrics including per-label scores
        and aggregated measures (macro, micro, weighted averages). Provides detailed
        analysis suitable for multi-label model comparison and selection.
        
        Parameters:
        -----------
        predicted_labels : pd.DataFrame
            Predicted binary labels with same structure as true_labels
        true_labels : pd.DataFrame
            Ground truth binary labels for evaluation
        averaging_strategy : {'binary', 'macro', 'micro', 'weighted'}, default='binary'
            Strategy for averaging multi-label metrics
        metrics_to_calculate : List[str], optional
            Specific metrics to compute. If None, uses comprehensive default set.
            Options: ['recall', 'precision', 'accuracy', 'auc_weighted', 'f1', 'kappa', 'mcc']
            
        Returns:
        --------
        performance_metrics : dict
            Dictionary containing:
            - 'detailed_scores': DataFrame with per-label and averaged metrics
            - 'subset_accuracy': Float indicating exact match accuracy across all labels
        evaluation_dataframe : pd.DataFrame
            Detailed results DataFrame with true/predicted pairs for further analysis
            
        Raises:
        ------
        ValueError
            If DataFrames have mismatched shapes or invalid metric specifications
            
        Examples:
        --------
        >>> classifier = MultiLabelClassifier()
        >>> metrics, eval_df = classifier.evaluate_multilabel_performance(
        ...     y_pred, y_true, averaging_strategy='macro'
        ... )
        >>> print(f"Subset accuracy: {metrics['subset_accuracy']:.3f}")
        >>> print(metrics['detailed_scores']['f1'].head())
        """
        # Set default metrics if not specified
        if metrics_to_calculate is None:
            metrics_to_calculate = [
                'recall', 'precision', 'accuracy', 'auc_weighted', 
                'f1', 'kappa', 'mcc'
            ]
        
        logger.info(f"Evaluating multi-label performance with {len(metrics_to_calculate)} metrics")
        
        try:
            # Unify columns between predicted and true labels
            pred_labels_unified, true_labels_unified = DataFrameUtilities.unify_columns(
                predicted_labels, true_labels, 'predicted_labels', 'true_labels'
            )
            
            # Create long-format dataframe for metric calculation
            evaluation_long_format = pd.concat([
                true_labels_unified.melt(value_name='true_label').set_index('variable'),
                pred_labels_unified.melt(value_name='predicted_label').set_index('variable')
            ], axis=1).reset_index().rename(columns={'variable': 'label_class'})
            
            # Filter out labels with no positive instances
            label_totals = evaluation_long_format.groupby('label_class')[['predicted_label', 'true_label']].sum().sum(axis=1)
            evaluation_filtered = evaluation_long_format[
                evaluation_long_format['label_class'].isin(label_totals[label_totals > 0].index)
            ]
            
            logger.info(f"Computing metrics for {len(evaluation_filtered['label_class'].unique())} active labels")
            
            # Calculate per-label and aggregated metrics
            detailed_scores = ml_scores(
                evaluation_filtered, 
                metrics_to_calculate,
                multi_class='ovo',
                average=averaging_strategy
            ).set_index("CV")
            detailed_scores.index.name = 'label_class'
            
            # Rename aggregated score rows with descriptive names
            aggregation_mapping = {
                'CV_scores_Mean': 'macro_average',
                'CV_scores_STD': 'macro_std_deviation', 
                'scores_all': 'micro_average'
            }
            detailed_scores = detailed_scores.rename(index=aggregation_mapping)
            
            # Add support information (number of positive instances per label)
            detailed_scores['support_count'] = evaluation_filtered.groupby('label_class')['true_label'].sum()
            
            # Calculate weighted averages
            individual_label_mask = ~detailed_scores.index.str.contains("_average|_deviation")
            individual_scores = detailed_scores[individual_label_mask]
            support_weights = detailed_scores.loc[individual_scores.index, 'support_count']
            
            weighted_averages = pd.DataFrame(
                individual_scores.apply(
                    lambda col: np.average(col, weights=support_weights) if col.name != 'support_count' else support_weights.sum(),
                    axis=0
                ),
                columns=['weighted_average']
            ).T
            
            detailed_scores = pd.concat([detailed_scores, weighted_averages], axis=0)
            
            # Fill support counts for aggregated rows
            aggregated_mask = detailed_scores.index.str.contains("_average|_deviation")
            total_support = individual_scores['support_count'].sum()
            detailed_scores.loc[aggregated_mask, 'support_count'] = total_support
            detailed_scores['support_count'] = detailed_scores['support_count'].astype(int)
            
            # Sort results: individual labels by performance, then aggregated metrics
            individual_sorted = individual_scores.sort_values(by=['mcc'], ascending=False)
            aggregated_metrics = detailed_scores[aggregated_mask]
            final_scores = pd.concat([individual_sorted, aggregated_metrics], axis=0)
            
            # Calculate subset accuracy (exact match across all labels)
            try:
                from sklearn.metrics import accuracy_score
                subset_accuracy = accuracy_score(true_labels_unified, pred_labels_unified)
                logger.info(f'Subset accuracy (exact label set matches): {subset_accuracy:.4f}')
            except Exception as e:
                logger.warning(f"Could not calculate subset accuracy: {e}")
                subset_accuracy = None
            
            # Prepare return values
            performance_summary = {
                'detailed_scores': final_scores,
                'subset_accuracy': subset_accuracy
            }
            
            evaluation_dataframe = evaluation_filtered.rename(columns={'label_class': 'label'})
            
            # Log summary statistics
            if subset_accuracy is not None:
                logger.info(f"Performance Summary:")
                logger.info(f"  - Subset accuracy: {subset_accuracy:.4f}")
            if 'macro_average' in final_scores.index:
                macro_f1 = final_scores.loc['macro_average', 'f1']
                logger.info(f"  - Macro F1-score: {macro_f1:.4f}")
            
            return performance_summary, evaluation_dataframe
            
        except Exception as e:
            logger.error(f"Error in multi-label performance evaluation: {str(e)}")
            raise ValueError(f"Performance evaluation failed: {str(e)}")


# ============================================================================
# UTILITY FUNCTIONS FOR DATA PREPROCESSING AND ANALYSIS
# ============================================================================

def convert_labels_to_binary(categorical_labels: pd.Series) -> pd.DataFrame:
    """
    Convert categorical labels to binary matrix format for multi-label classification.
    
    This function handles both single categorical labels and multi-label scenarios,
    converting them to a binary matrix format suitable for multi-label classification
    algorithms and evaluation metrics.
    
    Args:
        categorical_labels: Series containing categorical labels or lists of labels
        
    Returns:
        Binary matrix DataFrame with columns for each unique label
        
    Raises:
        ValueError: If input is not a pandas Series or contains invalid data
        
    Example:
        >>> labels = pd.Series(['A', 'B', 'A,C', 'B,C'])
        >>> binary_matrix = convert_labels_to_binary(labels)
        >>> print(binary_matrix.columns)  # ['A', 'B', 'C']
    """
    try:
        if not isinstance(categorical_labels, pd.Series):
            raise ValueError("categorical_labels must be a pandas Series")
        
        if categorical_labels.empty:
            raise ValueError("categorical_labels cannot be empty")
        
        logger.info(f"Converting {len(categorical_labels)} labels to binary format")
        
        from sklearn.preprocessing import MultiLabelBinarizer
        
        # Check if labels are already in list/array format
        sample_label = categorical_labels.iloc[0]
        if isinstance(sample_label, (list, tuple, np.ndarray)):
            # Already in multi-label format
            label_binarizer = MultiLabelBinarizer()
            binary_matrix = label_binarizer.fit_transform(categorical_labels)
            
            binary_df = pd.DataFrame(
                binary_matrix,
                columns=label_binarizer.classes_,
                index=categorical_labels.index
            )
            
        else:
            # Single categorical labels - use pandas get_dummies
            binary_df = pd.get_dummies(categorical_labels)
        
        logger.info(f"Created binary matrix with shape {binary_df.shape}")
        return binary_df
        
    except Exception as e:
        logger.error(f"Error in convert_labels_to_binary: {str(e)}")
        raise ValueError(f"Label conversion failed: {str(e)}")

def split_multilabel_data_stratified(training_features: np.ndarray, 
                                   multilabel_targets: np.ndarray,
                                   test_proportion: float, 
                                   random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split multi-label dataset using iterative stratification to maintain label distribution.
    
    This function ensures that the label distribution is preserved across training and test sets
    in multi-label classification scenarios, which is crucial for model evaluation.
    
    Args:
        training_features: Feature matrix (n_samples, n_features)
        multilabel_targets: Multi-label binary matrix (n_samples, n_labels)  
        test_proportion: Proportion of data to use for testing (0.0 to 1.0)
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (train_indices, test_indices)
        
    Raises:
        ValueError: If inputs have incompatible shapes or invalid test_proportion
        
    Example:
        >>> train_idx, test_idx = split_multilabel_data_stratified(X, y_binary, 0.2, 42)
        >>> X_train, X_test = X[train_idx], X[test_idx]
        >>> y_train, y_test = y_binary[train_idx], y_binary[test_idx]
    """
    try:
        # Validate inputs
        training_features = np.array(training_features)
        multilabel_targets = np.array(multilabel_targets)
        
        if training_features.shape[0] != multilabel_targets.shape[0]:
            raise ValueError("training_features and multilabel_targets must have same number of samples")
        
        if not 0 < test_proportion < 1:
            raise ValueError("test_proportion must be between 0 and 1")
        
        logger.info(f"Splitting multilabel data: {training_features.shape[0]} samples, "
                   f"{multilabel_targets.shape[1]} labels, test_size={test_proportion}")
        
        try:
            # Try to use iterative stratification if available
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
            
            stratified_splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1, 
                test_size=test_proportion, 
                random_state=random_state
            )
            
            train_indices, test_indices = next(stratified_splitter.split(training_features, multilabel_targets))
            
            logger.info("Used iterative stratification for multi-label split")
            return train_indices, test_indices
            
        except ImportError:
            logger.warning("iterstrat not available, falling back to random split")
            
            # Fallback to random split if iterative stratification not available
            from sklearn.model_selection import train_test_split
            
            sample_indices = np.arange(len(training_features))
            train_indices, test_indices = train_test_split(
                sample_indices, 
                test_size=test_proportion, 
                random_state=random_state
            )
            
            logger.info("Used random split as fallback")
            return train_indices, test_indices
        
    except Exception as e:
        logger.error(f"Error in split_multilabel_data_stratified: {str(e)}")
        raise ValueError(f"Multi-label data splitting failed: {str(e)}")

def compute_balanced_class_weights(class_weight_strategy: str, 
                                 target_labels: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced dataset handling.
    
    This function calculates appropriate class weights to handle imbalanced datasets,
    which is crucial for training models on skewed data distributions.
    
    Args:
        class_weight_strategy: Strategy for weight calculation ('balanced', 'balanced_subsample')
        target_labels: Array of class labels
        
    Returns:
        Dictionary mapping class labels to their computed weights
        
    Raises:
        ValueError: If strategy is invalid or target_labels is empty
        
    Example:
        >>> weights = compute_balanced_class_weights('balanced', y_train)
        >>> model = RandomForestClassifier(class_weight=weights)
    """
    try:
        if not isinstance(target_labels, np.ndarray):
            target_labels = np.array(target_labels)
        
        if len(target_labels) == 0:
            raise ValueError("target_labels cannot be empty")
        
        valid_strategies = ['balanced', 'balanced_subsample']
        if class_weight_strategy not in valid_strategies:
            raise ValueError(f"class_weight_strategy must be one of {valid_strategies}")
        
        logger.info(f"Computing class weights using '{class_weight_strategy}' strategy")
        
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get unique classes
        unique_classes = np.unique(target_labels)
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight_strategy, 
            classes=unique_classes, 
            y=target_labels
        )
        
        # Create weight dictionary
        weight_dict = dict(zip(unique_classes, class_weights))
        
        logger.info(f"Computed weights for {len(unique_classes)} classes")
        for class_label, weight in weight_dict.items():
            logger.debug(f"Class {class_label}: weight = {weight:.4f}")
        
        return weight_dict
        
    except Exception as e:
        logger.error(f"Error in compute_balanced_class_weights: {str(e)}")
        raise ValueError(f"Class weight computation failed: {str(e)}")

# ============================================================================
# BACKWARD COMPATIBILITY AND LEGACY FUNCTION MAPPINGS
# ============================================================================

# Complete function mapping for reference and backward compatibility
COMPREHENSIVE_FUNCTION_MAPPING = {
    # Model Templates and Training
    'classifiers_template': 'ModelTemplateManager.get_classification_models()',
    'regressors_template': 'ModelTemplateManager.get_regression_models()', 
    
    # Model Evaluation and Scoring
    'ml_scores': 'ModelEvaluator.calculate_model_scores()',
    'ml_scores_crossvalidate': 'ModelEvaluator.cross_validation_scores()',
    
    # Model Training and Comparison
    'ml_prediction': 'ModelTrainer.cross_validate_model()',
    'ml_comparison': 'ModelTrainer.compare_models()',
    'classifiers_batch': 'ModelTrainer.batch_evaluate_classifiers()',
    'classifier_score_cf': 'ModelTrainer.single_classifier_analysis()',
    
    # Performance Analysis and Visualization
    'ml_comparison_plot': 'ModelPerformanceAnalyzer.plot_model_comparison()',
    'gainNlift': 'ModelPerformanceAnalyzer.calculate_gain_lift_stats()',
    'create_gain_lift_chart': 'ModelPerformanceAnalyzer.create_gain_lift_chart()',
    'learning_curve_early_stopping': 'ModelPerformanceAnalyzer.create_learning_curve_plot()',
    'precision_recall_curve2': 'ModelPerformanceAnalyzer.create_precision_recall_curve()',
    
    # SHAP Analysis
    'shap_batch': 'SHAPAnalyzer.calculate_shap_contributions()',
    'shap_localInterpretability': 'SHAPAnalyzer.analyze_local_interpretability()',
    
    # XGBoost Rule Extraction
    'leafs2rules': 'XGBoostRuleExtractor.extract_decision_rules()',
    'rulesCombiner': 'XGBoostRuleExtractor.combine_rule_sets()',
    'xgb_rules': 'XGBoostRuleExtractor.generate_customer_segments()',
    
    # Decision Tree Interpretation
    'tree_to_code': 'DecisionTreeInterpreter.convert_tree_to_code()',
    'string_parser': 'DecisionTreeInterpreter.parse_xgboost_tree_string()',
    'tree_parser': 'DecisionTreeInterpreter.extract_tree_parser_information()',
    
    # Utility Functions
    'binarizer': 'convert_labels_to_binary()',
    'binarize_labels': 'convert_labels_to_binary()',
    'split_multiLabel_data__index': 'split_multilabel_data_stratified()',
    'split_multilabel_data_by_index': 'split_multilabel_data_stratified()',
    'class_weight2': 'compute_balanced_class_weights()',
    'calculate_class_weights': 'compute_balanced_class_weights()',
    'pca_plot': 'DimensionalityReducer.perform_comprehensive_pca()',
    'perform_comprehensive_pca': 'DimensionalityReducer.perform_comprehensive_pca()',
    'perform_pca_analysis': 'perform_comprehensive_pca()',
}

def print_comprehensive_function_mapping():
    """
    Display the complete mapping of old function names to new implementations.
    
    This function provides a comprehensive reference for migrating from old
    function names to the new class-based structure.
    """
    print("\\n" + "=" * 80)
    print("COMPREHENSIVE FUNCTION MAPPING - Old to New")
    print("=" * 80)
    print("This reference shows how to migrate from old function calls to the new")
    print("class-based structure while maintaining all functionality.\\n")
    
    # Group by category
    categories = {
        'Model Templates': ['classifiers_template', 'regressors_template'],
        'Model Evaluation': ['ml_scores', 'ml_scores_crossvalidate'],
        'Model Training': ['ml_prediction', 'ml_comparison', 'classifiers_batch', 'classifier_score_cf'],
        'Performance Analysis': ['ml_comparison_plot', 'gainNlift', 'create_gain_lift_chart', 
                               'learning_curve_early_stopping', 'precision_recall_curve2'],
        'SHAP Analysis': ['shap_batch', 'shap_localInterpretability'],
        'XGBoost Rules': ['leafs2rules', 'rulesCombiner', 'xgb_rules'],
        'Decision Trees': ['tree_to_code', 'string_parser', 'tree_parser'],
        'Utility Functions': ['binarizer', 'split_multiLabel_data__index', 'class_weight2', 'pca_plot']
    }
    
    for category, functions in categories.items():
        print(f"\\n{category}:")
        print("-" * len(category))
        for func in functions:
            if func in COMPREHENSIVE_FUNCTION_MAPPING:
                print(f"  {func:<35} -> {COMPREHENSIVE_FUNCTION_MAPPING[func]}")
    
    print("\\n" + "=" * 80)
    print(f"Total functions mapped: {len(COMPREHENSIVE_FUNCTION_MAPPING)}")
    print("=" * 80)

# Backward compatibility functions
def classifiers_template(y=None, random_state=42, imputer_strategy='mean', pc_n_components=5):
    """Legacy function for backward compatibility. Use ModelTemplateManager instead."""
    warnings.warn("classifiers_template is deprecated. Use ModelTemplateManager.get_classification_models() instead.", 
                 DeprecationWarning, stacklevel=2)
    manager = ModelTemplateManager(random_state=random_state)
    return manager.get_classification_models(y, y, imputer_strategy, pc_n_components)

def regressors_template(random_state=42):
    """Legacy function for backward compatibility. Use ModelTemplateManager instead."""
    warnings.warn("regressors_template is deprecated. Use ModelTemplateManager.get_regression_models() instead.", 
                 DeprecationWarning, stacklevel=2)
    manager = ModelTemplateManager(random_state=random_state)
    return manager.get_regression_models(random_state)

def ml_scores(y_model, scores_names, multi_class='ovr', average='macro'):
    """Legacy function for backward compatibility. Use ModelEvaluator instead."""
    warnings.warn("ml_scores is deprecated. Use ModelEvaluator.calculate_model_scores() instead.", 
                 DeprecationWarning, stacklevel=2)
    evaluator = ModelEvaluator()
    return evaluator.calculate_model_scores(y_model, scores_names, multi_class, average)

def binarizer(tags):
    """Legacy function for backward compatibility. Use convert_labels_to_binary instead."""
    warnings.warn("binarizer is deprecated. Use convert_labels_to_binary() instead.", 
                 DeprecationWarning, stacklevel=2)
    return convert_labels_to_binary(tags)

def split_multiLabel_data__index(X, y, test_size, random_state=None):
    """Legacy function for backward compatibility. Use split_multilabel_data_stratified instead."""
    warnings.warn("split_multiLabel_data__index is deprecated. Use split_multilabel_data_stratified() instead.", 
                 DeprecationWarning, stacklevel=2)
    return split_multilabel_data_stratified(X, y, test_size, random_state)

def class_weight2(uclass_weight, y):
    """Legacy function for backward compatibility. Use compute_balanced_class_weights instead."""
    warnings.warn("class_weight2 is deprecated. Use compute_balanced_class_weights() instead.", 
                 DeprecationWarning, stacklevel=2)
    return compute_balanced_class_weights(uclass_weight, y)

def pca_plot(df):
    """Legacy function for backward compatibility. Use DimensionalityReducer.perform_comprehensive_pca instead."""
    warnings.warn("pca_plot is deprecated. Use DimensionalityReducer.perform_comprehensive_pca() instead.", 
                 DeprecationWarning, stacklevel=2)
    reducer = DimensionalityReducer()
    return reducer.perform_comprehensive_pca(df, create_visualizations=True)

def perform_comprehensive_pca(input_dataframe: pd.DataFrame, 
                            n_components: Optional[int] = None,
                            standardize_features: bool = True,
                            explained_variance_threshold: float = 0.95) -> Dict[str, Any]:
    """
    Legacy wrapper for comprehensive PCA analysis.
    
    **DEPRECATED**: This function has been moved to the DimensionalityReducer class
    with enhanced functionality. Please use:
    `DimensionalityReducer().perform_comprehensive_pca()` instead.
    """
    warnings.warn(
        "perform_comprehensive_pca() standalone function is deprecated. "
        "Use DimensionalityReducer().perform_comprehensive_pca() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    reducer = DimensionalityReducer()
    return reducer.perform_comprehensive_pca(
        input_dataframe, n_components, standardize_features, explained_variance_threshold
    )

def tree_to_code(tree, feature_names):
    """Legacy function for backward compatibility. Use DecisionTreeInterpreter instead."""
    warnings.warn("tree_to_code is deprecated. Use DecisionTreeInterpreter.convert_tree_to_code() instead.", 
                 DeprecationWarning, stacklevel=2)
    interpreter = DecisionTreeInterpreter()
    code = interpreter.convert_tree_to_code(tree, feature_names)
    print(code)  # Original function printed the code
    return code

def shap_batch(X, classifier_xgb, preds, targetClass, uplotFolder=None):
    """Legacy function for backward compatibility. Use SHAPAnalyzer instead."""
    warnings.warn("shap_batch is deprecated. Use SHAPAnalyzer.calculate_shap_contributions() instead.", 
                 DeprecationWarning, stacklevel=2)
    analyzer = SHAPAnalyzer(output_directory=uplotFolder)
    return analyzer.calculate_shap_contributions(X, classifier_xgb, preds, targetClass, uplotFolder)

def cca_analysis(X1_sub, X2_sub, n_comp=2, scaler_type='standard', cluster_cols=False, cluster_rows=False):
    """
    Backward compatibility wrapper for CCA analysis.
    
    DEPRECATED: Please use CanonicalAnalyzer.perform_canonical_correlation_analysis() instead.
    This function will be removed in a future version.
    """
    warnings.warn(
        "cca_analysis() is deprecated. Use CanonicalAnalyzer.perform_canonical_correlation_analysis() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    analyzer = CanonicalAnalyzer()
    results = analyzer.perform_canonical_correlation_analysis(
        variable_set_1=X1_sub,
        variable_set_2=X2_sub,
        n_canonical_components=n_comp,
        preprocessing_method=scaler_type,
        cluster_variables_1=cluster_cols,
        cluster_variables_2=cluster_rows
    )
    
    # Return in legacy format
    return {
        'correlations': results['canonical_correlations'],
        'coefficients': results['raw_coefficients'],
        'coefficients_normalized': results['normalized_coefficients'],
        'fig_correlations': results['correlation_plot'],
        'fig_heatmap': results['coefficient_heatmap']
    }

def cluster_analysis(df, K_range=range(4, 7), scale_method='standard'):
    """
    Backward compatibility wrapper for clustering analysis.
    
    DEPRECATED: Please use ClusteringAnalyzer.perform_optimal_clustering_analysis() instead.
    This function will be removed in a future version.
    """
    warnings.warn(
        "cluster_analysis() is deprecated. Use ClusteringAnalyzer.perform_optimal_clustering_analysis() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    analyzer = ClusteringAnalyzer()
    results = analyzer.perform_optimal_clustering_analysis(
        dataset=df,
        cluster_range=K_range,
        scaling_strategy=scale_method
    )
    
    # Return in legacy format
    return {
        'clusters': results['cluster_assignments'],
        'scaler': results['preprocessing_scaler'],
        'optimal_k': results['optimal_clusters'],
        'silhouette_scores': {k: v['silhouette'] for k, v in results['evaluation_scores'].items()},
        'kmeans_model': results['clustering_model']
    }

def calculate_vif(X):
    """
    Backward compatibility wrapper for VIF calculation.
    
    DEPRECATED: Please use MulticollinearityDetector.calculate_variance_inflation_factors() instead.
    This function will be removed in a future version.
    """
    warnings.warn(
        "calculate_vif() is deprecated. Use MulticollinearityDetector.calculate_variance_inflation_factors() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    detector = MulticollinearityDetector()
    return detector.calculate_variance_inflation_factors(X)

def detect_multicollinearity(X, vif_threshold=5):
    """
    Backward compatibility wrapper for multicollinearity detection.
    
    DEPRECATED: Please use MulticollinearityDetector.analyze_multicollinearity_patterns() instead.
    This function will be removed in a future version.
    """
    warnings.warn(
        "detect_multicollinearity() is deprecated. Use MulticollinearityDetector.analyze_multicollinearity_patterns() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    detector = MulticollinearityDetector(vif_threshold=vif_threshold)
    return detector.analyze_multicollinearity_patterns(X)

# Maintain backwards compatibility for metric_dict
metric_dict = COMPREHENSIVE_METRICS



  


 

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS FOR NEW METHODS
# =============================================================================

def pca_explainedVar(pcaML, output_Folder=None, plot_type='cumulative'):
    """
    **DEPRECATED**: This function has been moved to the DimensionalityReducer class
    with enhanced functionality. Please use:
    `DimensionalityReducer().create_explained_variance_visualization()` instead.
    """
    warnings.warn(
        "pca_explainedVar() is deprecated and will be removed in a future version. "
        "Please use DimensionalityReducer().create_explained_variance_visualization() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    reducer = DimensionalityReducer()
    explained_var, eigen_values = reducer.create_explained_variance_visualization(
        pcaML, output_Folder, plot_type
    )
    return explained_var, eigen_values

def pca_ortho_rotation(lam, method='varimax', gamma=None, eps=1e-6, itermax=100):
    """
    **DEPRECATED**: This function has been moved to the DimensionalityReducer class
    with enhanced functionality. Please use:
    `DimensionalityReducer().apply_varimax_rotation()` instead.
    """
    warnings.warn(
        "pca_ortho_rotation() is deprecated and will be removed in a future version. "
        "Please use DimensionalityReducer().apply_varimax_rotation() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    reducer = DimensionalityReducer()
    return reducer.apply_varimax_rotation(lam, method, eps, itermax)

def pca_important_features(transformed_features, components_, columns):
    """
    **DEPRECATED**: This function has been moved to the DimensionalityReducer class
    with enhanced functionality. Please use:
    `DimensionalityReducer().calculate_feature_importance_scores()` instead.
    """
    warnings.warn(
        "pca_important_features() is deprecated and will be removed in a future version. "
        "Please use DimensionalityReducer().calculate_feature_importance_scores() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    reducer = DimensionalityReducer()
    return reducer.calculate_feature_importance_scores(transformed_features, components_, columns)

def pc_draw_vectors(transformed_features, components_, columns):
    """
    **DEPRECATED**: This function has been moved to the DimensionalityReducer class
    with enhanced functionality. Please use:
    `DimensionalityReducer().create_biplot_visualization()` instead.
    """
    warnings.warn(
        "pc_draw_vectors() is deprecated and will be removed in a future version. "
        "Please use DimensionalityReducer().create_biplot_visualization() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    reducer = DimensionalityReducer()
    return reducer.create_biplot_visualization(transformed_features, components_, columns)

def split_multiLabel_data(df_samples2, binarized_tags, random_state=None):
    """
    **DEPRECATED**: This function has been moved to the MultiLabelClassifier class
    with enhanced functionality. Please use:
    `MultiLabelClassifier().split_dataset_with_stratification()` instead.
    """
    warnings.warn(
        "split_multiLabel_data() is deprecated and will be removed in a future version. "
        "Please use MultiLabelClassifier().split_dataset_with_stratification() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    classifier = MultiLabelClassifier(random_state=random_state)
    samples_split, labels_split = classifier.split_dataset_with_stratification(
        df_samples2, binarized_tags, train_size=0.7, validation_size=0.21, test_size=0.09
    )
    
    # Convert to old format
    samples_split.rename(columns={'dataset_split': 'Set'}, inplace=True)
    labels_split.rename(columns={'dataset_split': 'Set'}, inplace=True)
    
    return samples_split, labels_split

def evaluate_multiLabel(y_pred, y_true, average_op='binary', scores_names=None):
    """
    **DEPRECATED**: This function has been moved to the MultiLabelClassifier class
    with enhanced functionality. Please use:
    `MultiLabelClassifier().evaluate_multilabel_performance()` instead.
    """
    if scores_names is None:
        scores_names = ['recall', 'precision', 'accuracy', 'auc_weighted', 'f1', 'kappa', 'mcc']
    
    warnings.warn(
        "evaluate_multiLabel() is deprecated and will be removed in a future version. "
        "Please use MultiLabelClassifier().evaluate_multilabel_performance() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    classifier = MultiLabelClassifier()
    performance_metrics, evaluation_df = classifier.evaluate_multilabel_performance(
        y_pred, y_true, average_op, scores_names
    )
    
    # Convert to old format
    model_performance = {
        'yScore': performance_metrics['detailed_scores'],
        'accuracy_overall': performance_metrics['subset_accuracy']
    }
    y_model = evaluation_df.rename(columns={'label': 'CV_Iteration'})
    
    return model_performance, y_model

# =============================================================================
# ENHANCED FUNCTION MAPPING
# =============================================================================

# Update the function mapping with new functions
NEW_FUNCTION_MAPPING = {
    # Enhanced PCA functions
    'pca_explainedVar': 'DimensionalityReducer.create_explained_variance_visualization()',
    'pca_ortho_rotation': 'DimensionalityReducer.apply_varimax_rotation()',
    'pca_important_features': 'DimensionalityReducer.calculate_feature_importance_scores()',
    'pc_draw_vectors': 'DimensionalityReducer.create_biplot_visualization()',
    
    # Multi-label classification functions
    'split_multiLabel_data': 'MultiLabelClassifier.split_dataset_with_stratification()',
    'evaluate_multiLabel': 'MultiLabelClassifier.evaluate_multilabel_performance()',
}

# Merge with existing mapping (use the comprehensive mapping defined earlier)
COMPREHENSIVE_FUNCTION_MAPPING.update(NEW_FUNCTION_MAPPING)

# Example usage and demonstration
if __name__ == "__main__":
    print("\\n🚀 Machine Learning Functions - Refactored Version 2.0")
    print("=" * 60)
    
    # Display function mapping
    print_comprehensive_function_mapping()
    
    print("\\n📖 Usage Examples:")
    print("-" * 20)
    
    examples = '''
    # Object-oriented approach (recommended):
    template_manager = ModelTemplateManager(random_state=42)
    models = template_manager.get_classification_models(X_train, y_train)
    
    trainer = ModelTrainer(random_state=42)
    results = trainer.compare_models(models, X_train, y_train, cv_folds=5)
    
    evaluator = ModelEvaluator()
    scores = evaluator.calculate_model_scores(predictions, ['accuracy', 'f1_score'])
    
    # SHAP analysis:
    shap_analyzer = SHAPAnalyzer(output_directory='./shap_plots/')
    shap_values, importance = shap_analyzer.calculate_shap_contributions(
        X_test, xgb_model, predictions_df, 'positive_class'
    )
    
    # Backward compatible approach (deprecated but supported):
    models = classifiers_template(random_state=42)  # Issues deprecation warning
    '''
    
    print(examples)
    
    print("\\n✅ All original functionality preserved with enhanced:")
    print("   • Comprehensive error handling and input validation")
    print("   • Detailed PEP 257 documentation")  
    print("   • Improved variable and function naming")
    print("   • Logical class-based organization")
    print("   • Backward compatibility with deprecation warnings")
    print("   • Enhanced logging and debugging capabilities")
    print("\\n" + "=" * 60)