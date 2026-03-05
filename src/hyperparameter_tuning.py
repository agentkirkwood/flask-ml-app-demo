"""
Module for hyperparameter tuning and model comparison.

This script performs comprehensive hyperparameter tuning for:
1. MultinomialNB with RandomizedSearchCV on alpha parameter
2. LogisticRegression with RandomizedSearchCV on l1_ratio (L1 vs L2), C, and tol

Uses 10-fold cross-validation and saves detailed results including individual fold scores.

USE:
    # Basic usage with defaults (Note: no .py extension when using -m)
    python -m src.hyperparameter_tuning
    
    # With custom options
    python -m src.hyperparameter_tuning --out src/model_scores --n-folds 10 --n-iter 50 --random-state 42
    
OPTIONS:
    --out           Output directory for results (default: src/model_scores)
    --n-folds       Number of cross-validation folds (default: 10)
    --n-iter        Number of iterations for RandomizedSearchCV (default: 50)
    --random-state  Random state for reproducibility (default: 42)

"""
import argparse
import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, make_scorer, precision_score, recall_score, f1_score
from scipy.stats import uniform, loguniform  # type: ignore
from src.build_model import get_data, OversamplingEstimator


# Create custom scorers with zero_division=0 to suppress warnings
SCORING = {
    'accuracy': 'accuracy',
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
}


def create_multinomial_nb_pipeline() -> Tuple[Pipeline, Dict[str, Any]]:
    """Create a pipeline for MultinomialNB with TfidfVectorizer.
    
    Note: Oversampling must be handled outside the pipeline since it needs access to labels.
    
    Returns
    -------
    pipeline: sklearn Pipeline object
    param_distributions: dict of parameter distributions for RandomizedSearchCV
    """
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english',
                                       token_pattern=r'[a-z]+',
                                       lowercase=True)),
        ('classifier', MultinomialNB())
    ])
    
    # Parameter distributions for RandomizedSearchCV
    # Note: Parameters are prefixed with 'pipeline__' because they're accessed through OversamplingEstimator
    param_distributions = {
        'pipeline__classifier__alpha': loguniform(1e-3, 10)  # Search alpha on log scale from 0.001 to 10
    }
    
    return pipeline, param_distributions


def create_logistic_regression_pipeline() -> Tuple[Pipeline, Dict[str, Any]]:
    """Create a pipeline for LogisticRegression with TfidfVectorizer.
    
    Note: Oversampling must be handled outside the pipeline since it needs access to labels.
    
    Returns
    -------
    pipeline: sklearn Pipeline object
    param_distributions: dict of parameter distributions for RandomizedSearchCV (l1_ratio, C, tol)
    """
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english',
                                       token_pattern=r'[a-z]+',
                                       lowercase=True)),
        ('classifier', LogisticRegression(max_iter=1000, solver='saga'))
    ])
    
    # Random search for all hyperparameters together
    # Note: In sklearn 1.8+, use l1_ratio instead of penalty parameter
    # l1_ratio=0.0 is pure L2 (ridge) - shrinks all features
    # l1_ratio=0.5 is Elastic Net (50/50) - balanced mix of both penalties
    # l1_ratio=1.0 is pure L1 (lasso) - feature selection via sparsity
    # Parameters are prefixed with 'pipeline__' because they're accessed through OversamplingEstimator
    param_distributions = {
        'pipeline__classifier__l1_ratio': [0.0, 1.0],  # Test L2 and L1, can also add intermediate values for Elastic Net if desired
        'pipeline__classifier__C': loguniform(0.01, 200),  # Inverse regularization, log scale 0.01 to 100
        'pipeline__classifier__tol': loguniform(1e-5, 1e-2)  # Stopping tolerance, log scale 1e-5 to 0.01
    }
    
    return pipeline, param_distributions


def perform_multinomial_nb_search(X: List[str], y: List[str], n_folds: int = 10, n_iter: int = 50, random_state: int = 42) -> Dict[str, Any]:
    """Perform hyperparameter tuning for MultinomialNB.
    
    Uses custom CV with oversampling applied only to training folds to avoid data leakage.
    
    Parameters
    ----------
    X: list of text fragments (original, unbalanced data)
    y: list of labels (original, unbalanced data)
    n_folds: int, number of cross-validation folds
    n_iter: int, number of iterations for RandomizedSearchCV
    random_state: int, random seed
    
    Returns
    -------
    dict containing best model, hyperparameters, and CV results
    """
    print("\n" + "="*80)
    print("MULTINOMIAL NAIVE BAYES HYPERPARAMETER TUNING")
    print("="*80)
    
    start_time: float = time.time()
    
    pipeline: Pipeline
    param_distributions: Dict[str, Any]
    pipeline, param_distributions = create_multinomial_nb_pipeline()
    
    # Create custom CV that oversamples only training folds
    from sklearn.model_selection import StratifiedKFold
    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Manually perform RandomizedSearchCV with oversampling in each training fold
    print(f"\nRunning RandomizedSearchCV with {n_iter} iterations and {n_folds}-fold CV...")
    print("Note: Oversampling applied to training folds only (test folds remain original distribution)")
    
    # Wrap pipeline with oversampling estimator
    wrapped_pipeline = OversamplingEstimator(pipeline)
    
    # Perform RandomizedSearchCV
    random_search: RandomizedSearchCV = RandomizedSearchCV(
        wrapped_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring='accuracy',
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True,
        verbose=3
    )
    
    random_search.fit(X, y)
    
    # Get detailed results
    best_wrapped: OversamplingEstimator = random_search.best_estimator_  # type: ignore
    best_model: Pipeline = best_wrapped.pipeline  # Extract the actual pipeline
    best_params: Dict[str, Any] = random_search.best_params_
    best_score: float = random_search.best_score_
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    
    # Get individual fold scores for the best model using the wrapper
    cv_results_detailed: Dict[str, Any] = cross_validate(  # type: ignore
        best_wrapped,
        X, y,
        cv=cv_splitter,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calculate per-class metrics across folds
    print(f"\nCalculating per-class metrics across {n_folds} folds...")
    per_class_metrics: Dict[str, Dict[str, Any]] = calculate_per_class_metrics(best_model, X, y, n_folds)
    
    end_time: float = time.time()
    total_time: float = end_time - start_time
    
    # Package results
    results: Dict[str, Any] = {
        'model_type': 'MultinomialNB',
        'best_model': best_model,
        'best_hyperparameters': best_params,
        'cv_folds': n_folds,
        'tuning_time_seconds': float(total_time),
        'individual_fold_scores': {
            'test_accuracy': cv_results_detailed['test_accuracy'].tolist(),
            'test_precision_macro': cv_results_detailed['test_precision_macro'].tolist(),
            'test_recall_macro': cv_results_detailed['test_recall_macro'].tolist(),
            'test_f1_macro': cv_results_detailed['test_f1_macro'].tolist(),
            'train_accuracy': cv_results_detailed['train_accuracy'].tolist(),
            'train_precision_macro': cv_results_detailed['train_precision_macro'].tolist(),
            'train_recall_macro': cv_results_detailed['train_recall_macro'].tolist(),
            'train_f1_macro': cv_results_detailed['train_f1_macro'].tolist(),
        },
        'average_cv_performance': {
            'test_accuracy_mean': float(np.mean(cv_results_detailed['test_accuracy'])),
            'test_accuracy_std': float(np.std(cv_results_detailed['test_accuracy'])),
            'test_precision_macro_mean': float(np.mean(cv_results_detailed['test_precision_macro'])),
            'test_precision_macro_std': float(np.std(cv_results_detailed['test_precision_macro'])),
            'test_recall_macro_mean': float(np.mean(cv_results_detailed['test_recall_macro'])),
            'test_recall_macro_std': float(np.std(cv_results_detailed['test_recall_macro'])),
            'test_f1_macro_mean': float(np.mean(cv_results_detailed['test_f1_macro'])),
            'test_f1_macro_std': float(np.std(cv_results_detailed['test_f1_macro'])),
            'train_accuracy_mean': float(np.mean(cv_results_detailed['train_accuracy'])),
            'train_accuracy_std': float(np.std(cv_results_detailed['train_accuracy'])),
        },
        'per_class_metrics': per_class_metrics
    }
    
    print("\n" + "-"*80)
    print("INDIVIDUAL FOLD SCORES (Test Set):")
    print("-"*80)
    for i, (acc, prec, rec, f1) in enumerate(zip(
        cv_results_detailed['test_accuracy'],
        cv_results_detailed['test_precision_macro'],
        cv_results_detailed['test_recall_macro'],
        cv_results_detailed['test_f1_macro']
    ), 1):
        print(f"Fold {i:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    print("\n" + "-"*80)
    print("AVERAGE PERFORMANCE:")
    print("-"*80)
    print(f"Test Accuracy:  {results['average_cv_performance']['test_accuracy_mean']:.4f} "
          f"(± {results['average_cv_performance']['test_accuracy_std']:.4f})")
    print(f"Test Precision: {results['average_cv_performance']['test_precision_macro_mean']:.4f} "
          f"(± {results['average_cv_performance']['test_precision_macro_std']:.4f})")
    print(f"Test Recall:    {results['average_cv_performance']['test_recall_macro_mean']:.4f} "
          f"(± {results['average_cv_performance']['test_recall_macro_std']:.4f})")
    print(f"Test F1:        {results['average_cv_performance']['test_f1_macro_mean']:.4f} "
          f"(± {results['average_cv_performance']['test_f1_macro_std']:.4f})")
    
    print("\n" + "-"*80)
    print("PER-CLASS METRICS (averaged across folds):")
    print("-"*80)
    for class_name in sorted(per_class_metrics.keys()):
        metrics = per_class_metrics[class_name]
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision_mean']:.4f} (± {metrics['precision_std']:.4f})")
        print(f"  Recall:    {metrics['recall_mean']:.4f} (± {metrics['recall_std']:.4f})")
        print(f"  F1-Score:  {metrics['f1_mean']:.4f} (± {metrics['f1_std']:.4f})")
    
    print("\n" + "-"*80)
    print(f"Total tuning time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("-"*80)
    
    return results


def calculate_per_class_metrics(model: Pipeline, X: List[str], y: List[str], n_folds: int) -> Dict[str, Dict[str, Any]]:
    """Calculate per-class precision, recall, and F1 scores across CV folds.
    
    Parameters
    ----------
    model: fitted sklearn pipeline or estimator
    X: list of text fragments
    y: list of labels
    n_folds: int, number of cross-validation folds
    
    Returns
    -------
    dict: per-class metrics with mean and std across folds
    """
    # Get unique classes
    classes: List[str] = sorted(set(y))
    
    # Initialize storage for per-fold, per-class metrics
    fold_metrics: Dict[str, Dict[str, List[float]]] = {cls: {'precision': [], 'recall': [], 'f1': []} for cls in classes}
    
    # Perform stratified K-fold cross-validation
    skf: StratifiedKFold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, test_idx) in tqdm(
        enumerate(skf.split(X, y), 1),
        total=n_folds,
        desc="Evaluating folds",
        unit="fold"
    ):
        # Split data
        X_train: List[str] = [X[i] for i in train_idx]
        y_train: List[str] = [y[i] for i in train_idx]
        X_test: List[str] = [X[i] for i in test_idx]
        y_test: List[str] = [y[i] for i in test_idx]
        
        # Oversample training fold only (test fold remains original distribution)
        X_train, y_train = OversamplingEstimator.oversample_minority_classes(X_train, y_train)
        
        # Clone and fit model on oversampled training fold
        from sklearn.base import clone
        fold_model: Pipeline = clone(model)  # type: ignore
        fold_model.fit(X_train, y_train)
        
        # Predict on test fold
        y_pred = fold_model.predict(X_test)
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=classes, average=None, zero_division=0
        )
        
        # Store metrics for each class
        for i, cls in enumerate(classes):
            fold_metrics[cls]['precision'].append(precision[i])
            fold_metrics[cls]['recall'].append(recall[i])
            fold_metrics[cls]['f1'].append(f1[i])
    
    # Calculate mean and std for each class
    per_class_results: Dict[str, Dict[str, Any]] = {}
    for cls in classes:
        per_class_results[cls] = {
            'precision_mean': float(np.mean(fold_metrics[cls]['precision'])),
            'precision_std': float(np.std(fold_metrics[cls]['precision'])),
            'recall_mean': float(np.mean(fold_metrics[cls]['recall'])),
            'recall_std': float(np.std(fold_metrics[cls]['recall'])),
            'f1_mean': float(np.mean(fold_metrics[cls]['f1'])),
            'f1_std': float(np.std(fold_metrics[cls]['f1'])),
            'precision_per_fold': [float(x) for x in fold_metrics[cls]['precision']],
            'recall_per_fold': [float(x) for x in fold_metrics[cls]['recall']],
            'f1_per_fold': [float(x) for x in fold_metrics[cls]['f1']]
        }
    
    return per_class_results


def perform_logistic_regression_search(X: List[str], y: List[str], n_folds: int = 10, n_iter_grid: int = 2, n_iter_random: int = 50, random_state: int = 42) -> Dict[str, Any]:
    """Perform hyperparameter tuning for LogisticRegression.
    
    Performs RandomizedSearchCV on l1_ratio (L1 vs L2), C, and tol simultaneously.
    Uses custom CV with oversampling applied only to training folds to avoid data leakage.
    
    Parameters
    ----------
    X: list of text fragments (original, unbalanced data)
    y: list of labels (original, unbalanced data)
    n_folds: int, number of cross-validation folds
    n_iter_grid: int, (deprecated, kept for backwards compatibility)
    n_iter_random: int, number of iterations for RandomizedSearchCV
    random_state: int, random seed
    
    Returns
    -------
    dict containing best model, hyperparameters, and CV results
    """
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION HYPERPARAMETER TUNING")
    print("="*80)
    
    start_time: float = time.time()
    
    pipeline: Pipeline
    param_distributions: Dict[str, Any]
    pipeline, param_distributions = create_logistic_regression_pipeline()
    
    # Create custom CV that oversamples only training folds
    from sklearn.model_selection import StratifiedKFold
    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Perform RandomizedSearchCV on all hyperparameters together
    print(f"\nRunning RandomizedSearchCV with {n_iter_random} iterations and {n_folds}-fold CV...")
    print("Searching over: l1_ratio (L1 vs L2), C (regularization), tol (tolerance)")
    print("Note: Oversampling applied to training folds only (test folds remain original distribution)")
    
    # Wrap pipeline with oversampling estimator
    wrapped_pipeline = OversamplingEstimator(pipeline)
    
    random_search: RandomizedSearchCV = RandomizedSearchCV(
        wrapped_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter_random,
        cv=cv_splitter,
        scoring='accuracy',
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True,
        verbose=3
    )
    
    random_search.fit(X, y)
    
    # Get detailed results
    best_wrapped: OversamplingEstimator = random_search.best_estimator_  # type: ignore
    best_model: Pipeline = best_wrapped.pipeline  # Extract the actual pipeline
    best_params: Dict[str, Any] = random_search.best_params_
    best_score: float = random_search.best_score_
    
    # Display best parameters with readable penalty type
    best_l1_ratio: float = best_params.get('pipeline__classifier__l1_ratio', 0.0)
    if best_l1_ratio == 0.0:
        penalty_type: str = "L2 (Ridge)"
    elif best_l1_ratio == 1.0:
        penalty_type = "L1 (Lasso)"
    else:
        penalty_type = f"Elastic Net (L1 ratio={best_l1_ratio})"
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best penalty type: {penalty_type}")
    print(f"Best CV score: {best_score:.4f}")
    
    # Get individual fold scores for the best model using the wrapper
    cv_results_detailed: Dict[str, Any] = cross_validate(  # type: ignore
        best_wrapped,
        X, y,
        cv=cv_splitter,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=-1
    )
     
    # Calculate per-class metrics across folds
    print(f"\nCalculating per-class metrics across {n_folds} folds...")
    per_class_metrics: Dict[str, Dict[str, Any]] = calculate_per_class_metrics(best_model, X, y, n_folds)
    
    end_time: float = time.time()
    total_time: float = end_time - start_time
    
    # Package results
    results: Dict[str, Any] = {
        'model_type': 'LogisticRegression',
        'best_model': best_model,
        'best_hyperparameters': best_params,
        'cv_folds': n_folds,
        'tuning_time_seconds': float(total_time),
        'individual_fold_scores': {
            'test_accuracy': cv_results_detailed['test_accuracy'].tolist(),
            'test_precision_macro': cv_results_detailed['test_precision_macro'].tolist(),
            'test_recall_macro': cv_results_detailed['test_recall_macro'].tolist(),
            'test_f1_macro': cv_results_detailed['test_f1_macro'].tolist(),
            'train_accuracy': cv_results_detailed['train_accuracy'].tolist(),
            'train_precision_macro': cv_results_detailed['train_precision_macro'].tolist(),
            'train_recall_macro': cv_results_detailed['train_recall_macro'].tolist(),
            'train_f1_macro': cv_results_detailed['train_f1_macro'].tolist(),
        },
        'average_cv_performance': {
            'test_accuracy_mean': float(np.mean(cv_results_detailed['test_accuracy'])),
            'test_accuracy_std': float(np.std(cv_results_detailed['test_accuracy'])),
            'test_precision_macro_mean': float(np.mean(cv_results_detailed['test_precision_macro'])),
            'test_precision_macro_std': float(np.std(cv_results_detailed['test_precision_macro'])),
            'test_recall_macro_mean': float(np.mean(cv_results_detailed['test_recall_macro'])),
            'test_recall_macro_std': float(np.std(cv_results_detailed['test_recall_macro'])),
            'test_f1_macro_mean': float(np.mean(cv_results_detailed['test_f1_macro'])),
            'test_f1_macro_std': float(np.std(cv_results_detailed['test_f1_macro'])),
            'train_accuracy_mean': float(np.mean(cv_results_detailed['train_accuracy'])),
            'train_accuracy_std': float(np.std(cv_results_detailed['train_accuracy'])),
        },
        'per_class_metrics': per_class_metrics
    }
    
    print("\n" + "-"*80)
    print("INDIVIDUAL FOLD SCORES (Test Set):")
    print("-"*80)
    for i, (acc, prec, rec, f1) in enumerate(zip(
        cv_results_detailed['test_accuracy'],
        cv_results_detailed['test_precision_macro'],
        cv_results_detailed['test_recall_macro'],
        cv_results_detailed['test_f1_macro']
    ), 1):
        print(f"Fold {i:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    
    print("\n" + "-"*80)
    print("AVERAGE PERFORMANCE:")
    print("-"*80)
    print(f"Test Accuracy:  {results['average_cv_performance']['test_accuracy_mean']:.4f} "
          f"(± {results['average_cv_performance']['test_accuracy_std']:.4f})")
    print(f"Test Precision: {results['average_cv_performance']['test_precision_macro_mean']:.4f} "
          f"(± {results['average_cv_performance']['test_precision_macro_std']:.4f})")
    print(f"Test Recall:    {results['average_cv_performance']['test_recall_macro_mean']:.4f} "
          f"(± {results['average_cv_performance']['test_recall_macro_std']:.4f})")
    print(f"Test F1:        {results['average_cv_performance']['test_f1_macro_mean']:.4f} "
          f"(± {results['average_cv_performance']['test_f1_macro_std']:.4f})")
    
    print("\n" + "-"*80)
    print("PER-CLASS METRICS (averaged across folds):")
    print("-"*80)
    for class_name in sorted(per_class_metrics.keys()):
        metrics = per_class_metrics[class_name]
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision_mean']:.4f} (± {metrics['precision_std']:.4f})")
        print(f"  Recall:    {metrics['recall_mean']:.4f} (± {metrics['recall_std']:.4f})")
        print(f"  F1-Score:  {metrics['f1_mean']:.4f} (± {metrics['f1_std']:.4f})")
    
    print("\n" + "-"*80)
    print(f"Total tuning time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("-"*80)
    
    return results


def save_results(nb_results: Dict[str, Any], lr_results: Dict[str, Any], output_dir: str) -> None:
    """Save all results to disk.
    
    Parameters
    ----------
    nb_results: dict, results from MultinomialNB tuning
    lr_results: dict, results from LogisticRegression tuning
    output_dir: str, directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for JSON metadata
    last_updated: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save models
    nb_model_path: str = os.path.join(output_dir, 'multinomial_nb_best_model.pkl')
    lr_model_path: str = os.path.join(output_dir, 'logistic_regression_best_model.pkl')
    
    with open(nb_model_path, 'wb') as f:
        pickle.dump(nb_results['best_model'], f)
    print(f"\nMultinomialNB best model saved to: {nb_model_path}")
    
    with open(lr_model_path, 'wb') as f:
        pickle.dump(lr_results['best_model'], f)
    print(f"LogisticRegression best model saved to: {lr_model_path}")
    
    # Save results (without the model objects) as JSON
    nb_results_json: Dict[str, Any] = {k: v for k, v in nb_results.items() if k != 'best_model'}
    nb_results_json['last_updated'] = last_updated
    
    lr_results_json: Dict[str, Any] = {k: v for k, v in lr_results.items() if k != 'best_model'}
    lr_results_json['last_updated'] = last_updated
    
    nb_json_path: str = os.path.join(output_dir, 'multinomial_nb_results.json')
    lr_json_path: str = os.path.join(output_dir, 'logistic_regression_results.json')
    
    with open(nb_json_path, 'w') as f:
        json.dump(nb_results_json, f, indent=2)
    print(f"MultinomialNB results saved to: {nb_json_path}")
    
    with open(lr_json_path, 'w') as f:
        json.dump(lr_results_json, f, indent=2)
    print(f"LogisticRegression results saved to: {lr_json_path}")
    
    # Save comparison summary
    comparison: Dict[str, Any] = {
        'last_updated': last_updated,
        'models_compared': ['MultinomialNB', 'LogisticRegression'],
        'cv_folds': nb_results['cv_folds'],
        'MultinomialNB': {
            'best_hyperparameters': nb_results['best_hyperparameters'],
            'average_cv_performance': nb_results['average_cv_performance'],
            'tuning_time_seconds': nb_results['tuning_time_seconds']
        },
        'LogisticRegression': {
            'best_hyperparameters': lr_results['best_hyperparameters'],
            'average_cv_performance': lr_results['average_cv_performance'],
            'tuning_time_seconds': lr_results['tuning_time_seconds']
        },
        'winner': 'MultinomialNB' if nb_results['average_cv_performance']['test_accuracy_mean'] > 
                                     lr_results['average_cv_performance']['test_accuracy_mean']
                                     else 'LogisticRegression'
    }
    
    comparison_path: str = os.path.join(output_dir, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Model comparison summary saved to: {comparison_path}")
    
    # Print final comparison
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON")
    print("="*80)
    print(f"\nMultinomialNB Test Accuracy: "
          f"{nb_results['average_cv_performance']['test_accuracy_mean']:.4f} "
          f"(± {nb_results['average_cv_performance']['test_accuracy_std']:.4f})")
    print(f"MultinomialNB Tuning Time: {nb_results['tuning_time_seconds']:.2f}s ({nb_results['tuning_time_seconds']/60:.2f}m)")
    
    print(f"\nLogisticRegression Test Accuracy: "
          f"{lr_results['average_cv_performance']['test_accuracy_mean']:.4f} "
          f"(± {lr_results['average_cv_performance']['test_accuracy_std']:.4f})")
    print(f"LogisticRegression Tuning Time: {lr_results['tuning_time_seconds']:.2f}s ({lr_results['tuning_time_seconds']/60:.2f}m)")
    
    print(f"\nWinner: {comparison['winner']}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform hyperparameter tuning for MultinomialNB and LogisticRegression models.')
    parser.add_argument('--out', default='src/model_scores',
                        help='Output directory to save results (default: src/model_scores)')
    parser.add_argument('--n-folds', type=int, default=10,
                        help='Number of cross-validation folds (default: 10)')
    parser.add_argument('--n-iter', type=int, default=50,
                        help='Number of iterations for RandomizedSearchCV (default: 50)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Load data
    print("Loading data from database...")
    X: List[str]
    y: List[str]
    X, y = get_data()
    print(f"Data loaded successfully. Total articles: {len(X)}")
    print(f"Unique classes: {len(set(y))}")
    print(f"Class distribution (original, unbalanced): {dict(pd.Series(y).value_counts())}")
    print("\n[Oversampling will be applied to training folds only during CV]")
    print("="*80)
    
    # Perform hyperparameter tuning for both models
    nb_results: Dict[str, Any] = perform_multinomial_nb_search(
        X, y,
        n_folds=args.n_folds,
        n_iter=args.n_iter,
        random_state=args.random_state
    )
    
    lr_results: Dict[str, Any] = perform_logistic_regression_search(
        X, y,
        n_folds=args.n_folds,
        n_iter_random=args.n_iter,
        random_state=args.random_state
    )
    
    # Save all results
    save_results(nb_results, lr_results, args.out)
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*80)
