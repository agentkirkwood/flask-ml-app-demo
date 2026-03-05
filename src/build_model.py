"""
Module containing model fitting code for a web application that 
implements a text classification model.

Includes OversamplingEstimator class for balancing minority classes
during training while preserving test set distributions.

When run as a module, this will load a csv dataset, train a 
classification model, and then pickle the resulting model object to disk.

USE:

python build_model.py --data path_to_input_data --out path_to_save_pickled_model

"""
import argparse
import pickle
import pandas as pd
import os
import random
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.create_db import Article, Base


class OversamplingEstimator(BaseEstimator):
    """Estimator wrapper that oversamples training data before fitting the pipeline.
    
    This ensures oversampling only affects training data in CV folds, not test data.
    During fit(), minority classes are oversampled to match the majority class count.
    During predict()/score(), data is passed through unchanged.
    
    This class is designed to wrap sklearn Pipelines for hyperparameter tuning.
    """
    
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
    
    @staticmethod
    def oversample_minority_classes(X: List[str], y: List[str]) -> Tuple[List[str], List[str]]:
        """Oversample minority classes to balance the training data.
        
        This static method duplicates samples from minority classes to match the count
        of the majority class, then shuffles the result.
        
        Parameters
        ----------
        X: list of text fragments
        y: list of labels
        
        Returns
        -------
        X_balanced: oversampled text fragments
        y_balanced: oversampled labels
        """
        # Find the maximum class frequency
        max_count = max([sum(1 for label in y if label == cls) for cls in set(y)])
        
        # Oversample minority classes
        X_balanced = []
        y_balanced = []
        for cls in set(y):
            X_cls = [X[i] for i in range(len(X)) if y[i] == cls]
            y_cls = [cls] * len(X_cls)
            
            # Duplicate samples to reach max_count
            repetitions = max_count // len(X_cls)
            remainder = max_count % len(X_cls)
            
            X_balanced.extend(X_cls * repetitions + X_cls[:remainder])
            y_balanced.extend(y_cls * repetitions + y_cls[:remainder])
        
        # Shuffle to mix classes
        combined = list(zip(X_balanced, y_balanced))
        random.shuffle(combined)
        X_balanced, y_balanced = zip(*combined)
        
        return list(X_balanced), list(y_balanced)
    
    def fit(self, X: List[str], y: List[str]) -> 'OversamplingEstimator':
        """Oversample training data and fit the pipeline."""
        X_train_balanced, y_train_balanced = self.oversample_minority_classes(X, y)
        self.pipeline.fit(X_train_balanced, y_train_balanced)
        return self
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Make predictions (no oversampling during prediction)."""
        return self.pipeline.predict(X)
    
    def score(self, X: List[str], y: List[str]) -> float:
        """Score the model (no oversampling during scoring)."""
        return self.pipeline.score(X, y)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.
        
        Returns the pipeline and optionally its nested parameters with 'pipeline__' prefix.
        """
        params = {'pipeline': self.pipeline}
        if deep:
            # Get nested parameters from the pipeline with 'pipeline__' prefix
            pipeline_params = self.pipeline.get_params(deep=True)
            for key, value in pipeline_params.items():
                params[f'pipeline__{key}'] = value
        return params
    
    def set_params(self, **params: Any) -> 'OversamplingEstimator':
        """Set parameters for this estimator.
        
        Handles both 'pipeline' parameter and nested 'pipeline__*' parameters.
        """
        pipeline_params = {}
        for key, value in params.items():
            if key == 'pipeline':
                self.pipeline = value
            elif key.startswith('pipeline__'):
                # Remove 'pipeline__' prefix and pass to pipeline
                pipeline_params[key[10:]] = value
            else:
                raise ValueError(f"Invalid parameter {key} for estimator OversamplingEstimator")
        
        if pipeline_params:
            self.pipeline.set_params(**pipeline_params)
        
        return self


class TextClassifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit the best classifier model (determined from hyperparameter tuning).

    This class loads the winning model type and hyperparameters from
    model_comparison.json (if available) and uses those for training.
    Falls back to default LogisticRegression if the file doesn't exist.

    This class implements the standard sklearn fit, predict, score interface.
    """

    def __init__(self, model_comparison_path: Optional[str] = None):
        """Initialize classifier with winning hyperparameters.
        
        Parameters
        ----------
        model_comparison_path: Path to model_comparison.json. If None, uses default location.
        """
        self._vectorizer = TfidfVectorizer(stop_words='english'
                                           , token_pattern=r'[a-z]+' #letters only, no numbers or punctuation 
                                           , lowercase=True)
        
        # Try to load winning model and hyperparameters
        if model_comparison_path is None:
            model_comparison_path = os.path.join(
                os.path.dirname(__file__), 'model_scores', 'model_comparison.json'
            )
        
        winning_model_type, best_params = self._load_winning_model(model_comparison_path)
        
        # Initialize classifier with winning hyperparameters
        if winning_model_type == 'LogisticRegression':
            self._classifier = LogisticRegression(
                C=best_params.get('C', 1.0),
                l1_ratio=best_params.get('l1_ratio', 0.0),
                tol=best_params.get('tol', 1e-4),
                max_iter=5000,
                solver='saga'
            )
            print(f"Using LogisticRegression with C={best_params.get('C', 1.0):.2f}, "
                  f"l1_ratio={best_params.get('l1_ratio', 0.0)}, "
                  f"tol={best_params.get('tol', 1e-4):.2e}")
        elif winning_model_type == 'MultinomialNB':
            self._classifier = MultinomialNB(
                alpha=best_params.get('alpha', 1.0)
            )
            print(f"Using MultinomialNB with alpha={best_params.get('alpha', 1.0):.4f}")
        else:
            # Fallback to default LogisticRegression with reasonable hyperparameters
            self._classifier = LogisticRegression(
                C=170,
                l1_ratio=0.0,
                tol=1e-4,
                max_iter=5000,
                solver='saga'
            )
            print("Using default LogisticRegression (no model_comparison.json found)")
    
    def _load_winning_model(self, path: str) -> Tuple[str, Dict[str, Any]]:
        """Load winning model type and hyperparameters from model_comparison.json.
        
        Parameters
        ----------
        path: Path to model_comparison.json
        
        Returns
        -------
        model_type: 'MultinomialNB' or 'LogisticRegression'
        best_params: Dict of hyperparameters (with pipeline__ prefix stripped)
        """
        try:
            with open(path, 'r') as f:
                comparison_data = json.load(f)
            
            winner: str = comparison_data.get('winner', 'LogisticRegression')
            raw_params: Dict[str, Any] = comparison_data.get(winner, {}).get('best_hyperparameters', {})
            
            # Strip 'pipeline__classifier__' prefix from parameter names
            best_params: Dict[str, Any] = {}
            for key, value in raw_params.items():
                # Remove the nested prefix to get just the param name
                clean_key = key.replace('pipeline__classifier__', '')
                best_params[clean_key] = value
            
            return winner, best_params
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Could not load model_comparison.json: {e}")
            return 'LogisticRegression', {}

    def fit(self, X, y):
        """Fit a text classifier model.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        # Oversample minority classes for balanced training
        X, y = OversamplingEstimator.oversample_minority_classes(X, y)
        
        X = self._vectorizer.fit_transform(X)
        self._classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        """Make probability predictions on new data.
        
        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.

        Returns
        -------
        probs: A (n_obs, n_classes) numpy array of predicted class probabilities. 
        """
        X = self._vectorizer.transform(X)
        return self._classifier.predict_proba(X)

    def predict(self, X):
        """Make class predictions on new data.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.

        Returns
        -------
        preds: A (n_obs,) numpy array containing the predicted class for each
        observation (i.e. the class with the maximal predicted class probabilitiy.
        """
        X = self._vectorizer.transform(X)
        return self._classifier.predict(X)

    def score(self, X, y):
        """Return a classification accuracy score on new data.

        Parameters
        ----------
        X: A numpy array or list of text fragments.
        y: A numpy array or python list of true class labels.
        """
        X = self._vectorizer.transform(X)
        return self._classifier.score(X, y)


def get_data(filename=None):
    """Load training data.

    If a CSV `filename` is provided (deprecated), load from CSV.
    Otherwise, load from the database indicated by `DATABASE_URL` or
    default `sqlite:///data/articles.db`.
    
    Filters out articles with blank subjects.

    Returns
    -------
    X: A list containing text fragments used for training.
    y: A list containing labels, used for model response.
    """
    if filename:
        # Deprecated path: load directly from CSV
        # Note: This path is deprecated and doesn't support subject field
        raise ValueError("Loading from CSV is no longer supported. Please use database.")
    
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/articles.db')
    # Fix Heroku postgres:// URL to postgresql+psycopg:// for psycopg3
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql+psycopg://', 1)
    elif DATABASE_URL and DATABASE_URL.startswith('postgresql://') and 'sqlite' not in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg://', 1)

    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Load articles with non-blank subjects only
        articles = session.query(Article).filter(Article.subject != '').filter(Article.subject.isnot(None)).all()
        bodies = [article.body for article in articles]
        subjects = [article.subject for article in articles]
    finally:
        session.close()
    
    return bodies, subjects


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit a Text Classifier model and save the results.')
    parser.add_argument('--data', help='(Deprecated) Path to CSV training data.')
    parser.add_argument('--out', help='A file to save the pickled model object to.')
    args = parser.parse_args()

    if args.data:
        print("[DEPRECATED] --data provided: loading training data from CSV.")
        X, y = get_data(args.data)
    else:
        print("Loading data from database...")
        X, y = get_data()
    print(f"Data loaded successfully. Total articles: {len(X)}")
    print(f"Unique sections: {len(set(y))}")
    
    print("Training text classifier model...")
    tc = TextClassifier()
    tc.fit(X, y)
    print("Model training complete.")
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(args.out)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    print(f"Saving model to {args.out}...")
    with open(args.out, 'wb') as f:  # Use 'wb' for binary write mode
        pickle.dump(tc, f)
    print("Model saved successfully!")
