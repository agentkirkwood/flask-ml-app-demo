"""
Module containing model fitting code for a web application that 
implements a text classification model.

When run as a module, this will load a csv dataset, train a 
classification model, and then pickle the resulting model object to disk.

USE:

python build_model.py --data path_to_input_data --out path_to_save_pickled_model

"""
import argparse
import pickle
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.create_db import Article, Base


class TextClassifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit a naive bayes model to the resulting features.

    The work done by this class could also be done with a sklean.pipeline
    object.  Since the author cannot guarantee that Pipelines have been
    introduced, he opted to write his own class implementing the model.

    This class is an example of coding to an interface, it implements the
    standard sklearn fit, predict, score interface.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._classifier = MultinomialNB()

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


def transform_section_name(section):
    """Transform and filter section names for the training dataset.
    
    Returns None if section should be excluded from training.
    """
    if section in ['Universal', 'NYT Now', 'Magazine']:
        return None
    
    mapping = {
        'Your Money': 'Business, Economics & Finance',
        'Job Market': 'Business, Economics & Finance',
        'Business Day': 'Business, Economics & Finance',
        'Style': 'Fashion & Style',
        'Great Homes & Destinations': 'Travel & Real Estate',
        'Real Estate': 'Travel & Real Estate',
        'Travel': 'Travel & Real Estate',
        'Automobiles': 'Technology',
        'Public Editor': 'Opinion',
        'The Upshot': 'Politics',
    }
    
    return mapping.get(section, section)


def get_data(filename=None):
    """Load training data.

    If a CSV `filename` is provided (deprecated), load from CSV.
    Otherwise, load from the database indicated by `DATABASE_URL` or
    default `sqlite:///data/articles.db`.
    
    Applies section name transformations and filtering as preprocessing.

    Returns
    -------
    X: A list containing text fragments used for training.
    y: A list containing labels, used for model response.
    """
    if filename:
        # Deprecated path: load directly from CSV
        df = pd.read_csv(filename)
        bodies = list(df.body)
        sections = list(df.section_name)
    else:
        DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/articles.db')

        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            articles = session.query(Article).all()
            bodies = [article.body for article in articles]
            sections = [article.section_name for article in articles]
        finally:
            session.close()

    # Filter and transform section names
    filtered_bodies = []
    filtered_sections = []
    
    for body, section in zip(bodies, sections):
        transformed = transform_section_name(section)
        if transformed is not None:
            filtered_bodies.append(body)
            filtered_sections.append(transformed)
    
    return filtered_bodies, filtered_sections


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
