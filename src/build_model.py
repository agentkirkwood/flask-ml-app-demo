"""
Module containing model fitting code for a web application that implements a
text classification model.

When run as a module, this will load a csv dataset, train a classification
model, and then pickle the resulting model object to disk.
"""
import pickle as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class TextClassifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit a naive bayes model to the resulting features.
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
        # Code to fit the model.
        X_transform = self._vectorizer.fit_transform(X)
        self._classifier.fit(X_transform, y)
        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        X_transform = self._vectorizer.transform(X)
        return self._classifier.predict_proba(X_transform)

    def predict(self, X):
        """Make predictions on new data."""
        X_transform = self._vectorizer.transform(X)
        return self._classifier.predict(X_transform)

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        X_transform = self._vectorizer.transform(X)
        return self._classifier.score(X_transform, y)


def get_data(filename, natural_language_col='body', label_col='section_name'):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.

    Returns
    -------
    X: A numpy array containing the text fragments used for training.
    y: A numpy array containing labels, used for model response.
    """
    df = pd.read_csv(filename)
    y = df.pop(label_col)
    return df[natural_language_col].values, y.values


if __name__ == '__main__':
    #Data
    X, y = get_data("data/articles.csv")

    tc = TextClassifier()
    tc.fit(X, y)

    #Save model - Text classifier
    with open('static/model.pkl', 'wb') as f:
        pickle.dump(tc, f)

    #load model
    with open('static/model.pkl', 'rb') as f:
        model = pickle.load(f)  

    #model stats
    print("Accuracy:", model.score(X, y))
    print("Predictions:", model.predict(X))
    
    
