#Just a test of the .pkl model

import pickle as pickle
import pandas as pd
from build_model import TextClassifier, get_data

if __name__ == '__main__':
	with open('static/model.pkl', 'rb') as f:
	    model = pickle.load(f)

	X, y = get_data('data/articles.csv')
	model.transform(X)

	print("Accuracy:", model.score(X, y))
	print("Predictions:", model.predict(X))