from flask import Flask, render_template, request
import pickle
from src.build_model import TextClassifier

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/submit')
def submission_page():
	return render_template('submit.html')

@app.route('/predict', methods=['POST'])
def predict():
	with open('static/model.pkl', 'rb') as f:
		model = pickle.load(f)  
	text = str(request.form['article_body'])
	pred = model.predict([text])[0]
	return render_template('predict.html', article=text, predicted=pred)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)
