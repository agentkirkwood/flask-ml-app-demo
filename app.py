from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
from src.build_model import TextClassifier, get_data # type: ignore
import os

app = Flask(__name__)

# Add route to serve Bootstrap files from shared directory
@app.route('/bootstrap/<path:filename>')
def bootstrap_static(filename):
    bootstrap_dir = os.path.join(os.path.dirname(__file__), '..', 'bootstrap')
    return send_from_directory(bootstrap_dir, filename)


model = None

def get_model():
    global model
    if model is None:
        model_path = 'static/model.pkl'
        
        # Check if model exists, if not build it
        if not os.path.exists(model_path):
            print("Model not found, building it now...")
            
            # Create static directory if it doesn't exist
            os.makedirs('static', exist_ok=True)
            
            # Load data and train model
            X, y = get_data()
            print(f"Data loaded successfully. Total articles: {len(X)}")
            
            model = TextClassifier()
            model.fit(X, y)
            print("Model training complete.")
            
            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to {model_path}")
        else:
            # Load existing model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded from disk")
    
    return model

@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('index.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render an AJAX-enabled form to collect article text."""
    return render_template('submit.html')

#The traditional form submission route
@app.route('/submit_traditional', methods=['GET'])
def submit_traditional():
    """Render a page containing a textarea input where the user can paste an
    article to be classified (traditional form submission)."""
    return render_template('submit_traditional.html')

@app.route('/predict', methods=['POST'])
def predict():
    """AJAX endpoint: Receive article text as JSON and return prediction as JSON.
    
    Expected JSON input:
    {
        "article_body": "text of the article..."
    }
    
    Returns JSON:
    {
        "prediction": "predicted_category",
        "success": true
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        article_text = data.get('article_body', '')
        
        if not article_text:
            return jsonify({
                'success': False,
                'error': 'No article text provided'
            }), 400
        
        # Make prediction with probability
        model = get_model()
        prediction = str(model.predict([article_text])[0])
        probabilities = model.predict_proba([article_text])[0]
        max_probability = float(max(probabilities))
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': max_probability
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

#The traditional form submission route
@app.route('/predict_traditional', methods=['POST'])
def predict_traditional():
    """Traditional form submission: Receive article from form and return HTML page.
    """
    data = str(request.form['article_body'])
    model = get_model()
    pred = str(model.predict([data])[0])
    probabilities = model.predict_proba([data])[0]
    max_prob = float(max(probabilities))
    return render_template('predict.html', article=data, predicted=pred, probability=max_prob)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000)) #finds port set by Heroku or defaults to 5000
    app.run(host='0.0.0.0'
            , port=port
            , threaded=True) 