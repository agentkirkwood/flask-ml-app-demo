# ML_APP - Text Classification Web Application

A Flask web application that classifies newspaper article text snippets using a Naive Bayes machine learning model.

## Overview

This application:
- Trains a text classification model on newspaper articles
- Uses TF-IDF vectorization for feature extraction
- Implements a Multinomial Naive Bayes classifier
- Provides a web interface for classifying new article text

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- Flask
- pandas
- scikit-learn
- pickle (built-in)

### 2. Build the Model

The model must be trained before running the web application.

**Command:**
```bash
python build_model.py --data <path_to_training_data> --out <path_to_save_model>
```

**Parameters:**
- `--data`: Path to a CSV file containing training data
  - Must have columns: `body` (article text) and `section_name` (category label)
  - Example: `data/articles.csv`
  
- `--out`: Path where the trained model will be saved as a pickle file
  - Recommended: `static/model.pkl` (this is where the app looks for it)

**Example:**
```bash
python build_model.py --data data/articles.csv --out static/model.pkl
```

**What it does:**
1. Loads the CSV data
2. Trains a TF-IDF vectorizer on the article text
3. Trains a Multinomial Naive Bayes classifier
4. Saves the complete model (vectorizer + classifier) as a pickle file

### 3. Run the Application

```bash
python app.py
```

The application will start on `http://0.0.0.0:5000`

## Usage

1. Navigate to the home page (`/`)
2. Click "Get Started!" to go to the submission form
3. Paste an article snippet into the text area
4. Click "Classify" to predict the article type
5. View the predicted category and your original text

## Project Structure

```
ML_APP/
├── app.py                  # Flask web application
├── build_model.py          # Model training script
├── data/                   # Training data directory
│   └── articles.csv        # Training dataset (required)
├── static/                 # Static files
│   └── model.pkl           # Trained model (generated)
└── templates/              # HTML templates
    ├── index.html          # Home page
    ├── submit.html         # Text submission form
    └── predict.html        # Prediction results
```

## Model Details

**TextClassifier Class:**
- `fit(X, y)`: Train the model on text data and labels
- `predict(X)`: Predict class labels for new text
- `predict_proba(X)`: Get class probabilities for new text
- `score(X, y)`: Calculate accuracy on test data

**Components:**
- **TfidfVectorizer**: Converts text to numerical features using Term Frequency-Inverse Document Frequency
- **MultinomialNB**: Naive Bayes classifier optimized for text classification

## Notes

- The model must be retrained (`build_model.py`) whenever you want to update it with new data
- The pickled model file (`static/model.pkl`) must exist before running the web app
- Make sure your training data CSV has the required columns: `body` and `section_name`
