from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
from src.build_model import TextClassifier, get_data # type: ignore
import os
import math
import argparse
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import sessionmaker
from src.create_db import Article, Author, Publisher, DATABASE_URL

app = Flask(__name__)

# Setup database connection
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def get_db_session():
    """Create a new database session"""
    return Session()

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

@app.route('/about', methods=['GET'])
def about():
    """Render the About page."""
    return render_template('about.html')

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

@app.route('/articles', methods=['GET'])
def articles():
    """Display a paginated list of articles with optional search."""
    session = get_db_session()
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 25, type=int)
        query_text = request.args.get('q', '', type=str).strip()
        subject_filter = request.args.get('subject', '', type=str).strip()
        publisher_filter = request.args.get('publisher', '', type=str).strip()
        partial = request.args.get('partial', 0, type=int)

        base_query = session.query(
            Article.id,
            Article.headline,
            Article.pub_date,
            Article.subject,
            Article.pred_subject,
            Author.name.label('author_name'),
            Publisher.name.label('publisher_name')
        ).outerjoin(Author, Article.auth_id == Author.id)\
         .outerjoin(Publisher, Article.pub_id == Publisher.id)

        if subject_filter:
            base_query = base_query.filter(Article.subject == subject_filter)

        if publisher_filter:
            base_query = base_query.filter(Publisher.name == publisher_filter)

        if query_text:
            like_pattern = f"%{query_text}%"
            base_query = base_query.filter(or_(
                Article.headline['main'].as_string().ilike(like_pattern),
                Article.body.ilike(like_pattern),
                Author.name.ilike(like_pattern),
                Publisher.name.ilike(like_pattern),
                Article.section.ilike(like_pattern),
                Article.subsection.ilike(like_pattern)
            ))

        total = base_query.order_by(None).count()
        total_pages = max(1, math.ceil(total / per_page)) if per_page > 0 else 1
        page = max(1, min(page, total_pages))

        articles = base_query.order_by(Article.pub_date.desc())\
            .offset((page - 1) * per_page)\
            .limit(per_page)\
            .all()

        if partial:
            return render_template('partials/_article_cards.html', articles=articles)

        subjects = [row[0] for row in session.query(Article.subject)
                    .filter(Article.subject.isnot(None))
                    .filter(Article.subject != '')
                    .distinct()
                    .order_by(Article.subject.asc())
                    .all()]

        publishers = [row[0] for row in session.query(Publisher.name)
                      .filter(Publisher.name.isnot(None))
                      .filter(Publisher.name != '')
                      .distinct()
                      .order_by(Publisher.name.asc())
                      .all()]

        return render_template(
            'articles.html',
            articles=articles,
            page=page,
            per_page=per_page,
            total=total,
            total_pages=total_pages,
            query_text=query_text,
            subject_filter=subject_filter,
            publisher_filter=publisher_filter,
            subjects=subjects,
            publishers=publishers
        )
    finally:
        session.close()

@app.route('/articles/<article_id>', methods=['GET'])
def article_detail(article_id):
    """Display detailed information for a specific article."""
    session = get_db_session()
    try:
        # Query the specific article with author and publisher info
        article_data = session.query(
            Article,
            Author.name.label('author_name'),
            Publisher.name.label('publisher_name')
        ).outerjoin(Author, Article.auth_id == Author.id)\
         .outerjoin(Publisher, Article.pub_id == Publisher.id)\
         .filter(Article.id == article_id)\
         .first()
        
        if not article_data:
            return "Article not found", 404
        
        article, author_name, publisher_name = article_data
        
        return render_template('article_detail.html', 
                             article=article,
                             author_name=author_name,
                             publisher_name=publisher_name)
    finally:
        session.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Flask app with optional rebuild commands')
    parser.add_argument('--db_rebuild', action='store_true', 
                       help='Rebuild the database from CSV before starting the app (also rebuilds model since it is downstream of the DB)')
    parser.add_argument('--model_rebuild', action='store_true',
                       help='Rebuild the machine learning model before starting the app')
    args = parser.parse_args()
    
    # NOTE: The model is downstream of the database (depends on DB data for training)
    # If the database is rebuilt, the model must also be rebuilt to reflect changes
    # in the data or schema (e.g., new subject field)
    
    # Rebuild database if requested
    if args.db_rebuild:
        print("Rebuilding database...")
        from src.create_db import load_articles_to_db
        load_articles_to_db()
        print("Database rebuild complete!")
        # Automatically rebuild model since it depends on DB data
        args.model_rebuild = True
        print("Model rebuild triggered (database is upstream of model)")
    
    # Rebuild model if requested (or triggered by db_rebuild)
    if args.model_rebuild:
        print("Rebuilding model...")
        model_path = 'static/model.pkl'
        
        # Remove old model if it exists
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Removed existing model at {model_path}")
        
        # Build new model using current database data
        os.makedirs('static', exist_ok=True)
        X, y = get_data()
        print(f"Data loaded successfully. Total articles: {len(X)}")
        print(f"Unique subjects: {len(set(y))}")
        
        model = TextClassifier()
        model.fit(X, y)
        print("Model training complete.")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
        
        # Predict subjects for all articles and update database
        print("\nGenerating predictions for all articles (only for those without predictions)...")
        db_session = get_db_session()
        try:
            # Get only articles without pred_subject to avoid re-predicting on each model rebuild
            articles_to_predict = db_session.query(Article).filter(
                (Article.pred_subject.is_(None)) | (Article.pred_subject == '')
            ).all()
            total_articles = len(articles_to_predict)
            
            if total_articles == 0:
                print("All articles already have predictions. Skipping prediction step.")
            else:
                print(f"Found {total_articles} articles without predictions")
                
                # Predict in batches for efficiency
                batch_size = 100
                for i in range(0, total_articles, batch_size):
                    batch = articles_to_predict[i:i + batch_size]
                    bodies = [article.body for article in batch]
                    predictions = model.predict(bodies)
                    
                    for article, pred in zip(batch, predictions):
                        article.pred_subject = pred
                    
                    db_session.commit()
                    progress = min(i + batch_size, total_articles)
                    print(f"  Predicted {progress}/{total_articles} articles", end='\r', flush=True)
                
                print(f"  Predicted {total_articles}/{total_articles} articles")
                print(f"Successfully predicted subjects for {total_articles} articles")
        except Exception as e:
            db_session.rollback()
            print(f"Error predicting subjects: {e}")
            raise
        finally:
            db_session.close()
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000)) #finds port set by Heroku or defaults to 5000
    app.run(host='0.0.0.0'
            , port=port
            , debug=False
            , use_reloader=False
            , threaded=True)
 