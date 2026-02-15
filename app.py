from flask import Flask, render_template, request, jsonify, send_from_directory, make_response, redirect, url_for
from flask.wrappers import Response
import pickle
from src.build_model import TextClassifier, get_data
import os
import math
import argparse
from typing import Any
from urllib.parse import urlencode
from sqlalchemy import create_engine, or_, func
from sqlalchemy.orm import sessionmaker
from src.create_db import Article, Author, Publisher, DATABASE_URL
import random

app = Flask(__name__)

# Setup database connection
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def get_db_session():
    """Create a new database session"""
    return Session()

# Add route to serve Bootstrap files from shared directory
@app.route('/bootstrap/<path:filename>')
def bootstrap_static(filename: str) -> Response:
    bootstrap_dir = os.path.join(os.path.dirname(__file__), '..', 'bootstrap')
    return send_from_directory(bootstrap_dir, filename)


model: TextClassifier | None = None

def get_model() -> TextClassifier:
    """Lazy-load or train the text classifier model.
    
    This function implements lazy initialization: on the first call, it either
    loads an existing model from disk or trains a new one from database.
    Subsequent calls return the cached model without retraining.
    
    After loading/training, it also auto-predicts subjects for any articles
    that don't yet have predictions (pred_subject is None or empty).
    
    Returns:
        TextClassifier: The trained and cached model object
    """
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
        
        # Auto-predict subjects for articles that don't have predictions yet
        # This ensures pred_subject is populated after model load/train
        print("\nChecking for articles without predictions...")
        db_session = get_db_session()
        try:
            articles_to_predict = db_session.query(Article).filter(
                (Article.pred_subject.is_(None)) | (Article.pred_subject == '')
            ).all()
            
            if articles_to_predict:
                total_articles = len(articles_to_predict)
                print(f"Found {total_articles} articles without predictions. Generating...")
                
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
                
                print(f"\n  Successfully predicted {total_articles} articles")
            else:
                print("All articles already have predictions.")
        except Exception as e:
            db_session.rollback()
            print(f"Error predicting subjects: {e}")
            raise
        finally:
            db_session.close()
    
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
def predict() -> Response:
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
def articles() -> Response:
    """Display a paginated list of articles with optional search."""
    session = get_db_session()
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 25, type=int)
        query_text = request.args.get('q', '', type=str).strip()
        subject_filter = [s.strip() for s in request.args.getlist('subject') if s.strip()]
        publisher_filter = [p.strip() for p in request.args.getlist('publisher') if p.strip()]
        sort_key = request.args.get('sort', 'random', type=str).strip()
        random_seed = request.args.get('seed', type=int)
        partial = request.args.get('partial', 0, type=int)
        include_filters = request.args.get('include_filters', 0, type=int)

        # Generate a new seed if not provided and sort is random
        if sort_key == 'random' and random_seed is None:
            random_seed = random.randint(1, 2147483647)
            # If this is NOT a partial request, redirect to add seed to URL
            if not partial:
                redirect_params = request.args.copy()
                redirect_params['seed'] = random_seed
                redirect_params['sort'] = 'random'  # Add this line
                return redirect(url_for('articles', **redirect_params))

        allowed_sorts = {
            'random': 'random',
            'date_desc': 'date_desc',
            'date_asc': 'date_asc',
            'author': 'author',
            'publisher': 'publisher'
        }
        sort_key = allowed_sorts.get(sort_key, 'random')

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
            base_query = base_query.filter(Article.subject.in_(subject_filter))

        if publisher_filter:
            base_query = base_query.filter(Publisher.name.in_(publisher_filter))

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

        # Apply sorting
        if sort_key == 'date_desc':
            base_query = base_query.order_by(Article.pub_date.desc())
        elif sort_key == 'date_asc':
            base_query = base_query.order_by(Article.pub_date.asc())
        elif sort_key == 'author':
            base_query = base_query.order_by(Author.name)
        elif sort_key == 'publisher':
            base_query = base_query.order_by(Publisher.name)
        else:  # random
            results = base_query.all()
            rng = random.Random(random_seed)
            rng.shuffle(results)
            total = len(results)
            start = (page - 1) * per_page
            end = start + per_page
            articles = results[start:end]

        # For non-random sorts, use normal pagination
        if sort_key != 'random':
            total = base_query.count()
            articles = base_query.offset((page - 1) * per_page).limit(per_page).all()

        total_pages = math.ceil(total / per_page) if total > 0 else 1

        query_params = request.args.to_dict(flat=False)
        query_params.pop('partial', None)
        query_string = urlencode(query_params, doseq=True)

        def apply_search_filters(query):
            if query_text:
                like_pattern = f"%{query_text}%"
                query = query.filter(or_(
                    Article.headline['main'].as_string().ilike(like_pattern),
                    Article.body.ilike(like_pattern),
                    Author.name.ilike(like_pattern),
                    Publisher.name.ilike(like_pattern),
                    Article.section.ilike(like_pattern),
                    Article.subsection.ilike(like_pattern)
                ))
            return query

        # Always fetch subjects and publishers for the full page render
        subject_query = session.query(Article.subject)
        subject_query = subject_query.outerjoin(Publisher, Article.pub_id == Publisher.id)\
            .outerjoin(Author, Article.auth_id == Author.id)
        subject_query = apply_search_filters(subject_query)
        if publisher_filter:
            subject_query = subject_query.filter(Publisher.name.in_(publisher_filter))
        subjects = [row[0] for row in subject_query
                    .filter(Article.subject.isnot(None))
                    .filter(Article.subject != '')
                    .distinct()
                    .order_by(Article.subject.asc())
                    .all()]

        publisher_query = session.query(Publisher.name)
        publisher_query = publisher_query.join(Article, Article.pub_id == Publisher.id)\
            .outerjoin(Author, Article.auth_id == Author.id)
        publisher_query = apply_search_filters(publisher_query)
        if subject_filter:
            publisher_query = publisher_query.filter(Article.subject.in_(subject_filter))
        publishers = [row[0] for row in publisher_query
                      .filter(Publisher.name.isnot(None))
                      .filter(Publisher.name != '')
                      .distinct()
                      .order_by(Publisher.name.asc())
                      .all()]

        if partial and include_filters:
            return jsonify({
                'html': render_template('partials/_article_cards.html', articles=articles, query_params=query_string),
                'total': total,
                'total_pages': total_pages,
                'page': page,
                'subjects': subjects,
                'publishers': publishers,
                'seed': random_seed
            })

        if partial:
            return jsonify({
                'html': render_template('partials/_article_cards.html', articles=articles, query_params=query_string),
                'total': total,
                'total_pages': total_pages,
                'page': page,
                'subjects': None,
                'publishers': None,
                'seed': random_seed
            })

        return render_template(
            'articles.html',
            articles=articles,
            query_params=query_string,
            total=total,
            total_pages=total_pages,
            page=page,
            per_page=per_page,
            query_text=query_text,
            subject_filter=subject_filter,
            publisher_filter=publisher_filter,
            sort_key=sort_key,
            seed=random_seed,
            subjects=subjects,
            publishers=publishers
        )
    finally:
        session.close()

@app.route('/articles/<article_id>', methods=['GET'])
def article_detail(article_id) -> Response:
    """Display detailed information for a specific article."""
    seed = request.args.get('seed', type=int)
    sort_key = request.args.get('sort', 'random', type=str).strip()
    
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
        
        query_params = request.args.to_dict(flat=False)
        query_params.pop('partial', None)
        query_string = urlencode(query_params, doseq=True)

        context: dict[str, Any] = {
            'article': article,
            'seed': seed,
            'sort': sort_key,
            'author_name': author_name,
            'publisher_name': publisher_name,
            'query_params': query_string
        }
        return render_template('article_detail.html', **context)
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
