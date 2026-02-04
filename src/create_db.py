import os
import ast
import pandas as pd
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

# Get database URL from environment, default to data/articles.db
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/articles.db')
# Fix Heroku postgres:// URL to postgresql+psycopg:// for psycopg3
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql+psycopg://', 1)
elif DATABASE_URL and DATABASE_URL.startswith('postgresql://') and 'sqlite' not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg://', 1)

# Create engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define the Publishers table
class Publisher(Base):
    __tablename__ = 'publishers'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True)

# Define the Authors table
class Author(Base):
    __tablename__ = 'authors'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True)

# Define the Articles table
class Article(Base):
    __tablename__ = 'articles'
    
    id = Column(String, primary_key=True)
    headline = Column(JSON)
    pub_date = Column(DateTime)
    auth_id = Column(Integer, ForeignKey('authors.id'))
    pub_id = Column(Integer, ForeignKey('publishers.id'))
    section = Column(String)
    subsection = Column(String)
    subject = Column(String)  # Transformed subject from section and subsection
    pred_subject = Column(String)  # Predicted subject from trained model
    body = Column(Text)
    web_url = Column(String)

def normalize_byline(byline_value):
    """Ensure byline data is a dict with a 'person' list."""
    if isinstance(byline_value, str):
        try:
            byline_value = ast.literal_eval(byline_value)
        except Exception:
            return {'person': []}
    if isinstance(byline_value, dict):
        return byline_value
    if isinstance(byline_value, list):
        return {'person': byline_value}
    return {'person': []}


def normalize_headline(headline_value):
    """Ensure headline data is a dict with a 'main' field."""
    if isinstance(headline_value, str):
        try:
            headline_value = ast.literal_eval(headline_value)
        except Exception:
            return {'main': headline_value, 'print_headline': ''}
    if isinstance(headline_value, dict):
        headline_value.setdefault('main', '')
        headline_value.setdefault('print_headline', headline_value.get('main', ''))
        return headline_value
    return {'main': '', 'print_headline': ''}


def extract_person(byline_value):
    if isinstance(byline_value, dict):
        return byline_value.get('person', [])
    if isinstance(byline_value, list):
        return byline_value
    return []


def proper_case(name_part):
    if not isinstance(name_part, str):
        return ''
    return name_part.title() if name_part.isupper() else name_part


def first_person_name(person_list):
    if person_list:
        person = person_list[0]
        firstname = proper_case(person.get('firstname', '').strip())
        lastname = proper_case(person.get('lastname', '').strip())
        return f"{firstname} {lastname}".strip()
    return ''

def create_subject(section, subsection):
    """Transform section and subsection into a subject field.
    
    Returns empty string if section should be excluded from training,
    or if section is U.S./Sports and subsection is blank.
    """
    if not section:
        return ''
    
    # Filter out sections that should be excluded
    if section in ['Universal', 'NYT Now', 'Magazine', 'World']:
        return ''
    
    # For Sports sections, consolidate all into just "Sports"
    if section == 'Sports':
        return 'Sports'
    
    # For U.S. section, use Politics
    if section == 'U.S.':
        return ''
    
    # Apply section name mappings for consolidation
    mapping = {
        'Your Money': 'Business',
        'Job Market': 'Business',
        'Business Day': 'Business',
        'Arts': 'Arts & Entertainment',
        'Books': 'Arts & Entertainment',
        'Movies': 'Arts & Entertainment',
        'Theater': 'Arts & Entertainment',
        'Style': 'Lifestyle',
        'Fashion & Style': 'Lifestyle',
        'Food': 'Lifestyle',
        'Health': 'Lifestyle',
        'Great Homes & Destinations': 'Lifestyle',
        'Real Estate': 'Lifestyle',
        'Travel': 'Lifestyle',
        'Automobiles': 'Science & Technology',
        'Technology': 'Science & Technology',
        'Science': 'Science & Technology',
        'Education': 'Science & Technology',
        'Public Editor': 'Opinion',
        'The Upshot': 'Politics',
        'N.Y. / Region': 'Regional News',
        'Corrections': 'Opinion',
    }
    
    return mapping.get(section, section)

def load_articles_to_db():
    """Load articles from CSV into the database"""
    
    # Read CSV with required columns
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'articles.csv')
    df = pd.read_csv(csv_path)
    
    # Select only the required columns
    required_columns = ['_id', 'pub_date', 'source', 'section_name', 'headline', 'subsection_name', 'body', 'web_url', 'byline']
    df = df[required_columns]
    
    # Rename section_name to section for consistency
    df = df.rename(columns={'section_name': 'section'})

    # Clean up data - replace NaN with None
    df = df.where(pd.notna(df), None)

    # Normalize and extract author info
    df['byline'] = df['byline'].apply(normalize_byline)
    df['person'] = df['byline'].apply(extract_person)
    df['author'] = df['person'].apply(first_person_name)

    # Normalize headline to dict
    df['headline'] = df['headline'].apply(normalize_headline)

    # Convert pub_date to datetime
    df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
    
    # Drop all existing tables and create them fresh
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    
    # Insert data into database
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Get or create publishers
        publishers_map = {}
        for source in df['source'].unique():
            if source is not None:
                pub = session.query(Publisher).filter_by(name=source).first()
                if not pub:
                    pub = Publisher(name=source)
                    session.add(pub)
                    session.flush()
                publishers_map[source] = pub.id
        
        # Get or create authors
        authors_map = {}
        for author in df['author'].unique():
            if author is not None and author.strip():
                auth = session.query(Author).filter_by(name=author).first()
                if not auth:
                    auth = Author(name=author)
                    session.add(auth)
                    session.flush()
                authors_map[author] = auth.id
        
        # Insert articles with foreign keys
        for idx, row in df.iterrows():
            pub_id = publishers_map.get(row['source'])
            auth_id = authors_map.get(row['author'])
            
            # Create subject from section and subsection
            subject = create_subject(row['section'], row['subsection_name'])
            
            article = Article(
                id=str(row['_id']),
                pub_date=row['pub_date'],
                pub_id=pub_id,
                section=row['section'],
                headline=row['headline'],
                subsection=row['subsection_name'],
                subject=subject,
                body=row['body'],
                web_url=row['web_url'],
                auth_id=auth_id
            )
            session.add(article)
        
        session.commit()
        print(f"Successfully loaded {len(df)} articles into the database")
        print(f"Added {len(publishers_map)} publishers and {len(authors_map)} authors")
        
    except Exception as e:
        session.rollback()
        print(f"Error loading articles: {e}")
        raise
    finally:
        session.close()

if __name__ == '__main__':
    load_articles_to_db()
