import os
import ast
import pandas as pd
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///articles.db') # Default to SQLite for local testing

# Create engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Define the Articles table
class Article(Base):
    __tablename__ = 'articles'
    
    id = Column(String, primary_key=True)
    pub_date = Column(DateTime)
    source = Column(String)
    section_name = Column(String)
    body = Column(Text)
    web_url = Column(String)
    author = Column(String)


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

def load_articles_to_db():
    """Load articles from CSV into the database"""
    
    # Read CSV with required columns
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'articles.csv')
    df = pd.read_csv(csv_path)
    
    # Select only the required columns
    required_columns = ['pub_date', 'source', 'section_name', 'body', 'web_url', 'byline']
    df = df[required_columns]

    # Clean up data - replace NaN with None
    df = df.where(pd.notna(df), None)

    # Normalize and extract author info
    df['byline'] = df['byline'].apply(normalize_byline)
    df['person'] = df['byline'].apply(extract_person)
    df['author'] = df['person'].apply(first_person_name)

    # Convert pub_date to datetime
    df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Insert data into database
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        for idx, row in df.iterrows():
            article = Article(
                id=str(idx),
                pub_date=row['pub_date'],
                source=row['source'],
                section_name=row['section_name'],
                body=row['body'],
                web_url=row['web_url'],
                author=row['author']
            )
            session.add(article)
        
        session.commit()
        print(f"Successfully loaded {len(df)} articles into the database")
        
    except Exception as e:
        session.rollback()
        print(f"Error loading articles: {e}")
        raise
    finally:
        session.close()

if __name__ == '__main__':
    load_articles_to_db()
