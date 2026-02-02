release: python -u src/create_db.py && python -u src/build_model.py --out static/model.pkl
web: gunicorn app:app