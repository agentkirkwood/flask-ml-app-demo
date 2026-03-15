release: python -m src.create_db && python -m src.build_model --predictions-only --out static/model.pkl
web: gunicorn app:app