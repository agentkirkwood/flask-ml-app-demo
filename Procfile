release: python -u -m src.create_db && python -u -m src.build_model --predictions-only --out static/model.pkl
web: gunicorn app:app