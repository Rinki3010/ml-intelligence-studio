# Forma backend (local development)

## Run on Windows PowerShell

```powershell
cd backend
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/api/v1/health

## Run a full classification pipeline

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/pipelines/classification/run -ContentType 'application/json' -Body '{"config":{"samples":320,"random_seed":42,"test_size":0.25,"quality_threshold":0.7,"run_tuning":true}}'
```

The pipeline generates synthetic data, validates it, creates a stratified train/test split, trains a baseline Random Forest, tunes parameters with 3-fold GridSearchCV on training data only, evaluates both models against the same held-out test set, runs an F1 quality gate, and saves the dataset, models, trial results, feature importance, evaluation report, and logs under `../artifacts/<run_id>/`.

SQLite run metadata is stored in `backend/studio.sqlite3`. Never use this local development setup as an untrusted multi-user execution service.
