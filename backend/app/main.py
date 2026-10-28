from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.core.config import ARTIFACTS_DIR
from app.schemas import PipelineRunRequest
from app.components.registry import list_components
from app.pipeline.classification import run_classification_pipeline
from app.storage.sqlite_store import initialize_db, save_run, get_run, list_runs
from app.pipeline.csv_pipeline import UPLOAD_DIR, profile_csv, run_csv_pipeline
from app.pipeline.profiling import COMPONENTS as PROFILING_COMPONENTS, execute_component as run_profiling_analysis
from pydantic import BaseModel, Field
import uuid, shutil, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import joblib

app = FastAPI(title='Forma AI Intelligence Studio API', version='0.1.0', description='Local ML pipeline execution backend')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
initialize_db()


class CSVRunRequest(BaseModel):
    target_column: str
    task: str = Field(default="classification", pattern="^(classification|regression)$")
    test_size: float = Field(default=0.25, gt=0.1, lt=0.5)
    random_seed: int = Field(default=42, ge=0, le=1000000)
    run_tuning: bool = True
    quality_threshold: float = Field(default=0.7, ge=0, le=1)
    feature_options: dict = Field(default_factory=dict)

@app.post('/api/v1/datasets/upload')
def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail='Please upload a .csv file.')
    dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"
    path = UPLOAD_DIR / f"{dataset_id}.csv"
    try:
        with path.open('wb') as dest:
            shutil.copyfileobj(file.file, dest)
        if path.stat().st_size > 50 * 1024 * 1024:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=413, detail='CSV exceeds the 50 MB local prototype limit.')
        profile = profile_csv(path)
        if profile['columns_count'] < 2 or profile['rows'] < 2:
            raise HTTPException(status_code=400, detail='CSV must contain at least two columns and two rows.')
        return {'dataset_id': dataset_id, 'filename': Path(file.filename).name, **profile}
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f'Could not read CSV: {exc}')

@app.post('/api/v1/datasets/{dataset_id}/run')
def run_uploaded_dataset(dataset_id: str, request: CSVRunRequest):
    try:
        return run_csv_pipeline(dataset_id, request.target_column, request.task, request.test_size, request.random_seed, request.run_tuning, request.quality_threshold, request.feature_options)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        from app.storage.sqlite_store import get_run
        import logging
        logging.exception('Uploaded CSV pipeline failed')
        # Pipeline persists failure details when it reaches execution; expose a useful message.
        raise HTTPException(status_code=500, detail=str(exc))

class ProfilingAnalysisRequest(BaseModel):
    component_id: str
    columns: list[str] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)

@app.get('/api/v1/profiling/components')
def profiling_components():
    return {'items': PROFILING_COMPONENTS}

@app.post('/api/v1/datasets/{dataset_id}/profiling/analyze')
def analyze_uploaded_dataset(dataset_id: str, request: ProfilingAnalysisRequest):
    """Execute a reusable profiling/statistical component against the uploaded CSV."""
    if not dataset_id.startswith('dataset_') or not dataset_id.replace('_', '').isalnum():
        raise HTTPException(status_code=400, detail='Invalid dataset id.')
    path = (UPLOAD_DIR / f'{dataset_id}.csv').resolve()
    if not path.is_relative_to(UPLOAD_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail='Dataset not found. Upload the CSV again.')
    try:
        frame = pd.read_csv(path, low_memory=False)
        result = run_profiling_analysis(frame, request.component_id, request.columns, request.config)
        result.update({'dataset_id': dataset_id, 'filename': path.name, 'status': 'succeeded'})
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        import logging
        logging.exception('Profiling component failed')
        raise HTTPException(status_code=500, detail=f'Profiling component failed: {exc}')

@app.get('/api/v1/health')
def health():
    return {'status':'ok','service':'forma-backend','execution':'local'}

@app.get('/api/v1/components')
def components():
    return {'items':list_components()}

@app.post('/api/v1/pipelines/classification/run')
def run_pipeline(request: PipelineRunRequest):
    result = run_classification_pipeline(request.config)
    save_run(result)
    if result['status'] == 'failed':
        raise HTTPException(status_code=500, detail=result)
    return result

@app.get('/api/v1/runs')
def runs(limit: int = 20):
    if limit < 1 or limit > 100: raise HTTPException(status_code=422, detail='limit must be between 1 and 100')
    return {'items':list_runs(limit)}

@app.get('/api/v1/runs/{run_id}')
def run_detail(run_id: str):
    result = get_run(run_id)
    if not result: raise HTTPException(status_code=404, detail='Run not found')
    return result


@app.get('/api/v1/runs/{run_id}/analytics')
def run_analytics(run_id: str):
    """Return measured explainability and feature-engineering analytics for a CSV-backed run."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail='Run not found')
    dataset_id = run.get('dataset_id')
    if not dataset_id:
        raise HTTPException(status_code=422, detail='This run does not reference an uploaded CSV dataset.')
    run_dir = (ARTIFACTS_DIR / run_id).resolve()
    if not run_dir.is_relative_to(ARTIFACTS_DIR.resolve()):
        raise HTTPException(status_code=400, detail='Invalid run path')
    cache_path = run_dir / 'analytics.json'
    if cache_path.is_file():
        try:
            return json.loads(cache_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    report_path = run_dir / 'evaluation_report.json'
    model_path = run_dir / 'selected_model.joblib'
    dataset_path = (UPLOAD_DIR / f'{dataset_id}.csv').resolve()
    if not dataset_path.is_relative_to(UPLOAD_DIR.resolve()) or not dataset_path.is_file():
        raise HTTPException(status_code=404, detail='Uploaded dataset is no longer available. Upload it again and rerun.')
    if not report_path.is_file() or not model_path.is_file():
        raise HTTPException(status_code=404, detail='Run artifacts are incomplete.')
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        model = joblib.load(model_path)
        df = pd.read_csv(dataset_path)
        target = report['target_column']
        task = report['task']
        config = run.get('config', {})
        df = df.dropna(subset=[target])
        y = df[target]
        if task == 'classification':
            y = y.astype(str)
        else:
            y = pd.to_numeric(y, errors='coerce')
            valid = y.notna()
            df, y = df.loc[valid], y.loc[valid]
        X = df.drop(columns=[target]).copy()
        dropped = report.get('dropped_columns', [])
        X = X.drop(columns=[c for c in dropped if c in X.columns], errors='ignore')
        fe_options = report.get('feature_engineering', {}).get('options', {})
        for col in fe_options.get('log_transform_columns', []):
            if col in X.columns:
                X[col] = np.log1p(pd.to_numeric(X[col], errors='coerce'))
        stratify = y if task == 'classification' and y.value_counts().min() >= 2 else None
        _, X_test, _, y_test = train_test_split(X, y, test_size=float(config.get('test_size', .25)), random_state=int(config.get('random_seed', 42)), stratify=stratify)
        scoring = 'f1_weighted' if task == 'classification' else 'r2'
        perm = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42, scoring=scoring, n_jobs=1)
        permutation = sorted([{'feature': str(col), 'importance_mean': round(float(mean), 6), 'importance_std': round(float(std), 6)} for col, mean, std in zip(X_test.columns, perm.importances_mean, perm.importances_std)], key=lambda item: item['importance_mean'], reverse=True)
        sample = X_test.head(25)
        predictions = model.predict(sample)
        rows = []
        probabilities = None
        if task == 'classification' and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(sample)
        classes = [str(c) for c in getattr(model, 'classes_', [])]
        for idx, (row_idx, pred) in enumerate(zip(sample.index.tolist(), predictions)):
            item = {'row_index': int(row_idx), 'prediction': str(pred), 'actual': str(y_test.loc[row_idx])}
            if probabilities is not None:
                item['confidence'] = round(float(np.max(probabilities[idx])), 6)
                item['probabilities'] = {classes[j]: round(float(probabilities[idx][j]), 6) for j in range(len(classes))}
            rows.append(item)
        numeric = [str(c) for c in X.select_dtypes(include=['number', 'bool']).columns]
        categorical = [str(c) for c in X.columns if c not in numeric]
        feature_engineering = report.get('feature_engineering', {})
        if not feature_engineering:
            feature_engineering = {'numeric_features': numeric, 'categorical_features': categorical, 'dropped_columns': dropped, 'transformations': [], 'input_feature_count': len(X.columns), 'encoded_feature_count': None, 'notes': ['Transformations are applied by the fitted preprocessing pipeline.']}
        target_counts = y.astype(str).value_counts().head(30) if task == 'classification' else pd.Series(dtype=int)
        target_distribution = [{'label': str(label), 'count': int(count)} for label, count in target_counts.items()]
        analytics = {'run_id': run_id, 'dataset_id': dataset_id, 'task': task, 'target_column': target, 'explainability': {'method': 'Permutation importance on held-out test data', 'scoring_metric': scoring, 'importance': permutation[:30], 'prediction_samples': rows, 'caveat': 'Importance measures predictive reliance on this evaluation sample; it does not establish causation.'}, 'feature_engineering': feature_engineering, 'target_distribution': target_distribution, 'numeric_feature_correlations': df[numeric].corr(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0).round(3).to_dict() if len(numeric) > 1 else {}, 'data_quality': {'rows_used': int(len(df)), 'input_features': int(len(X.columns)), 'missing_cells_in_features': int(X.isna().sum().sum()), 'numeric_features': len(numeric), 'categorical_features': len(categorical), 'excluded_columns': dropped}}
        cache_path.write_text(json.dumps(analytics, indent=2, default=str), encoding='utf-8')
        return analytics
    except HTTPException:
        raise
    except Exception as exc:
        import logging
        logging.exception('Analytics generation failed for %s', run_id)
        raise HTTPException(status_code=500, detail=f'Could not generate analytics: {exc}')

@app.get('/api/v1/artifacts/{run_id}/{filename}')
def download_artifact(run_id: str, filename: str):
    if not run_id.startswith('run_') or Path(filename).name != filename or filename in {'.','..'}:
        raise HTTPException(status_code=400, detail='Invalid artifact path')
    path=(ARTIFACTS_DIR/run_id/filename).resolve()
    if not path.is_relative_to(ARTIFACTS_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail='Artifact not found')
    return FileResponse(path, filename=filename)
