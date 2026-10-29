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


class FeatureIntelligenceRequest(BaseModel):
    analysis_type: str = Field(default="importance", pattern="^(importance|pca|selection)$")
    target_column: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    n_components: int = Field(default=2, ge=2, le=10)
    random_seed: int = Field(default=42, ge=0, le=1000000)
    selection_method: str = Field(default="mutual_info", pattern="^(mutual_info|f_regression|f_classif|variance)$")

@app.post('/api/v1/datasets/{dataset_id}/feature-intelligence/analyze')
def analyze_feature_intelligence(dataset_id: str, request: FeatureIntelligenceRequest):
    """Run real feature-importance, PCA, or feature-selection calculations on an uploaded CSV."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif, f_regression, VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    path = (UPLOAD_DIR / f"{dataset_id}.csv").resolve()
    if not path.is_relative_to(UPLOAD_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a CSV first.")
    try:
        df = pd.read_csv(path)
        if df.empty or len(df.columns) < 2:
            raise HTTPException(status_code=422, detail="Dataset must contain rows and at least two columns.")
        target = request.target_column
        if target and target not in df.columns:
            raise HTTPException(status_code=422, detail=f"Target column '{target}' does not exist.")
        source = df.drop(columns=[target], errors='ignore') if request.analysis_type != 'importance' and target else df.drop(columns=[target], errors='ignore') if target else df.copy()
        # Drop columns that have no usable values; retain original names in the response.
        source = source.dropna(axis=1, how='all')
        if source.shape[1] == 0:
            raise HTTPException(status_code=422, detail="No usable feature columns were found.")
        numeric_cols = source.select_dtypes(include=['number', 'bool']).columns.tolist()
        categorical_cols = [c for c in source.columns if c not in numeric_cols]
        if request.analysis_type == 'pca':
            if len(numeric_cols) < 2:
                raise HTTPException(status_code=422, detail="PCA requires at least two numeric feature columns. Encode categorical columns first.")
            X = source[numeric_cols].apply(pd.to_numeric, errors='coerce')
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median()).fillna(0)
            Xs = StandardScaler().fit_transform(X)
            n = min(request.n_components, len(numeric_cols), max(2, len(Xs)))
            model = PCA(n_components=n, random_state=request.random_seed)
            transformed = model.fit_transform(Xs)
            return {"dataset_id": dataset_id, "analysis_type": "pca", "rows_used": int(len(X)), "features": numeric_cols, "components": int(n), "explained_variance_ratio": [round(float(v), 6) for v in model.explained_variance_ratio_], "cumulative_explained_variance": [round(float(v), 6) for v in np.cumsum(model.explained_variance_ratio_)], "loadings": [{"feature": col, **{f"PC{i+1}": round(float(model.components_[i, j]), 6) for i in range(n)}} for j, col in enumerate(numeric_cols)], "points": [{"row": int(i), **{f"PC{k+1}": round(float(transformed[i, k]), 6) for k in range(n)}} for i in range(min(len(transformed), 500))], "note": "PCA is fitted on this dataset for exploration. Fit it on training folds only when building a predictive model."}
        if request.analysis_type == 'importance':
            if not target:
                raise HTTPException(status_code=422, detail="Select a target column to calculate supervised feature importance.")
            y = df[target]
            valid = y.notna()
            X, y = source.loc[valid], y.loc[valid]
            if len(X) < 5 or y.nunique() < 2:
                raise HTTPException(status_code=422, detail="At least five valid rows and two target values are required.")
            is_classification = (y.dtype == 'object' or str(y.dtype).startswith('category') or y.nunique() <= min(20, max(2, int(len(y)*0.05))))
            if is_classification:
                y = y.astype(str)
                estimator = ExtraTreesClassifier(n_estimators=150, random_state=request.random_seed, n_jobs=-1, class_weight='balanced')
            else:
                y = pd.to_numeric(y, errors='coerce')
                valid_y = y.notna(); X, y = X.loc[valid_y], y.loc[valid_y]
                estimator = ExtraTreesRegressor(n_estimators=150, random_state=request.random_seed, n_jobs=-1)
            nums = X.select_dtypes(include=['number', 'bool']).columns.tolist()
            cats = [c for c in X.columns if c not in nums]
            transformers = []
            if nums: transformers.append(('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), nums))
            if cats: transformers.append(('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cats))
            if not transformers: raise HTTPException(status_code=422, detail="No usable feature columns remain.")
            prep = ColumnTransformer(transformers, remainder='drop')
            pipeline = Pipeline([('preprocess', prep), ('model', estimator)])
            # Exploratory ranking only. Production model evaluation should use a held-out split/CV.
            pipeline.fit(X, y)
            names = pipeline.named_steps['preprocess'].get_feature_names_out()
            values = pipeline.named_steps['model'].feature_importances_
            ranked = sorted([{"feature": str(name).split('__', 1)[-1], "importance": round(float(value), 6)} for name, value in zip(names, values)], key=lambda item:item['importance'], reverse=True)[:request.top_k]
            return {"dataset_id": dataset_id, "analysis_type": "importance", "target_column": target, "task": "classification" if is_classification else "regression", "method": "Extra Trees impurity-based importance", "rows_used": int(len(X)), "features_used": int(len(source.columns)), "importance": ranked, "note": "Exploratory importance fitted on the available dataset; it is not causal evidence and may be biased toward high-cardinality features."}
        # Feature selection
        if not target or request.selection_method == 'variance':
            X = source.select_dtypes(include=['number', 'bool']).apply(pd.to_numeric, errors='coerce')
            if X.shape[1] == 0: raise HTTPException(status_code=422, detail="Variance selection requires numeric features.")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median()).fillna(0)
            scores = X.var().to_dict()
            ranked = sorted([{"feature": str(k), "score": round(float(v), 8)} for k,v in scores.items()], key=lambda item:item['score'], reverse=True)
            return {"dataset_id": dataset_id, "analysis_type": "selection", "method": "variance ranking", "target_column": target, "selected_features": [x['feature'] for x in ranked[:request.top_k]], "ranking": ranked[:request.top_k], "note": "Variance ranking is unsupervised; low variance alone does not prove a feature is unhelpful."}
        y = df[target]
        valid = y.notna(); X, y = source.loc[valid], y.loc[valid]
        nums = X.select_dtypes(include=['number', 'bool']).columns.tolist(); cats = [c for c in X.columns if c not in nums]
        transformers=[]
        if nums: transformers.append(('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), nums))
        if cats: transformers.append(('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cats))
        prep = ColumnTransformer(transformers)
        Xt = prep.fit_transform(X)
        names = prep.get_feature_names_out()
        if hasattr(Xt, 'toarray'): Xt = Xt.toarray()
        if request.selection_method == 'mutual_info':
            classification = y.dtype == 'object' or y.nunique() <= min(20, max(2, int(len(y)*0.05)))
            if classification: scores = mutual_info_classif(Xt, y.astype(str), random_state=request.random_seed)
            else: scores = mutual_info_regression(Xt, pd.to_numeric(y, errors='coerce').fillna(pd.to_numeric(y, errors='coerce').median()), random_state=request.random_seed)
        elif request.selection_method == 'f_regression':
            scores = f_regression(Xt, pd.to_numeric(y, errors='coerce').fillna(pd.to_numeric(y, errors='coerce').median()))[0]
        else:
            scores = f_classif(Xt, y.astype(str))[0]
        ranked = sorted([{"feature": str(name).split('__',1)[-1], "score": round(float(score), 6) if np.isfinite(score) else 0.0} for name,score in zip(names,scores)], key=lambda item:item['score'], reverse=True)
        return {"dataset_id": dataset_id, "analysis_type": "selection", "method": request.selection_method, "target_column": target, "selected_features": [x['feature'] for x in ranked[:request.top_k]], "ranking": ranked[:request.top_k], "note": "Exploratory ranking on the available data. Fit selectors inside training folds to avoid data leakage during model evaluation."}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature intelligence analysis failed: {exc}")
