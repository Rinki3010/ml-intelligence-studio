from __future__ import annotations
import json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from app.core.config import ARTIFACTS_DIR, DATA_DIR
from app.storage.sqlite_store import save_run

UPLOAD_DIR = DATA_DIR / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def profile_csv(path: Path) -> dict[str, Any]:
    # Fast schema preview: sample only a small prefix instead of scanning the full dataset.
    # Full data is read later only when the user starts training.
    df = pd.read_csv(path, nrows=1500, low_memory=False)
    columns=[]
    for col in df.columns:
        s=df[col]
        columns.append({'name':str(col),'dtype':str(s.dtype),'null_count':int(s.isna().sum()),'unique_count_sample':int(s.nunique(dropna=True)),'sample_values':s.dropna().astype(str).head(4).tolist()})
    return {'rows_previewed':int(len(df)),'rows':int(len(df)),'is_sampled':True,'columns_count':int(len(df.columns)),'columns':columns,'preview':df.head(8).replace({np.nan:None}).to_dict(orient='records'),'missing_cells':int(df.isna().sum().sum())}

def run_csv_pipeline(dataset_id: str, target_column: str, task: str, test_size: float=.25, random_seed: int=42, run_tuning: bool=True, quality_threshold: float=.7, feature_options: dict[str, Any] | None = None):
    path=(UPLOAD_DIR/f'{dataset_id}.csv').resolve()
    if not path.is_relative_to(UPLOAD_DIR.resolve()) or not path.is_file(): raise ValueError('Dataset not found. Upload the CSV again.')
    df=pd.read_csv(path)
    if target_column not in df.columns: raise ValueError(f'Target column {target_column!r} does not exist.')
    if len(df)<20: raise ValueError('Upload at least 20 rows for a meaningful train/test split.')
    if df[target_column].isna().any(): df=df.dropna(subset=[target_column])
    if df[target_column].nunique()<2: raise ValueError('Target must contain at least two distinct values.')
    if len(df)<20: raise ValueError('Not enough rows remain after removing rows with a missing target.')
    if task not in ('classification','regression'): raise ValueError('Task must be classification or regression.')
    y=df[target_column]
    if task=='classification':
        if y.nunique()>50 and pd.api.types.is_numeric_dtype(y): raise ValueError('This target has many numeric values. Choose regression or select a categorical target.')
        y=y.astype(str)
        stratify=y if y.value_counts().min()>=2 else None
    else:
        y=pd.to_numeric(y, errors='coerce'); valid=y.notna(); df=df.loc[valid]; y=y.loc[valid]
        if len(df)<20: raise ValueError('Regression target must be numeric and have at least 20 valid rows.')
        stratify=None
    feature_options = feature_options or {}
    numeric_imputation = feature_options.get('numeric_imputation', 'median')
    numeric_scaling = feature_options.get('numeric_scaling', 'standard')
    log_transform_columns = feature_options.get('log_transform_columns', []) or []
    if numeric_imputation not in ('median', 'mean'):
        raise ValueError('numeric_imputation must be median or mean.')
    if numeric_scaling not in ('standard', 'minmax', 'none'):
        raise ValueError('numeric_scaling must be standard, minmax, or none.')
    X=df.drop(columns=[target_column]).copy()
    # Avoid accidental training on unique identifiers and high-cardinality text columns.
    dropped=[]
    for col in list(X.columns):
        if X[col].nunique(dropna=True) >= max(20, int(.95*len(X))) and (('id' in str(col).lower()) or X[col].dtype=='object'):
            dropped.append(col); X=X.drop(columns=[col])
    if X.shape[1]==0: raise ValueError('No usable feature columns remain after excluding identifier-like columns.')
    invalid_log = [c for c in log_transform_columns if c not in X.columns]
    if invalid_log: raise ValueError(f'Unknown log-transform columns: {invalid_log}')
    for col in log_transform_columns:
        if not pd.api.types.is_numeric_dtype(X[col]): raise ValueError(f'Log1p transform is only supported for numeric columns; {col!r} is not numeric.')
        values = pd.to_numeric(X[col], errors='coerce')
        if (values.dropna() < 0).any(): raise ValueError(f'Log1p transform requires non-negative values; {col!r} contains negatives.')
        X[col] = np.log1p(values)
    numeric=list(X.select_dtypes(include=['number','bool']).columns)
    categorical=[c for c in X.columns if c not in numeric]
    transformers=[]
    if numeric:
        numeric_steps=[('imputer',SimpleImputer(strategy=numeric_imputation))]
        if numeric_scaling == 'standard': numeric_steps.append(('scaler',StandardScaler()))
        elif numeric_scaling == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            numeric_steps.append(('scaler',MinMaxScaler()))
        transformers.append(('numeric',Pipeline(numeric_steps),numeric))
    if categorical: transformers.append(('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore', sparse_output=True))]),categorical))
    prep=ColumnTransformer(transformers, remainder='drop')
    model_cls=RandomForestClassifier if task=='classification' else RandomForestRegressor
    model=model_cls(random_state=42, n_estimators=100, n_jobs=-1)
    pipe=Pipeline([('preprocess',prep),('model',model)])
    stratify=y if task=='classification' and y.value_counts().min()>=2 else None
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=random_seed,stratify=stratify)
    run_id=f'run_{uuid.uuid4().hex[:12]}'; run_dir=ARTIFACTS_DIR/run_id; run_dir.mkdir(parents=True,exist_ok=True)
    start=datetime.now(timezone.utc); tick=time.perf_counter(); logs=[]; nodes=[]
    def node(node_id, component, fn):
        t=time.perf_counter(); ts=datetime.now(timezone.utc).isoformat(); logs.append(f'Started {component}.')
        result=fn(); duration=round(time.perf_counter()-t,4)
        nodes.append({'node_id':node_id,'component':component,'status':'succeeded','started_at':ts,'finished_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':duration,'summary':f'{component} completed.','metrics':{}})
        return result
    try:
        node('profile','Profile CSV',lambda: logs.append(f'Loaded {len(df)} rows and {X.shape[1]} usable features; target={target_column}.'))
        baseline=node('train-baseline','Train Baseline Model',lambda: pipe.fit(X_train,y_train))
        def evaluate(m):
            pred=m.predict(X_test)
            if task=='classification':
                return {'accuracy':round(float(accuracy_score(y_test,pred)),4),'precision':round(float(precision_score(y_test,pred,average='weighted',zero_division=0)),4),'recall':round(float(recall_score(y_test,pred,average='weighted',zero_division=0)),4),'f1':round(float(f1_score(y_test,pred,average='weighted',zero_division=0)),4),'confusion_matrix':confusion_matrix(y_test,pred).tolist(),'classification_report':classification_report(y_test,pred,output_dict=True,zero_division=0)}
            return {'mae':round(float(mean_absolute_error(y_test,pred)),4),'rmse':round(float(np.sqrt(mean_squared_error(y_test,pred))),4),'r2':round(float(r2_score(y_test,pred)),4)}
        baseline_metrics=node('evaluate-baseline','Evaluate Baseline Model',lambda:evaluate(baseline))
        selected=baseline; tuned_metrics=baseline_metrics; trials=[]; best_params={}; tuning={'enabled':False,'trials':0}
        if run_tuning:
            param_grid={'model__n_estimators':[80,150], 'model__max_depth':[None,8], 'model__min_samples_leaf':[1,2]}
            scoring='f1_weighted' if task=='classification' else 'neg_mean_absolute_error'
            cv=min(3, int(y_train.value_counts().min())) if task=='classification' else 3
            if cv>=2:
                search=node('tune-model','Tune Hyperparameters',lambda:GridSearchCV(pipe,param_grid,scoring=scoring,cv=cv,n_jobs=-1,refit=True).fit(X_train,y_train))
                selected=search.best_estimator_; best_params=search.best_params_; tuned_metrics=evaluate(selected)
                results=search.cv_results_
                for i,p in enumerate(results['params']): trials.append({'trial':i+1,'params':p,'mean_cv_score':round(float(results['mean_test_score'][i]),4)})
                tuning={'enabled':True,'trials':len(trials),'best_cv_score':round(float(search.best_score_),4),'cv_folds':cv}
            else: logs.append('Skipped tuning because there are too few examples per class for cross-validation.')
        metric_name='f1' if task=='classification' else 'r2'
        selected_metric=selected.predict(X_test)
        selected_metrics=evaluate(selected)
        joblib.dump(baseline,run_dir/'baseline_model.joblib'); joblib.dump(selected,run_dir/'selected_model.joblib')
        if task=='classification' and hasattr(selected.named_steps['model'],'feature_importances_'):
            try:
                names=selected.named_steps['preprocess'].get_feature_names_out(); vals=selected.named_steps['model'].feature_importances_
                fi=sorted([{'feature':str(n),'importance':round(float(v),6)} for n,v in zip(names,vals)],key=lambda x:x['importance'],reverse=True)[:50]
            except Exception: fi=[]
        else: fi=[]
        pd.DataFrame(fi).to_csv(run_dir/'feature_importance.csv',index=False)
        feature_engineering = {
            'numeric_features': numeric,
            'categorical_features': categorical,
            'dropped_columns': dropped,
            'transformations': ([{'column_group':'numeric','steps':[f'{numeric_imputation} imputation'] + ([f'{numeric_scaling} scaling'] if numeric_scaling != 'none' else []) + ([f'log1p transform: {", ".join(log_transform_columns)}'] if log_transform_columns else []),'columns':numeric}] if numeric else []) + ([{'column_group':'categorical','steps':['most-frequent imputation','one-hot encoding'],'columns':categorical}] if categorical else []),
            'options': {'numeric_imputation': numeric_imputation, 'numeric_scaling': numeric_scaling, 'log_transform_columns': log_transform_columns},
            'target_column': target_column,
            'input_feature_count': int(len(numeric) + len(categorical)),
            'encoded_feature_count': int(len(selected.named_steps['preprocess'].get_feature_names_out())),
            'notes': ['Preprocessing is fitted inside the training pipeline to avoid leakage.', 'High-cardinality identifier-like columns were excluded when detected.']
        }
        (run_dir/'feature_engineering.json').write_text(json.dumps(feature_engineering,indent=2,default=str),encoding='utf-8')
        (run_dir/'tuning_trials.json').write_text(json.dumps(trials,indent=2,default=str),encoding='utf-8')
        (run_dir/'run_logs.txt').write_text('\n'.join(logs),encoding='utf-8')
        threshold_passed=selected_metrics.get(metric_name,0)>=quality_threshold if task=='classification' else selected_metrics.get(metric_name,-1)>=quality_threshold
        report={'run_id':run_id,'dataset_id':dataset_id,'task':task,'target_column':target_column,'rows':len(df),'feature_columns':list(X.columns),'dropped_columns':dropped,'baseline_metrics':baseline_metrics,'selected_metrics':selected_metrics,'best_params':best_params,'tuning':tuning,'feature_importance':fi,'feature_engineering':feature_engineering,'quality_gate':{'metric':metric_name,'value':selected_metrics.get(metric_name),'threshold':quality_threshold,'passed':bool(threshold_passed)},'logs':logs}
        (run_dir/'evaluation_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
        artifacts=[]
        for filename,kind,name in [('baseline_model.joblib','model','Baseline model'),('selected_model.joblib','model','Selected model'),('feature_importance.csv','explanation','Feature importance'),('feature_engineering.json','data','Feature engineering report'),('evaluation_report.json','report','Evaluation report'),('run_logs.txt','logs','Execution logs'),('tuning_trials.json','experiment','Tuning trials')]:
            f=run_dir/filename
            if f.exists(): artifacts.append({'id':f'{run_id}:{filename}','name':name,'kind':kind,'download_url':f'/api/v1/artifacts/{run_id}/{filename}','size_bytes':f.stat().st_size})
        logs.append('Pipeline completed successfully.')
        result={'run_id':run_id,'pipeline_id':f'csv-{task}-v1','dataset_id':dataset_id,'status':'succeeded','started_at':start.isoformat(),'finished_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':round(time.perf_counter()-tick,3),'config':{'target_column':target_column,'task':task,'test_size':test_size,'random_seed':random_seed,'run_tuning':run_tuning,'quality_threshold':quality_threshold},'nodes':nodes,'metrics':{'baseline':baseline_metrics,'selected':selected_metrics,'tuning':tuning},'comparison':{'baseline':baseline_metrics,'selected':selected_metrics,'delta':{k:round(selected_metrics[k]-baseline_metrics[k],4) for k in selected_metrics if isinstance(selected_metrics.get(k),(int,float)) and isinstance(baseline_metrics.get(k),(int,float))}},'quality_gate':report['quality_gate'],'artifacts':artifacts,'logs':logs,'dataset':{'rows':len(df),'features':len(X.columns),'target':target_column,'dropped_columns':dropped}}
        save_run(result); return result
    except Exception as exc:
        result={'run_id':run_id,'pipeline_id':f'csv-{task}-v1','dataset_id':dataset_id,'status':'failed','started_at':start.isoformat(),'finished_at':datetime.now(timezone.utc).isoformat(),'duration_seconds':round(time.perf_counter()-tick,3),'config':{'target_column':target_column,'task':task},'nodes':nodes,'metrics':{},'comparison':{},'quality_gate':{'passed':False,'reason':str(exc)},'artifacts':[],'logs':logs+[f'{type(exc).__name__}: {exc}']}
        save_run(result); raise
