import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from app.core.config import ARTIFACTS_DIR, DATA_DIR
from app.schemas import PipelineConfig

def utcnow():
    return datetime.now(timezone.utc)

def iso(dt):
    return dt.isoformat()

def run_classification_pipeline(config: PipelineConfig) -> dict[str, Any]:
    started = utcnow(); wall_start = time.perf_counter(); run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_dir = ARTIFACTS_DIR / run_id; run_dir.mkdir(parents=True, exist_ok=False)
    logs: list[str] = []
    node_results: list[dict[str, Any]] = []
    def execute_node(node_id, component, summary, fn):
        node_start = utcnow(); tick = time.perf_counter()
        try:
            result = fn()
            status = 'succeeded'
            return result
        except Exception as exc:
            status = 'failed'; logs.append(f"ERROR {component}: {type(exc).__name__}: {exc}")
            raise
        finally:
            finished = utcnow(); duration = round(time.perf_counter() - tick, 4)
            node_results.append({'node_id': node_id, 'component': component, 'status': status,
                'started_at': iso(node_start), 'finished_at': iso(finished), 'duration_seconds': duration,
                'summary': summary, 'metrics': {}})
            logs.append(f"{iso(finished)} | {component} | {status} | {duration:.3f}s")
    try:
        def generate():
            X, y = make_classification(n_samples=config.samples, n_features=12, n_informative=7,
                n_redundant=2, n_repeated=0, n_classes=2, weights=[0.55,0.45],
                class_sep=1.0, random_state=config.random_seed)
            cols = [f'feature_{i+1:02d}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols); df['target'] = y
            df.to_csv(run_dir / 'dataset.csv', index=False)
            return df, cols
        df, features = execute_node('generate-dataset','Generate Dataset',f'Generated {config.samples} rows with {12} features.',generate)

        def validate():
            if df.empty or 'target' not in df.columns: raise ValueError('Dataset is empty or target is missing.')
            missing = int(df.isna().sum().sum())
            if missing: raise ValueError(f'Dataset contains {missing} missing cells.')
            if df['target'].nunique() < 2: raise ValueError('Classification target must contain at least two classes.')
            return {'rows':len(df),'features':len(features),'missing_values':missing,'target_classes':int(df.target.nunique())}
        data_summary = execute_node('validate-data','Validate Data','Schema, target and missing-value checks passed.',validate)
        node_results[-1]['metrics'] = data_summary

        def split():
            return train_test_split(df[features],df['target'],test_size=config.test_size,random_state=config.random_seed,stratify=df['target'])
        X_train,X_test,y_train,y_test = execute_node('split-dataset','Split Dataset','Created stratified train and held-out test partitions.',split)
        data_summary.update({'train_rows':len(X_train),'test_rows':len(X_test),'test_size':config.test_size})
        node_results[-1]['metrics']={'train_rows':len(X_train),'test_rows':len(X_test),'test_size':config.test_size}

        def score(model, X, y):
            pred=model.predict(X)
            return {'accuracy':round(float(accuracy_score(y,pred)),4),'precision':round(float(precision_score(y,pred,zero_division=0)),4),
                    'recall':round(float(recall_score(y,pred,zero_division=0)),4),'f1':round(float(f1_score(y,pred,zero_division=0)),4),
                    'confusion_matrix':confusion_matrix(y,pred).tolist()}
        def train_baseline():
            model=RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=1,random_state=config.random_seed,n_jobs=-1)
            model.fit(X_train,y_train); return model
        baseline_model=execute_node('train-baseline','Train Baseline Model','Trained RandomForestClassifier on training data only.',train_baseline)
        baseline_metrics=score(baseline_model,X_test,y_test)
        node_results[-1]['metrics']=baseline_metrics

        best_model=baseline_model; best_params=baseline_model.get_params(deep=False); tuned_metrics=baseline_metrics; tuning_summary={'enabled':False,'trials':0}
        if config.run_tuning:
            def tune():
                search=GridSearchCV(RandomForestClassifier(random_state=config.random_seed,n_jobs=-1),
                    {'n_estimators':[60,100], 'max_depth':[None,8], 'min_samples_leaf':[1,2]},
                    scoring='f1',cv=3,n_jobs=-1,refit=True,return_train_score=False)
                search.fit(X_train,y_train)
                cv_results=[]
                for i,row in enumerate(search.cv_results_['params']):
                    cv_results.append({'trial':i+1,'params':row,'mean_cv_f1':round(float(search.cv_results_['mean_test_score'][i]),4),
                                       'std_cv_f1':round(float(search.cv_results_['std_test_score'][i]),4)})
                (run_dir/'tuning_trials.json').write_text(json.dumps(cv_results,indent=2),encoding='utf-8')
                return search.best_estimator_,search.best_params_,{'enabled':True,'trials':len(cv_results),'best_cv_f1':round(float(search.best_score_),4),'cv_folds':3},cv_results
            best_model,best_params,tuning_summary,trial_results=execute_node('tune-model','Tune Hyperparameters','Ran 8 candidate configurations with 3-fold CV on training data only.',tune)
            tuned_metrics=score(best_model,X_test,y_test)
            node_results[-1]['metrics']={'trials':tuning_summary['trials'],'best_cv_f1':tuning_summary['best_cv_f1'],'best_params':best_params}
        else:
            trial_results=[]
            node_results.append({'node_id':'tune-model','component':'Tune Hyperparameters','status':'skipped','started_at':iso(utcnow()),'finished_at':iso(utcnow()),'duration_seconds':0,'summary':'Tuning disabled by configuration.','metrics':{}})

        # Test data is only used for final evaluation; hyperparameter search never sees it.
        chosen_model=best_model if config.run_tuning else baseline_model
        chosen_metrics=tuned_metrics if config.run_tuning else baseline_metrics
        def evaluate():
            joblib.dump(baseline_model,run_dir/'baseline_model.joblib')
            joblib.dump(chosen_model,run_dir/'selected_model.joblib')
            importances=getattr(chosen_model,'feature_importances_',np.zeros(len(features)))
            imp=sorted([{'feature':f,'importance':round(float(v),6)} for f,v in zip(features,importances)],key=lambda x:x['importance'],reverse=True)
            pd.DataFrame(imp).to_csv(run_dir/'feature_importance.csv',index=False)
            return {'baseline':baseline_metrics,'tuned':tuned_metrics,'selected':chosen_metrics,'feature_importance':imp,
                    'classification_report':classification_report(y_test,chosen_model.predict(X_test),output_dict=True,zero_division=0)}
        evaluation=execute_node('evaluate-model','Evaluate Models','Evaluated baseline and tuned candidate on the same held-out test set.',evaluate)
        node_results[-1]['metrics']={'baseline_f1':baseline_metrics['f1'],'tuned_f1':tuned_metrics['f1'],'selected_f1':chosen_metrics['f1']}

        gate_passed=chosen_metrics['f1'] >= config.quality_threshold
        def quality_gate():
            return {'metric':'f1','value':chosen_metrics['f1'],'threshold':config.quality_threshold,'passed':gate_passed}
        gate=execute_node('quality-gate','Quality Gate',f"F1 {chosen_metrics['f1']:.3f} {'meets' if gate_passed else 'does not meet'} threshold {config.quality_threshold:.3f}.",quality_gate)
        node_results[-1]['metrics']=gate
        report={'run_id':run_id,'pipeline_id':'intelligent-classification-v1','dataset':{'rows':len(df),'features':features,'target':'target','train_rows':len(X_train),'test_rows':len(X_test),'random_seed':config.random_seed},
                'baseline_metrics':baseline_metrics,'tuned_metrics':tuned_metrics,'selected_metrics':chosen_metrics,'best_params':best_params,
                'tuning':tuning_summary,'quality_gate':gate,'feature_importance':evaluation['feature_importance'],
                'classification_report':evaluation['classification_report'],'node_results':node_results,'logs':logs}
        (run_dir/'evaluation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (run_dir/'run_logs.txt').write_text('\n'.join(logs),encoding='utf-8')
        # Copy dataset to shared data folder for discoverability; run-specific copy remains the source artifact.
        dataset_ref=f'{run_id}/dataset.csv'
        artifacts=[]
        for file,kind,name in [('dataset.csv','dataset','Generated dataset'),('baseline_model.joblib','model','Baseline model'),('selected_model.joblib','model','Selected model'),('feature_importance.csv','explanation','Feature importance'),('evaluation_report.json','report','Evaluation report'),('run_logs.txt','logs','Execution logs')]:
            path=run_dir/file
            artifacts.append({'id':f'{run_id}:{file}','name':name,'kind':kind,'download_url':f'/api/v1/artifacts/{run_id}/{file}','size_bytes':path.stat().st_size})
        if config.run_tuning:
            path=run_dir/'tuning_trials.json'; artifacts.append({'id':f'{run_id}:tuning_trials.json','name':'Tuning trials','kind':'experiment','download_url':f'/api/v1/artifacts/{run_id}/tuning_trials.json','size_bytes':path.stat().st_size})
        finished=utcnow(); duration=round(time.perf_counter()-wall_start,3)
        return {'run_id':run_id,'pipeline_id':'intelligent-classification-v1','status':'succeeded','started_at':iso(started),'finished_at':iso(finished),'duration_seconds':duration,
            'config':config.model_dump(),'nodes':node_results,'metrics':{'baseline':baseline_metrics,'tuned':tuned_metrics,'selected':chosen_metrics,'dataset':data_summary,'tuning':tuning_summary},
            'comparison':{'baseline_f1':baseline_metrics['f1'],'tuned_f1':tuned_metrics['f1'],'f1_delta':round(tuned_metrics['f1']-baseline_metrics['f1'],4),'best_params':best_params,'trials':trial_results},
            'quality_gate':gate,'artifacts':artifacts,'logs':logs}
    except Exception as exc:
        finished=utcnow(); duration=round(time.perf_counter()-wall_start,3)
        return {'run_id':run_id,'pipeline_id':'intelligent-classification-v1','status':'failed','started_at':iso(started),'finished_at':iso(finished),'duration_seconds':duration,
            'config':config.model_dump(),'nodes':node_results,'metrics':{},'comparison':{},'quality_gate':{'passed':False,'reason':str(exc)},'artifacts':[],
            'logs':logs+[f'Pipeline failed: {type(exc).__name__}: {exc}']}
