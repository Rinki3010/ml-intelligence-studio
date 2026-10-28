"""Reusable, data-backed statistical profiling components.

Each analysis is a small function with a stable ID, input contract and JSON-safe output.
These functions can be called by the profiling UI, pipeline graph, or future tool adapters.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COMPONENTS = [
 {"id":"overview","name":"Dataset overview","category":"Structure","description":"Shape, types, duplicate rows and column-level metadata.","columns":0},
 {"id":"missingness","name":"Missing-value analysis","category":"Quality","description":"Missing counts, percentages and completeness by column.","columns":0},
 {"id":"numeric_summary","name":"Numeric statistics","category":"Statistics","description":"Mean, median, spread, quantiles, skewness and kurtosis.","columns":1},
 {"id":"distribution","name":"Distribution analysis","category":"Visual analysis","description":"Histogram bins and box-plot summary for a numeric column.","columns":1},
 {"id":"frequency","name":"Category frequency","category":"Statistics","description":"Counts and percentages for categorical values.","columns":1},
 {"id":"correlation","name":"Correlation matrix","category":"Relationships","description":"Pearson, Spearman or Kendall correlations between numeric columns.","columns":0},
 {"id":"outliers","name":"Outlier detection","category":"Quality","description":"IQR fences or Z-score detection with affected row indices.","columns":1},
 {"id":"normality","name":"Normality test","category":"Hypothesis testing","description":"Shapiro-Wilk test (sampled when n is large) and Q-Q points.","columns":1},
 {"id":"two_group_test","name":"Two-group hypothesis test","category":"Hypothesis testing","description":"Welch t-test or Mann–Whitney U for a numeric column by a two-level group.","columns":2},
 {"id":"anova","name":"Multiple-group test","category":"Hypothesis testing","description":"One-way ANOVA or Kruskal–Wallis across 2+ groups.","columns":2},
 {"id":"chi_square","name":"Categorical association","category":"Hypothesis testing","description":"Chi-square test of independence between two categorical columns.","columns":2},
 {"id":"linear_regression","name":"Simple linear regression","category":"Regression diagnostics","description":"Fit y against x and return fitted line, R², MAE, RMSE and residuals.","columns":2},
 {"id":"categorical_summary","name":"Categorical profile","category":"Statistics","description":"Cardinality, most common values, rare categories and imbalance.","columns":1},
 {"id":"feature_quality","name":"Feature quality & leakage clues","category":"Quality","description":"Find constant/near-constant columns, identifier-like fields, high cardinality and duplicates.","columns":0},
 {"id":"datetime_profile","name":"Date/time profile","category":"Visual analysis","description":"Parse date columns and inspect coverage, gaps, weekday and monthly patterns.","columns":1},
 {"id":"covariance_matrix","name":"Covariance matrix","category":"Relationships","description":"Measure pairwise covariance between numeric features.","columns":0},
 {"id":"feature_association","name":"Feature–target association","category":"Feature relevance","description":"Rank features against a selected target using correlation or mutual information.","columns":2},
 {"id":"pairwise_relationships","name":"Pairwise relationships","category":"Visual analysis","description":"Generate scatter points and correlation summaries for selected numeric features.","columns":0},
 {"id":"target_balance","name":"Target/class balance","category":"Feature relevance","description":"Inspect class proportions, imbalance ratio and rare classes before classification.","columns":1},
 {"id":"multicollinearity","name":"Multicollinearity (VIF) checks","category":"Relationships","description":"Estimate variance inflation factors to flag redundant numeric predictors.","columns":0},
]

def _clean(v: Any):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else round(float(v), 8)
    if isinstance(v, (np.bool_,)): return bool(v)
    if pd.isna(v): return None
    return v.item() if isinstance(v, np.generic) else v

def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns: raise ValueError(f"Column {col!r} does not exist.")
    s = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2: raise ValueError(f"Column {col!r} must contain at least two valid numeric values.")
    return s

def run_analysis(df: pd.DataFrame, analysis_id: str, columns: list[str] | None = None, config: dict | None = None) -> dict:
    columns = columns or []; config = config or {}
    unknown = [c for c in columns if c not in df.columns]
    if unknown: raise ValueError(f"Unknown columns: {unknown}")
    result: dict[str, Any] = {"component_id": analysis_id, "rows_analyzed": int(len(df)), "columns": columns}
    if analysis_id == 'overview':
        details=[]
        for c in df.columns:
            s=df[c]; details.append({"column":str(c),"dtype":str(s.dtype),"non_null":int(s.notna().sum()),"missing":int(s.isna().sum()),"missing_pct":_clean(s.isna().mean()*100),"unique":int(s.nunique(dropna=True),),"sample": [_clean(x) for x in s.dropna().head(3).tolist()]})
        result.update({"row_count":int(len(df)),"column_count":int(len(df.columns)),"duplicate_rows":int(df.duplicated().sum()),"memory_bytes":int(df.memory_usage(deep=True).sum()),"columns_detail":details})
    elif analysis_id == 'missingness':
        rows=[{"column":str(c),"missing":int(df[c].isna().sum()),"missing_pct":_clean(df[c].isna().mean()*100),"complete":int(df[c].notna().sum())} for c in df.columns]
        result.update({"missing_cells":int(df.isna().sum().sum()),"completeness_pct":_clean(100*(1-df.isna().sum().sum()/max(1,df.size))),"by_column":rows,"pattern_sample":[{"row":int(i),"missing_columns":[str(c) for c in df.columns[df.loc[i].isna()]]} for i in df.index[df.isna().any(axis=1)][:30]]})
    elif analysis_id == 'numeric_summary':
        if not columns: raise ValueError('Select at least one numeric column.')
        out=[]
        for c in columns:
            s=_numeric(df,c); q=s.quantile([.01,.05,.25,.5,.75,.95,.99])
            out.append({"column":c,"count":int(s.size),"mean":_clean(s.mean()),"median":_clean(s.median()),"std":_clean(s.std()),"variance":_clean(s.var()),"min":_clean(s.min()),"max":_clean(s.max()),"skewness":_clean(s.skew()),"kurtosis":_clean(s.kurtosis()),"quantiles":{str(k):_clean(v) for k,v in q.items()}})
        result["statistics"]=out
    elif analysis_id == 'distribution':
        if len(columns)!=1: raise ValueError('Select exactly one numeric column.')
        s=_numeric(df,columns[0]); bins=int(config.get('bins',20)); bins=max(5,min(100,bins)); counts,edges=np.histogram(s,bins=bins)
        q1,q3=s.quantile(.25),s.quantile(.75); iqr=q3-q1
        result.update({"column":columns[0],"histogram":[{"bin_start":_clean(edges[i]),"bin_end":_clean(edges[i+1]),"count":int(counts[i])} for i in range(len(counts))],"box":{"min":_clean(s.min()),"q1":_clean(q1),"median":_clean(s.median()),"q3":_clean(q3),"max":_clean(s.max()),"lower_fence":_clean(q1-1.5*iqr),"upper_fence":_clean(q3+1.5*iqr)},"sample_values":[_clean(v) for v in s.head(500).tolist()]})
    elif analysis_id == 'frequency':
        if len(columns)!=1: raise ValueError('Select exactly one column.')
        s=df[columns[0]].fillna('(missing)').astype(str); vc=s.value_counts(dropna=False).head(int(config.get('top_n',30)))
        result.update({"column":columns[0],"unique_count":int(s.nunique()),"frequencies":[{"value":str(k),"count":int(v),"pct":_clean(v/len(s)*100)} for k,v in vc.items()]})
    elif analysis_id == 'correlation':
        cols=columns or df.select_dtypes(include=np.number).columns.tolist(); numeric=df[cols].apply(pd.to_numeric,errors='coerce')
        numeric=numeric.select_dtypes(include=np.number).dropna(axis=1,how='all')
        if numeric.shape[1]<2: raise ValueError('Select at least two columns with numeric values.')
        method=config.get('method','pearson')
        if method not in ('pearson','spearman','kendall'): raise ValueError('method must be pearson, spearman or kendall.')
        corr=numeric.corr(method=method)
        result.update({"method":method,"matrix":[{"column":str(i),"values":[_clean(x) for x in corr.loc[i].tolist()]} for i in corr.index],"columns":[str(c) for c in corr.columns],"high_correlations":[{"x":str(corr.columns[i]),"y":str(corr.columns[j]),"correlation":_clean(corr.iloc[i,j])} for i in range(len(corr.columns)) for j in range(i+1,len(corr.columns)) if abs(corr.iloc[i,j])>=float(config.get('threshold',.8))]})
    elif analysis_id == 'outliers':
        if len(columns)!=1: raise ValueError('Select exactly one numeric column.')
        s=_numeric(df,columns[0]); method=config.get('method','iqr')
        if method=='zscore':
            sd=s.std(ddof=0); z=(s-s.mean())/(sd if sd else 1); mask=z.abs()>float(config.get('threshold',3)); score=z
        else:
            q1,q3=s.quantile(.25),s.quantile(.75); iqr=q3-q1; lo=q1-float(config.get('factor',1.5))*iqr; hi=q3+float(config.get('factor',1.5))*iqr; mask=(s<lo)|(s>hi); score=s
        ids=s.index[mask]
        result.update({"column":columns[0],"method":method,"outlier_count":int(mask.sum()),"outlier_pct":_clean(mask.mean()*100),"outlier_rows":[{"row_index":int(i),"value":_clean(s.loc[i]),"score":_clean(score.loc[i])} for i in ids[:500]],"threshold":_clean(config.get('threshold',3) if method=='zscore' else config.get('factor',1.5))})
    elif analysis_id == 'normality':
        if len(columns)!=1: raise ValueError('Select exactly one numeric column.')
        s=_numeric(df,columns[0]); sample=s.sample(n=min(5000,len(s)),random_state=42) if len(s)>5000 else s
        stat,p=stats.shapiro(sample); theoretical=np.sort(stats.norm.ppf((np.arange(len(sample))+.5)/len(sample))); observed=np.sort((sample-sample.mean())/(sample.std(ddof=1) or 1))
        result.update({"column":columns[0],"test":"Shapiro-Wilk","statistic":_clean(stat),"p_value":_clean(p),"alpha":float(config.get('alpha',.05)),"reject_normality":bool(p<float(config.get('alpha',.05))),"interpretation":"Evidence against normality at the selected alpha." if p<float(config.get('alpha',.05)) else "Insufficient evidence to reject normality; this does not prove normality.","qq_points":[{"theoretical":_clean(theoretical[i]),"observed":_clean(observed[i])} for i in np.linspace(0,len(sample)-1,min(100,len(sample))).astype(int)]})
    elif analysis_id in ('two_group_test','anova'):
        if len(columns)!=2: raise ValueError('Select a numeric outcome column and a group column.')
        y=pd.to_numeric(df[columns[0]],errors='coerce'); groups=df[columns[1]]; tmp=pd.DataFrame({'y':y,'g':groups}).dropna(); vals=[g.y.to_numpy() for _,g in tmp.groupby('g') if len(g)>=2]
        names=[str(k) for k,g in tmp.groupby('g') if len(g)>=2]
        if len(vals)<2: raise ValueError('At least two groups with two numeric observations each are required.')
        if analysis_id=='two_group_test' and len(vals)!=2: raise ValueError('Two-group test requires exactly two eligible groups; filter or choose another grouping column.')
        method=config.get('method','welch' if analysis_id=='two_group_test' else 'anova'); alpha=float(config.get('alpha',.05))
        if analysis_id=='two_group_test': stat,p=stats.ttest_ind(vals[0],vals[1],equal_var=False) if method=='welch' else stats.mannwhitneyu(vals[0],vals[1],alternative='two-sided')
        else: stat,p=stats.f_oneway(*vals) if method=='anova' else stats.kruskal(*vals)
        result.update({"test":method,"group_column":columns[1],"outcome_column":columns[0],"statistic":_clean(stat),"p_value":_clean(p),"alpha":alpha,"reject_null":bool(p<alpha),"group_summaries":[{"group":names[i],"n":int(len(v)),"mean":_clean(np.mean(v)),"median":_clean(np.median(v)),"std":_clean(np.std(v,ddof=1))} for i,v in enumerate(vals)],"interpretation":"Statistically significant difference/association detected at the selected alpha; inspect effect size and assumptions." if p<alpha else "No statistically significant difference detected at the selected alpha; this is not proof of equality."})
    elif analysis_id == 'chi_square':
        if len(columns)!=2: raise ValueError('Select two categorical columns.')
        table=pd.crosstab(df[columns[0]],df[columns[1]])
        if min(table.shape)<2: raise ValueError('Both columns must contain at least two observed categories.')
        chi,p,dof,expected=stats.chi2_contingency(table)
        result.update({"test":"chi-square independence","statistic":_clean(chi),"p_value":_clean(p),"degrees_of_freedom":int(dof),"alpha":float(config.get('alpha',.05)),"reject_null":bool(p<float(config.get('alpha',.05))),"rows":[str(x) for x in table.index],"columns_names":[str(x) for x in table.columns],"contingency":table.astype(int).values.tolist(),"expected_min_count":_clean(expected.min())})
    elif analysis_id == 'categorical_summary':
        if len(columns)!=1: raise ValueError('Select exactly one categorical column.')
        raw=df[columns[0]]; s=raw.astype('string').fillna('(missing)'); vc=s.value_counts(dropna=False)
        total=max(1,len(s)); top_n=max(5,min(100,int(config.get('top_n',20))))
        result.update({"column":columns[0],"count":int(len(s)),"unique_count":int(raw.nunique(dropna=True)),"missing_count":int(raw.isna().sum()),"dominant_value":str(vc.index[0]) if len(vc) else None,"dominant_pct":_clean(float(vc.iloc[0])/total*100) if len(vc) else None,"rare_category_count":int((vc/total*100<float(config.get('rare_pct',1))).sum()),"frequencies":[{"value":str(k),"count":int(v),"pct":_clean(v/total*100)} for k,v in vc.head(top_n).items()]})
    elif analysis_id == 'feature_quality':
        rows=[]; n=max(1,len(df))
        for c in df.columns:
            s=df[c]; counts=s.value_counts(dropna=False); unique=int(s.nunique(dropna=True)); unique_ratio=unique/n
            rows.append({"column":str(c),"dtype":str(s.dtype),"missing_pct":_clean(s.isna().mean()*100),"unique_count":unique,"unique_ratio_pct":_clean(unique_ratio*100),"dominant_pct":_clean((counts.iloc[0]/n*100) if len(counts) else 0),"constant":bool(unique<=1),"near_constant":bool(len(counts)>0 and counts.iloc[0]/n>=float(config.get('dominant_threshold',.98))),"identifier_like":bool(unique_ratio>=float(config.get('id_unique_ratio',.98)) and unique>1),"duplicate_values":int(s.duplicated().sum())})
        result.update({"feature_quality":rows,"constant_columns":[r['column'] for r in rows if r['constant']],"near_constant_columns":[r['column'] for r in rows if r['near_constant'] and not r['constant']],"identifier_like_columns":[r['column'] for r in rows if r['identifier_like']]})
    elif analysis_id == 'datetime_profile':
        if len(columns)!=1: raise ValueError('Select exactly one date/time column.')
        dt=pd.to_datetime(df[columns[0]],errors='coerce',utc=True).dropna()
        if dt.empty: raise ValueError('No values in this column could be parsed as dates or timestamps.')
        weekdays=dt.dt.day_name().value_counts(); months=dt.dt.tz_localize(None).dt.to_period('M').astype(str).value_counts().sort_index()
        result.update({"column":columns[0],"parsed_count":int(len(dt)),"unparsed_count":int(df[columns[0]].notna().sum()-len(dt)),"min_datetime":dt.min().isoformat(),"max_datetime":dt.max().isoformat(),"unique_days":int(dt.dt.date.nunique()),"weekday_counts":[{"value":k,"count":int(v)} for k,v in weekdays.items()],"monthly_counts":[{"value":str(k),"count":int(v)} for k,v in months.items()],"hour_counts":[{"value":str(k).zfill(2),"count":int(v)} for k,v in dt.dt.hour.value_counts().sort_index().items()]})
    elif analysis_id == 'covariance_matrix':
        cols=columns or df.select_dtypes(include=np.number).columns.tolist(); numeric=df[cols].apply(pd.to_numeric,errors='coerce').select_dtypes(include=np.number).dropna(axis=1,how='all')
        if numeric.shape[1]<2: raise ValueError('At least two numeric columns are required.')
        cov=numeric.cov(); result.update({"columns":[str(c) for c in cov.columns],"matrix":[{"column":str(c),"values":[_clean(v) for v in cov.loc[c].tolist()]} for c in cov.index]})
    elif analysis_id == 'feature_association':
        if len(columns)!=2: raise ValueError('Select a feature column and a target column.')
        feature,target=columns; tmp=df[[feature,target]].dropna()
        if len(tmp)<3: raise ValueError('At least three complete rows are required.')
        x=tmp[feature]; y=tmp[target]; xnum=pd.to_numeric(x,errors='coerce'); ynum=pd.to_numeric(y,errors='coerce')
        if xnum.notna().mean()>.9 and ynum.notna().mean()>.9:
            valid=xnum.notna() & ynum.notna(); corr=float(xnum[valid].corr(ynum[valid])); result.update({"association_type":"numeric-numeric","feature":feature,"target":target,"pearson_correlation":_clean(corr),"spearman_correlation":_clean(xnum[valid].corr(ynum[valid],method='spearman')),"absolute_strength":_clean(abs(corr)),"interpretation":"Association is descriptive and does not imply causation."})
        else:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            X=pd.DataFrame({"x":pd.factorize(x.astype(str))[0] if x.dtype=='object' or str(x.dtype).startswith('category') else xnum.fillna(xnum.median())})
            if ynum.notna().mean()>.9 and y.nunique()>10:
                target_values=ynum; valid=target_values.notna(); score=float(mutual_info_regression(X.loc[valid],target_values[valid],random_state=42)[0]); typ='mutual_information_regression'
            else:
                target_values=pd.factorize(y.astype(str))[0]; score=float(mutual_info_classif(X,target_values,random_state=42)[0]); typ='mutual_information_classification'
            result.update({"association_type":typ,"feature":feature,"target":target,"mutual_information":_clean(score),"interpretation":"Mutual information measures dependency, not direction or causation."})
    elif analysis_id == 'target_balance':
        if len(columns)!=1: raise ValueError('Select one target/class column.')
        s=df[columns[0]].astype('string').fillna('(missing)'); vc=s.value_counts(dropna=False); total=max(1,len(s))
        proportions=vc/total; positive=proportions[proportions>0]
        result.update({"target":columns[0],"class_count":int(len(vc)),"sample_count":int(total),"imbalance_ratio":_clean(float(positive.max()/positive.min())) if len(positive) else None,"classes":[{"class":str(k),"count":int(v),"pct":_clean(v/total*100)} for k,v in vc.items()],"minority_class":str(vc.index[-1]) if len(vc) else None,"minority_pct":_clean(float(vc.iloc[-1])/total*100) if len(vc) else None,"note":"For regression targets, use numeric distribution and outlier tools instead. A class balance result is descriptive, not a decision to resample automatically."})
    elif analysis_id == 'multicollinearity':
        cols=columns or df.select_dtypes(include=np.number).columns.tolist(); numeric=df[cols].apply(pd.to_numeric,errors='coerce').dropna(axis=1,how='all')
        numeric=numeric.loc[:,numeric.nunique(dropna=True)>1].dropna()
        if numeric.shape[1]<2 or len(numeric)<3: raise ValueError('Choose at least two non-constant numeric columns with three or more complete rows.')
        corr=numeric.corr().to_numpy(dtype=float); inv=np.linalg.pinv(corr); vifs=[]
        for i,c in enumerate(numeric.columns):
            value=float(inv[i,i]); vifs.append({"feature":str(c),"vif":_clean(value),"flagged":bool(value>=float(config.get('vif_threshold',5)))} )
        result.update({"vif_threshold":float(config.get('vif_threshold',5)),"vif_results":sorted(vifs,key=lambda r:(r['vif'] or 0),reverse=True),"flagged_features":[r['feature'] for r in vifs if r['flagged']],"note":"VIF is a linear multicollinearity diagnostic. Very high values can also occur when features are duplicated or nearly dependent."})
    elif analysis_id == 'pairwise_relationships':
        cols=columns or df.select_dtypes(include=np.number).columns.tolist()[:6]
        numeric=df[cols].apply(pd.to_numeric,errors='coerce').select_dtypes(include=np.number).dropna(axis=1,how='all')
        if numeric.shape[1]<2: raise ValueError('Select at least two numeric columns.')
        numeric=numeric.iloc[:, :8]; corr=numeric.corr(method='pearson'); pairs=[]
        for i,a in enumerate(numeric.columns):
            for b in numeric.columns[i+1:]:
                pair=numeric[[a,b]].dropna(); idx=np.linspace(0,len(pair)-1,min(250,len(pair))).astype(int) if len(pair) else []
                pairs.append({"x_column":str(a),"y_column":str(b),"correlation":_clean(pair[a].corr(pair[b])) if len(pair)>1 else None,"points":[{"x":_clean(pair[a].iloc[j]),"y":_clean(pair[b].iloc[j])} for j in idx]})
        result.update({"pair_count":len(pairs),"pairs":pairs,"note":"Scatter points are sampled for rendering; correlations use all complete pairwise observations."})
    elif analysis_id == 'linear_regression':
        if len(columns)!=2: raise ValueError('Select predictor x and outcome y.')
        tmp=pd.DataFrame({'x':pd.to_numeric(df[columns[0]],errors='coerce'),'y':pd.to_numeric(df[columns[1]],errors='coerce')}).dropna()
        if len(tmp)<3 or tmp.x.nunique()<2: raise ValueError('Regression requires at least 3 complete rows and at least 2 distinct predictor values.')
        model=LinearRegression().fit(tmp[['x']],tmp.y); pred=model.predict(tmp[['x']]); residual=tmp.y.to_numpy()-pred
        sample_idx=np.linspace(0,len(tmp)-1,min(300,len(tmp))).astype(int)
        result.update({"predictor":columns[0],"outcome":columns[1],"n":int(len(tmp)),"intercept":_clean(model.intercept_),"slope":_clean(model.coef_[0]),"r2":_clean(r2_score(tmp.y,pred)),"mae":_clean(mean_absolute_error(tmp.y,pred)),"rmse":_clean(np.sqrt(mean_squared_error(tmp.y,pred))),"points":[{"x":_clean(tmp.x.iloc[i]),"y":_clean(tmp.y.iloc[i]),"fitted":_clean(pred[i]),"residual":_clean(residual[i])} for i in sample_idx],"residual_summary":{"mean":_clean(np.mean(residual)),"std":_clean(np.std(residual,ddof=1)),"min":_clean(np.min(residual)),"max":_clean(np.max(residual))}})
    else: raise ValueError(f"Unknown profiling component {analysis_id!r}.")
    return result

class ProfilingComponent:
    """Reusable component contract shared by pipeline stages and future tool adapters."""
    component_id: str = ''
    def execute(self, df: pd.DataFrame, columns: list[str] | None = None, config: dict | None = None) -> dict:
        return run_analysis(df, self.component_id, columns, config)

# Stable component classes are exposed through a registry so graph execution and
# tool adapters can resolve a component by ID without knowing its implementation.
def _component_type(component_id: str):
    return type(''.join(part.title() for part in component_id.split('_')) + 'Component', (ProfilingComponent,), {'component_id': component_id, '__module__': __name__})

COMPONENT_CLASSES = {item['id']: _component_type(item['id']) for item in COMPONENTS}

def execute_component(df: pd.DataFrame, component_id: str, columns: list[str] | None = None, config: dict | None = None) -> dict:
    """Public stable entry point for pipeline stages, API endpoints and tool wrappers."""
    component_cls = COMPONENT_CLASSES.get(component_id)
    if component_cls is None:
        raise ValueError(f"Unknown profiling component {component_id!r}.")
    return component_cls().execute(df, columns, config)
