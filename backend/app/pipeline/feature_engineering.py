"""Reusable, user-selectable tabular feature engineering operations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, PowerTransformer, QuantileTransformer, PolynomialFeatures, OneHotEncoder, OrdinalEncoder

SUPPORTED_OPERATIONS = [
 {"type":"impute_numeric","group":"Missing values","label":"Impute numeric columns","strategies":["mean","median","constant"]},
 {"type":"impute_categorical","group":"Missing values","label":"Impute categorical columns","strategies":["most_frequent","constant"]},
 {"type":"scale_numeric","group":"Scaling","label":"Scale numeric columns","strategies":["standard","minmax","robust","normalize"]},
 {"type":"log_transform","group":"Numeric transforms","label":"Log transform (log1p)","strategies":[]},
 {"type":"sqrt_transform","group":"Numeric transforms","label":"Square-root transform","strategies":[]},
 {"type":"power_transform","group":"Numeric transforms","label":"Power transform (Yeo-Johnson)","strategies":[]},
 {"type":"quantile_transform","group":"Numeric transforms","label":"Quantile transform","strategies":["normal","uniform"]},
 {"type":"bin_numeric","group":"Numeric transforms","label":"Bin numeric columns","strategies":["equal_width","quantile"]},
 {"type":"polynomial_features","group":"Feature creation","label":"Polynomial features","strategies":["degree_2","degree_3"]},
 {"type":"interaction_features","group":"Feature creation","label":"Pairwise interaction features","strategies":[]},
 {"type":"datetime_parts","group":"Date & time","label":"Extract datetime parts","strategies":["year","month","day","dayofweek","hour","quarter","is_weekend"]},
 {"type":"text_features","group":"Text","label":"Text length and word-count features","strategies":["char_count","word_count","digit_count"]},
 {"type":"one_hot_encode","group":"Categorical encoding","label":"One-hot encode categories","strategies":[]},
 {"type":"ordinal_encode","group":"Categorical encoding","label":"Ordinal encode categories","strategies":[]},
 {"type":"frequency_encode","group":"Categorical encoding","label":"Frequency encode categories","strategies":[]},
 {"type":"drop_columns","group":"Feature selection","label":"Drop selected columns","strategies":[]},
 {"type":"drop_low_variance","group":"Feature selection","label":"Remove low-variance numeric features","strategies":[]},
 {"type":"drop_correlated","group":"Feature selection","label":"Remove highly correlated numeric features","strategies":[]},
]

def _cols(df, op, numeric=False, exclude=None):
    cols = op.get('columns') or []
    cols = [c for c in cols if c in df.columns and c != exclude]
    if numeric:
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return cols

def apply_feature_operations(frame: pd.DataFrame, operations: list[dict], target: str | None = None):
    df = frame.copy()
    logs = []
    for index, op in enumerate(operations):
        kind = op.get('type')
        columns = _cols(df, op, exclude=target)
        strategy = op.get('strategy') or 'median'
        before_cols, before_rows = len(df.columns), len(df)
        if kind == 'impute_numeric':
            cols = _cols(df, op, True, target)
            for c in cols:
                s = pd.to_numeric(df[c], errors='coerce')
                fill = float(op.get('value', 0)) if strategy == 'constant' else (s.mean() if strategy == 'mean' else s.median())
                df[c] = s.fillna(0 if pd.isna(fill) else fill)
        elif kind == 'impute_categorical':
            for c in columns:
                if strategy == 'constant': fill = str(op.get('value', 'Unknown'))
                else:
                    modes = df[c].mode(dropna=True); fill = modes.iloc[0] if len(modes) else 'Unknown'
                df[c] = df[c].fillna(fill)
        elif kind == 'scale_numeric':
            cols = _cols(df, op, True, target)
            if cols:
                x = df[cols].apply(pd.to_numeric, errors='coerce')
                x = x.fillna(x.median()).fillna(0)
                scaler = {'standard':StandardScaler,'minmax':MinMaxScaler,'robust':RobustScaler,'normalize':Normalizer}.get(strategy, StandardScaler)()
                df[cols] = scaler.fit_transform(x)
        elif kind in ('log_transform','sqrt_transform','power_transform','quantile_transform'):
            cols = _cols(df, op, True, target)
            for c in cols:
                s = pd.to_numeric(df[c], errors='coerce')
                fill = s.median(); s = s.fillna(0 if pd.isna(fill) else fill)
                if kind == 'log_transform':
                    if (s < 0).any(): raise ValueError(f"{c}: log1p requires values >= -1; shift or choose another transform.")
                    df[f'{c}__log1p'] = np.log1p(s)
                elif kind == 'sqrt_transform':
                    if (s < 0).any(): raise ValueError(f"{c}: square root requires non-negative values.")
                    df[f'{c}__sqrt'] = np.sqrt(s)
                elif kind == 'power_transform':
                    df[f'{c}__yeojohnson'] = PowerTransformer(method='yeo-johnson', standardize=False).fit_transform(s.to_numpy().reshape(-1,1)).ravel()
                else:
                    output = 'normal' if strategy == 'normal' else 'uniform'
                    df[f'{c}__quantile'] = QuantileTransformer(n_quantiles=max(1,min(1000,len(s))), output_distribution=output, random_state=42).fit_transform(s.to_numpy().reshape(-1,1)).ravel()
        elif kind == 'bin_numeric':
            cols = _cols(df, op, True, target); bins = max(2,min(int(op.get('bins',5)),50))
            for c in cols:
                s = pd.to_numeric(df[c], errors='coerce')
                if s.nunique(dropna=True) < 2: continue
                df[f'{c}__bin'] = pd.qcut(s, q=bins, duplicates='drop').astype('string') if strategy == 'quantile' else pd.cut(s, bins=bins).astype('string')
                df[f'{c}__bin'] = df[f'{c}__bin'].fillna('Missing')
        elif kind == 'polynomial_features':
            cols = _cols(df, op, True, target)
            if len(cols) > 12: raise ValueError('Polynomial expansion is limited to 12 selected columns to control feature explosion.')
            if cols:
                degree = 3 if strategy == 'degree_3' else 2
                x = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                values = poly.fit_transform(x)
                names = poly.get_feature_names_out(cols)
                for j, name in enumerate(names):
                    if name not in cols and name not in df.columns: df[name.replace(' ', '__')] = values[:,j]
        elif kind == 'interaction_features':
            cols = _cols(df, op, True, target)
            if len(cols) > 12: raise ValueError('Interaction generation is limited to 12 selected columns.')
            for i, a in enumerate(cols):
                for b in cols[i+1:]: df[f'{a}__x__{b}'] = pd.to_numeric(df[a],errors='coerce').fillna(0) * pd.to_numeric(df[b],errors='coerce').fillna(0)
        elif kind == 'datetime_parts':
            parts = op.get('parts') or ['year','month','day','dayofweek']
            for c in columns:
                dt = pd.to_datetime(df[c], errors='coerce')
                for part in parts:
                    if part == 'year': val = dt.dt.year
                    elif part == 'month': val = dt.dt.month
                    elif part == 'day': val = dt.dt.day
                    elif part == 'dayofweek': val = dt.dt.dayofweek
                    elif part == 'hour': val = dt.dt.hour
                    elif part == 'quarter': val = dt.dt.quarter
                    elif part == 'is_weekend': val = dt.dt.dayofweek.isin([5,6]).astype('int8')
                    else: continue
                    df[f'{c}__{part}'] = val
        elif kind == 'text_features':
            feats = op.get('features') or ['char_count','word_count']
            for c in columns:
                s = df[c].fillna('').astype(str)
                if 'char_count' in feats: df[f'{c}__char_count'] = s.str.len()
                if 'word_count' in feats: df[f'{c}__word_count'] = s.str.split().str.len()
                if 'digit_count' in feats: df[f'{c}__digit_count'] = s.str.count(r'\d')
        elif kind == 'one_hot_encode':
            for c in columns:
                values = df[c].fillna('Missing').astype(str)
                dummies = pd.get_dummies(values, prefix=c, dtype='int8')
                if len(dummies.columns) > 100: raise ValueError(f'{c}: one-hot encoding would create more than 100 columns; group rare categories first.')
                df = pd.concat([df.drop(columns=[c]), dummies], axis=1)
        elif kind == 'ordinal_encode':
            for c in columns:
                values = df[c].fillna('Missing').astype(str)
                df[f'{c}__ordinal'] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(values.to_frame()).astype('int32')
                df.drop(columns=[c], inplace=True)
        elif kind == 'frequency_encode':
            for c in columns:
                values = df[c].fillna('Missing').astype(str)
                freq = values.value_counts(normalize=True)
                df[f'{c}__frequency'] = values.map(freq).astype(float)
                df.drop(columns=[c], inplace=True)
        elif kind == 'drop_columns':
            removable = [c for c in columns if c != target]
            if removable: df.drop(columns=removable, inplace=True)
        elif kind == 'drop_low_variance':
            threshold = float(op.get('threshold', 0.0))
            cols = _cols(df, op, True, target)
            drop = [c for c in cols if float(pd.to_numeric(df[c],errors='coerce').var()) <= threshold]
            if drop: df.drop(columns=drop, inplace=True)
        elif kind == 'drop_correlated':
            threshold = float(op.get('threshold', .95)); cols = _cols(df, op, True, target)
            if len(cols) > 1:
                corr = df[cols].corr().abs(); upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
                drop = [c for c in upper.columns if (upper[c] > threshold).any()]
                if drop: df.drop(columns=drop, inplace=True)
        else:
            raise ValueError(f'Unsupported feature engineering operation: {kind}')
        logs.append({'index':index+1,'type':kind,'columns':columns,'rows_before':before_rows,'rows_after':len(df),'columns_before':before_cols,'columns_after':len(df.columns),'columns_added':max(0,len(df.columns)-before_cols),'columns_removed':max(0,before_cols-len(df.columns)),'status':'succeeded'})
    return df, logs
