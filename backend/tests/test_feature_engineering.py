import pandas as pd
from app.pipeline.feature_engineering import SUPPORTED_OPERATIONS, apply_feature_operations

def test_catalogue_has_all_groups_and_methods():
    assert len(SUPPORTED_OPERATIONS) >= 18
    assert {x['group'] for x in SUPPORTED_OPERATIONS} >= {'Missing values','Scaling','Numeric transforms','Feature creation','Date & time','Text','Categorical encoding','Feature selection'}

def test_impute_and_create_numeric_features():
    df = pd.DataFrame({'amount':[1.0,None,9.0], 'target':[0,1,0]})
    out, logs = apply_feature_operations(df, [
        {'type':'impute_numeric','columns':['amount'],'strategy':'median'},
        {'type':'log_transform','columns':['amount']},
        {'type':'bin_numeric','columns':['amount'],'strategy':'quantile','bins':2},
    ], target='target')
    assert out['amount'].isna().sum() == 0
    assert 'amount__log1p' in out and 'amount__bin' in out
    assert out['target'].tolist() == [0,1,0]
    assert len(logs) == 3

def test_categorical_encodings_and_text_features():
    df = pd.DataFrame({'city':['A','B','A'], 'description':['hello world','one','a 2']})
    out, _ = apply_feature_operations(df, [
        {'type':'one_hot_encode','columns':['city']},
        {'type':'text_features','columns':['description'],'features':['char_count','word_count','digit_count']},
    ])
    assert 'city_A' in out and 'city_B' in out
    assert 'description__word_count' in out and 'description__digit_count' in out

def test_target_column_is_not_scaled_or_dropped():
    df = pd.DataFrame({'x':[1,2,3], 'label':[0,1,0]})
    out, _ = apply_feature_operations(df, [
        {'type':'scale_numeric','columns':['x','label'],'strategy':'standard'},
        {'type':'drop_columns','columns':['label']},
    ], target='label')
    assert 'label' in out
    assert out['label'].tolist() == [0,1,0]
