import pandas as pd
from app.pipeline.profiling import COMPONENTS, execute_component

def sample():
    return pd.DataFrame({'x':[1,2,3,4,5,6], 'y':[2,4,5,8,9,12], 'group':['a','a','b','b','a','b'], 'kind':['p','p','q','q','p',None]})

def test_registry_and_overview():
    assert len(COMPONENTS) >= 10
    out=execute_component(sample(),'overview')
    assert out['row_count']==6 and out['column_count']==4

def test_distribution_is_real_and_chart_ready():
    out=execute_component(sample(),'distribution',['x'],{'bins':5})
    assert sum(x['count'] for x in out['histogram']) == 6
    assert out['box']['median'] == 3.5

def test_regression_reports_metrics_and_points():
    out=execute_component(sample(),'linear_regression',['x','y'])
    assert out['n']==6 and out['r2'] > .9 and len(out['points'])==6

def test_hypothesis_test_uses_alpha():
    out=execute_component(sample(),'two_group_test',['x','group'],{'alpha':.05})
    assert 0 <= out['p_value'] <= 1 and out['alpha']==.05 and 'reject_null' in out

def test_unknown_component_rejected():
    try: execute_component(sample(),'not_a_component')
    except ValueError as exc: assert 'Unknown profiling component' in str(exc)
    else: raise AssertionError('unknown component should fail')

def test_new_profiling_components():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6], 'y': [2, 4, 5, 8, 9, 12],
        'group': ['a', 'a', 'b', 'b', 'a', 'b'],
        'kind': ['p', 'p', 'q', 'q', 'p', None],
        'date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-02-01', '2026-02-02', 'bad'],
        'constant': [1, 1, 1, 1, 1, 1], 'id': list(range(6)),
    })
    assert execute_component(df, 'categorical_summary', ['kind'])['unique_count'] == 2
    assert 'constant' in execute_component(df, 'feature_quality')['constant_columns']
    dt = execute_component(df, 'datetime_profile', ['date'])
    assert dt['parsed_count'] == 5 and dt['unparsed_count'] == 1
    assert len(execute_component(df, 'covariance_matrix', ['x', 'y'])['matrix']) == 2
    assert 'pearson_correlation' in execute_component(df, 'feature_association', ['x', 'y'])
    assert execute_component(df, 'pairwise_relationships', ['x', 'y'])['pair_count'] == 1

def test_target_balance_and_vif():
    df = pd.DataFrame({'a':[1,2,3,4,5,6], 'b':[2,4,6,8,10,12], 'target':['yes','yes','yes','yes','no','no']})
    balance = execute_component(df, 'target_balance', ['target'])
    assert balance['class_count'] == 2 and balance['sample_count'] == 6
    vif = execute_component(df, 'multicollinearity', ['a', 'b'])
    assert len(vif['vif_results']) == 2
