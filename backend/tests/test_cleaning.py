import pandas as pd
import pytest
from app.pipeline.cleaning import apply_operations, _quality


def test_fill_missing_median_and_report():
    source = pd.DataFrame({'age': [10, None, 30], 'city': ['a', 'b', None]})
    cleaned, logs = apply_operations(source, [{'type': 'fill_missing', 'columns': ['age'], 'strategy': 'median'}])
    assert cleaned['age'].tolist() == [10.0, 20.0, 30.0]
    assert _quality(cleaned)['missing_cells'] == 1
    assert logs[0]['changed'] == 1


def test_remove_duplicates_and_trim_text():
    source = pd.DataFrame({'name': [' Ana ', 'Ana', 'Bob'], 'n': [1, 1, 2]})
    cleaned, logs = apply_operations(source, [
        {'type': 'trim_text', 'columns': ['name']},
        {'type': 'drop_duplicates', 'columns': []},
    ])
    assert cleaned['name'].tolist() == ['Ana', 'Bob']
    assert logs[1]['changed'] == 1


def test_type_conversion_marks_invalid_values_missing():
    source = pd.DataFrame({'amount': ['1', 'oops', '3']})
    cleaned, logs = apply_operations(source, [{'type': 'convert_type', 'columns': ['amount'], 'target_type': 'numeric'}])
    assert cleaned['amount'].isna().sum() == 1
    assert logs[0]['changed'] == 1


def test_dropping_every_column_is_rejected():
    source = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    with pytest.raises(ValueError, match='every column'):
        apply_operations(source, [{'type': 'drop_columns', 'columns': ['a', 'b']}])


def test_unknown_column_is_rejected():
    with pytest.raises(ValueError, match='Unknown column'):
        apply_operations(pd.DataFrame({'a': [1, 2]}), [{'type': 'fill_missing', 'columns': ['missing'], 'strategy': 'median'}])
