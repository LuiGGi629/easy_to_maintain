import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.transformers import GoalAdjustor, TimeTransformer

test_goal_transformer_testdata = [
    (pd.DataFrame({'goal': [5], 'static_usd_rate': [2]}),
     pd.DataFrame({'adjusted_goal': [10]})),
    (pd.DataFrame({'goal': [0], 'static_usd_rate': [1]}),
     pd.DataFrame({'adjusted_goal': [0]})),
    (pd.DataFrame({'goal': [0], 'static_usd_rate': [1]}),
     pd.DataFrame({'adjusted_goal': [0]})),
]


@pytest.mark.parametrize(
    "sample_df, expected_df", test_goal_transformer_testdata)
def test_goal_adjustor(sample_df, expected_df):
    """
    Parametrised tests provide a template parametrised by inputs and result to
    assert on. In this case it's a list with different scenarios that I want
    to test for. This list contains tuples. Tuple has input as the left hand
    side and the expected output as the right hand side. Decorator says:
    first you have to provide a string and this string is esentially the list
    of parameters that's gonna be maped to input and output.
    """
    adjustor = GoalAdjustor()

    result_df = adjustor.transform(sample_df)

    assert_frame_equal(result_df, expected_df)


test_time_transformer_testdata = [
    (pd.DataFrame({'deadline': [1459283229], 'created_at': [1455845363],
     'launched_at': [1456694829]}),
     pd.DataFrame({'launched_to_deadline': [29], 'created_to_launched': [9]})),
    (pd.DataFrame({'deadline': [0], 'created_at': [0], 'launched_at': [0]}),
     pd.DataFrame({'launched_to_deadline': [0], 'created_to_launched': [0]}))
]

@pytest.mark.parametrize(
    "sample_df, expected_df", test_time_transformer_testdata)
def test_time_transformer(sample_df, expected_df):
    time_transformer = TimeTransformer()

    result_df = time_transformer.transform(sample_df)

    assert_frame_equal(result_df, expected_df)
