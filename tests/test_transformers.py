import pandas as pd
from pandas.testing import assert_frame_equal
from src.transformers import CountryTransformer, GoalAdjustor, TimeTransformer


def test_correct_country_returned_with_simple_df():
    df = pd.DataFrame({'country': ["CA", "GB"]})
    country_transformer = CountryTransformer()

    result_df = country_transformer.transform(df)

    assert len(result_df.index) == 2
    assert result_df["country"][0] == "Canada"
    assert result_df["country"][1] == "UK & Ireland"


def test_unknown_country_returns_default():
    df = pd.DataFrame({'country': ["SA"]})
    country_transformer = CountryTransformer()

    result_df = country_transformer.transform(df)

    assert len(result_df.index) == 1
    assert result_df["country"][0] == "Other"


def test_time_transformer():
    """
    Creates pandas DataFrame which is sample data - that's the input.
    I know what the expected output is and I just want to assert that
    what we expect is the same as the result.
    """
    time_transformer = TimeTransformer()

    deadline_timestamp = 1459283229
    created_at_timestamp = 1455845363
    launched_at_timestamp = 1456694829

    sample_df = pd.DataFrame({'deadline': [deadline_timestamp],
                              'created_at': [created_at_timestamp],
                              'launched_at': [launched_at_timestamp]})

    expected_df = pd.DataFrame(
        {'launched_to_deadline': [29], 'created_to_launched': [9]})

    result_df = time_transformer.transform(sample_df)

    assert_frame_equal(result_df, expected_df)


def test_goal_adjustor_with_value():
    # given
    adjustor = GoalAdjustor()

    goal_value = 10
    usd_rate_value = 2

    sample_df = pd.DataFrame(
        {'goal': [goal_value], 'static_usd_rate': [usd_rate_value]})
    # when
    result_df = adjustor.transform(sample_df)
    # then
    expected_adjusted_goal_value = 20
    expected_df = pd.DataFrame(
        {'adjusted_goal': [expected_adjusted_goal_value]})
    assert_frame_equal(result_df, expected_df)
