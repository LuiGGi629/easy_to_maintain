from hypothesis import given
from hypothesis.extra.pandas import column, data_frames
from hypothesis.strategies import fixed_dictionaries, from_regex
from src.transformers import GoalAdjustor, CategoriesExtractor
import json


# strategy
@given(data_frames(
    [column('goal', dtype=float), column('static_usd_rate', dtype=float)]))
def test_goal_adjustor(sample_df):
    """
    You need a strategy to generate inputs. Then it generates random
    DataFrames that have two columns with floating point numbers, and that
    gets passed to unittest everytime it's generated. Afterwards set up
    property expecting len of sample to be equal to result.
    """
    adjustor = GoalAdjustor()

    resulf_df = adjustor.transform(sample_df)
    # property / rule
    assert len(sample_df.index) == len(resulf_df.index)


# strategy
@given(fixed_dictionaries({'slug': from_regex("/")}).map(json.dumps))
def test_extract_categories_with_hypothesis(json_string):
    result = CategoriesExtractor.extract_categories(json_string, False)
    # property / rule
    assert len(result) == 2
