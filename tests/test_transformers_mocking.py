import pandas as pd
from unittest.mock import MagicMock, call
from pandas.util.testing import assert_frame_equal
from src.transformers import CountryFullTransformer


def test_correct_country_retured_with_simple_df():
    """
    Whenever get_region_from_code is called create mock. Side effect allows
    to return iterable, so that means everytime that function is call we
    iterate through those values. Then check if get_region_from_code was
    called, and it's called withing transformer somewhere within processing
    is happening. Mock verifies it the correct interaction is happening
    under the hood.
    """
    df = pd.DataFrame({'country': ["CA", "GB"]})

    country_transformer = CountryFullTransformer()
    country_transformer.get_region_from_code = MagicMock()
    country_transformer.get_region_from_code.side_effect = ["Canada",
                                                            "UK & Ireland"]

    expected_df = pd.DataFrame({'country': ["Canada", "UK & Ireland"]})
    result_df = country_transformer.transform(df)

    country_transformer.get_region_from_code.assert_has_calls(
        [call("CA"), call("GB")])
    assert_frame_equal(result_df, expected_df)
