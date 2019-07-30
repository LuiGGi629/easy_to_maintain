import pandas as pd
from src.transformers import CountryTransformer


def test_correct_country_returned_with_simple_df():
    df = pd.DataFrame({'country': ["CA", "GB"]})
    country_transformer = CountryTransformer()

    result_df = country_transformer.transform(df)

    assert len(result_df.index) == 2
    assert result_df["country"][0] == "Canada"
    assert result_df["country"][1] == "UK & Ireland"
