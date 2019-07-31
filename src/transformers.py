import json
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoriesExtractor(BaseEstimator, TransformerMixin):
    """
    Extract categories from json string. By default it will only keep the
    hardcoded categories defined below to avoid having too many dummies.
    """
    misc = "misc"
    gen_cats = ["music", "film & video", "publishing", "art", "games"]
    precise_cats = [
        "rock", "fiction", "webseries", "indie rock", "children's books",
        "shorts", "documentary", "video games"
    ]

    @classmethod
    def extract_categories(cls, json_string, validate=True):
        """
        Defines a parameter validate. That is a trick, that allows to
        decide whether you want to extract all categories or only hard coded
        initially e.g.gen_cats, precise cats -> which you're intrested in, or
        default "misc" which I'm not. Helper loads the string using json into
        dict, getting slug method, and two different values as a tuple.
        If you set to True validate parameter you'll filter. If the first
        is not in the list category u care about, then you return default.
        Way to make sure we don't have too many dummy features later.
        """
        categories = json.loads(json_string).get("slug", "/").split("/")
        # Only keep hardcoded categories
        if validate:
            if categories[0] not in cls.gen_cats:
                categories[0] = cls.misc
            if categories[1] not in cls.precise_cats:
                categories[1] = cls.misc

        return categories

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categories = X["category"]

        return pd.DataFrame({
            "gen_cat": categories.apply(
                lambda x: self.extract_categories(x)[0]),
            "precise_cat": categories.apply(
                lambda x: self.extract_categories(x)[1])
        })


class GoalAdjustor(BaseEstimator, TransformerMixin):
    """
    Adjusts the goal feature to USD.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({"adjusted_goal": X.goal * X.static_usd_rate})


class TimeTransformer(BaseEstimator, TransformerMixin):
    """
    Builds features computed from timestamps.
    """
    def __init__(self, adj=1000_000_000):
        self.adj = adj

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Loads dates into datetime object. Multiplying the timestamp by
        constant value.
        """
        deadline = pd.to_datetime(X.deadline * self.adj)
        created = pd.to_datetime(X.created_at * self.adj)
        launched = pd.to_datetime(X.launched_at * self.adj)

        return pd.DataFrame({
            "launched_to_deadline": (deadline - launched).dt.days,
            "created_to_launched": (launched - created).dt.days
        })


class CountryTransformer(BaseEstimator, TransformerMixin):
    """
    Transform countries into larger groups to avoid having too many dummies.
    """
    countries = {
        'US': 'US',
        'CA': 'Canada',
        'GB': 'UK & Ireland',
        'AU': 'Oceania',
        'IE': 'UK & Ireland',
        'SE': 'Europe',
        'CH': "Europe",
        'IT': 'Europe',
        'FR': 'Europe',
        'NZ': 'Oceania',
        'DE': 'Europe',
        'NL': 'Europe',
        'NO': 'Europe',
        'MX': 'Other',
        'ES': 'Europe',
        'DK': 'Europe',
        'BE': 'Europe',
        'AT': 'Europe',
        'HK': 'Other',
        'SG': 'Other',
        'LU': 'Europe',
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({"country": X.country.map(
            self.countries)}).fillna("Other")
