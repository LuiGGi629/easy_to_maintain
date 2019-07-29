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

    def __init__(self, use_all=False):
        """
        Defines a hyperparameter use_all. That is a trick, that allows to
        decide whether you want to extract all categories or only hard coded
        initially e.g.gen_cats, precise cats -> which you're intrested in, or
        default "misc" which I'm not.
        """
        self.use_all = use_all

    def _get_slug(self, x):
        """
        Helper loads the string using json into dict, getting slug method,
        and two different values as a tuple. If you're not specyfing
        use_all parameter you'll filter. If the first is not in the list
        category u care about, then you return default. Way to make sure
        we don't have too many dummy features later.
        """
        categories = json.loads(x).get("slug", "/").split("/")
        # Only keep hardcoded categories
        if not self.use_all:
            if categories[0] not in self.gen_cats:
                categories[0] = self.misc
            if categories[1] not in self.precise_cats:
                categories[1] = self.misc

        return categories

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categories = X["category"]

        return pd.DataFrame({
            "gen_cat": categories.apply(lambda x: self._get_slug(x)[0]),
            "precise_cat": categories.apply(lambda x: self._get_slug(x)[1])
            })


class GoalAdjustor():
    """
    Adjusts the goal feature to USD.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({"adjusted_goal": X.goal * X.static_usd_rate})


class TimeTransformer():
    """
    Builds features computed from timestamps.
    """

    def fit(self):
        pass

    def transform(self):
        pass


class CountryTransformer():
    """
    Transform countries into larger groups to avoid having too many dummies.
    """

    def fit(self):
        pass

    def transform(self):
        pass