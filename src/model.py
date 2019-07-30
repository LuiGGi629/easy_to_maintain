import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from src.transformers import (CategoriesExtractor, CountryTransformer,
                              GoalAdjustor, TimeTransformer)
from src.config import (DTYPES, DATA_DIR, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST,
                        MODEL_NAME, GRID_PARAMS, PARAMS)


def load_dataset(x_path, y_path):
    x = pd.read_csv(os.sep.join([DATA_DIR, x_path]),
                    dtype=DTYPES,
                    index_col="id")
    y = pd.read_csv(os.sep.join([DATA_DIR, y_path]))

    return x, y


def build_model():
    cat_processor = Pipeline([("transformer", CategoriesExtractor()),
                              ("one_hot",
                               OneHotEncoder(sparse=False,
                                             handle_unknown="ignore"))])
    country_processor = Pipeline([("transformer", CountryTransformer()),
                                  ("one_hot",
                                   OneHotEncoder(sparse=False,
                                                 handle_unknown="ignore"))])
    # The main column transformer
    preprocessor = ColumnTransformer([
        ("goal", GoalAdjustor(), ["goal", "static_usd_rate"]),
        ("categories", cat_processor, ["category"]),
        ("disable_communication", "passthrough", ["disable_communication"]),
        ("time", TimeTransformer(), ["deadline", "created_at", "launched_at"]),
        ("countries", country_processor, ["country"])
    ])

    model = Pipeline([("preprocessor", preprocessor),
                      ("model", DecisionTreeClassifier())])

    return model


def tune_model():
    X_train, y_train = load_dataset(X_TRAIN, Y_TRAIN)
    model = build_model()

    gs = GridSearchCV(model, GRID_PARAMS, scoring="accuracy", n_jobs=-1, cv=5)
    gs.fit(X_train, y_train)

    print(f"Best Hyperparameters: {gs.best_params_}")
    print("Best score: {:.2f}%".format(100 * gs.best_score_))


def train_model(print_params=False):
    X_train, y_train = load_dataset(X_TRAIN, Y_TRAIN)

    model = build_model()
    model.set_params(**PARAMS)

    if print_params:
        print(model.get_params())

    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_NAME)


def test_model():
    X_test, y_test = load_dataset(X_TEST, Y_TEST)

    model = joblib.load(MODEL_NAME)

    y_pred = model.predict(X_test)

    print("Accuracy on test set: {:.2f}%".format(
        100 * accuracy_score(y_test, y_pred)))
    print(classification_report(y_test, y_pred))
