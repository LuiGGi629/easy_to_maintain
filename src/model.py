import os
import pandas as pd
from config import DTYPES, DATA_DIR


def load_dataset(x_path, y_path):
    x = pd.read_csv(os.sep.join([DATA_DIR, x_path]),
                    dtype=DTYPES,
                    index_col="id")
    y = pd.read_csv(os.sep.join([DATA_DIR, y_path]))

    return x, y
