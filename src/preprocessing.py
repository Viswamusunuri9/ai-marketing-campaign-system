import pandas as pd

def load_data(path):
    return pd.read_csv(path)


def preprocess(df):

    df = df.copy()

    drop_cols = ["ID", "Dt_Customer"]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Basic null handling
    df = df.fillna(0)

    return df