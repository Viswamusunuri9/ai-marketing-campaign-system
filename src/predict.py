import pickle
import pandas as pd

from src.feature_engineering import create_features
from src.preprocessing import preprocess


# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model():
    model = pickle.load(open("models/model.pkl", "rb"))
    columns = pickle.load(open("models/columns.pkl", "rb"))
    return model, columns


# ----------------------------
# PREPARE INPUT
# ----------------------------
def prepare_input(input_dict, columns):

    df = pd.DataFrame([input_dict])

    df = preprocess(df)
    df = create_features(df)

    # IMPORTANT: match training features exactly
    df = df[[
        "Income", "Kidhome", "Teenhome",
        "MntWines", "MntMeatProducts",
        "AcceptedCmp1", "AcceptedCmp2",
        "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
        "Year_Birth",
        "MntFruits", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds"
    ]]

    df = pd.get_dummies(df)

    # Align with training columns
    df = df.reindex(columns=columns, fill_value=0)

    # Clean column names (same as training)
    df.columns = df.columns.astype(str)
    df.columns = df.columns.str.replace(r"[\[\]<]", "", regex=True)

    return df


# ----------------------------
# PREDICT
# ----------------------------
def predict_customer(input_dict):

    model, columns = load_model()

    df = prepare_input(input_dict, columns)

    prob = model.predict_proba(df)[0][1]

    return prob