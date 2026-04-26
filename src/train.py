import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.preprocessing import load_data, preprocess
from src.feature_engineering import create_features


# ----------------------------
# FEATURE SELECTION (🔥 CRITICAL FIX)
# ----------------------------
SELECTED_FEATURES = [
    # Core profile
    "Income", "Kidhome", "Teenhome", "Year_Birth",

    # Spending
    "MntWines", "MntMeatProducts",
    "MntFruits", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds",

    # Behavior (🔥 IMPORTANT ADD)
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",

    # Campaign history (🔥 STRONG SIGNAL)
    "AcceptedCmp1", "AcceptedCmp2",
    "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"
]

# ----------------------------
# PREPARE DATA
# ----------------------------
def prepare_data(df):

    df = df[SELECTED_FEATURES + ["Response"]]

    y = df["Response"]
    X = df.drop(columns=["Response"])

    X = pd.get_dummies(X, drop_first=True)

    # Clean column names (XGBoost safe)
    X.columns = X.columns.astype(str)
    X.columns = X.columns.str.replace(r"[\[\]<]", "", regex=True)

    return X, y


# ----------------------------
# TRAIN MODELS
# ----------------------------
def train_models(X_train, X_test, y_train, y_test):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=42, verbose=0
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "auc": auc
        }

        print(f"{name} → Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}")

    return results


# ----------------------------
# MAIN TRAINING
# ----------------------------
def train():

    print("Loading data...")
    df = load_data("data/marketing_data_cleaned.csv")

    print("Preprocessing...")
    df = preprocess(df)

    print("Feature engineering...")
    df = create_features(df)

    print("Preparing data...")
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n--- Training Models ---")
    results = train_models(X_train, X_test, y_train, y_test)

    # Select best model
    best_model_name = max(results, key=lambda x: results[x]["auc"])
    best_model = results[best_model_name]["model"]

    print(f"\nBest Model: {best_model_name}")

    # Save artifacts
    pickle.dump(best_model, open("models/model.pkl", "wb"))
    pickle.dump(X.columns, open("models/columns.pkl", "wb"))

    print("Model saved successfully!")


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    train()