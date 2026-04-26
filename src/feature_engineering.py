import datetime

def create_features(df):

    # df["Age"] = 2024 - df["Year_Birth"]
    CURRENT_YEAR = datetime.datetime.now().year
    df["Age"] = CURRENT_YEAR - df["Year_Birth"]

    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    df["Total_Spending"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )

    return df