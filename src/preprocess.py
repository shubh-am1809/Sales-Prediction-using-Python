def preprocess_data(df):
    X = df.drop("Sales", axis=1)
    y = df["Sales"]
    return X, y
