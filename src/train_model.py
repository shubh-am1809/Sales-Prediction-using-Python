from sklearn.linear_model import LinearRegression
import joblib
import os

def train(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/sales_prediction_model.pkl")

    return model
