import joblib
import os

def predict_sales(sample_features):
    model = joblib.load("models/sales_prediction_model.pkl")
    prediction = model.predict([sample_features])[0]

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sample_prediction.txt", "w") as f:
        f.write(f"Advertising Spend (TV, Radio, Newspaper): {sample_features}\n")
        f.write(f"Predicted Sales: {prediction}\n")

    return prediction
