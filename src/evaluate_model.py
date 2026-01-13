from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_metrics.txt", "w") as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R2 Score: {r2}\n")

    print("Evaluation metrics saved to outputs/evaluation_metrics.txt")
