from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train_model import train
from src.evaluate_model import evaluate
from src.predict import predict_sales
from sklearn.model_selection import train_test_split

def main():
    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)

    # Sample prediction using first row
    sample = X.iloc[0].tolist()
    result = predict_sales(sample)

    print("Predicted Sales:", result)
    print("All outputs saved in outputs/ folder")

if __name__ == "__main__":
    main()
