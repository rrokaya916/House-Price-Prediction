import numpy as np
import pandas as pd
import joblib
import time

# ========================================
# Load Preprocessor & Trained Model
# ========================================
print("Loading preprocessor and trained model...\n")
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("best_model.pkl")


# ========================================
# Predict Function
# ========================================
def predict_price(input_data):
    """
    input_data: pandas DataFrame with the same columns as training data (except price)
    """
    transformed = preprocessor.transform(input_data)
    prediction = model.predict(transformed)
    return prediction


# ========================================
# Predict From CSV (formatted output)
# ========================================
def predict_from_csv(csv_path):
    print(f"Reading file: {csv_path}")

    start_time = time.time()
    df = pd.read_csv(csv_path)

    # If price exists → drop it
    if "price" in df.columns:
        df = df.drop(columns=["price"])

    # Run prediction
    preds = predict_price(df)

    # Add predictions formatted
    df_result = df.copy()
    df_result["predicted_price"] = preds
    df_result["predicted_price_formatted"] = df_result["predicted_price"].apply(lambda x: f"{x:,.2f} USD")

    # Save output
    output_path = "predictions_output.csv"
    df_result.to_csv(output_path, index=False)

    # ============================
    # Console Summary
    # ============================
    print("\n====================================")
    print("        Prediction Summary")
    print("====================================")
    print(f"Rows processed: {len(df)}")
    print(f"Execution time: {time.time() - start_time:.2f} sec")

    print("\nSample predicted prices:")
    for p in preds[:10]:
        print(f"{p:,.2f} USD")

    print("\nPreview of output file (first 5 rows):")
    print(df_result.head())

    print(f"\nFile saved as: {output_path}")
    print("====================================\n")


# ========================================
# Manual Example Mode
# ========================================
def manual_example():
    print("\nRunning manual example...")

    sample = {
        "bedrooms": [3],
        "bathrooms": [2],
        "sqft_living": [1800],
        "sqft_lot": [5000],
        "floors": [1],
        "waterfront": [0],
        "view": [0],
        "condition": [3],
        "grade": [7],
        "sqft_above": [1500],
        "sqft_basement": [300],
        "yr_built": [1990],
        "yr_renovated": [0],
        "zipcode": ["98103"],
        "lat": [47.65],
        "long": [-122.34],
        "sqft_living15": [1800],
        "sqft_lot15": [4000]
    }

    df = pd.DataFrame(sample)
    pred = predict_price(df)

    print(f"\nPredicted Price = {pred[0]:,.2f} USD\n")


# ========================================
# Main Program
# ========================================
if __name__ == "__main__":
    print("Choose prediction mode:")
    print("1 - Predict from CSV file")
    print("2 - Manual prediction example")

    choice = input("Enter choice (1/2): ")

    if choice == "1":
        path = input("Enter CSV file path: ")
        predict_from_csv(path)

    elif choice == "2":
        manual_example()

    else:
        print("Invalid choice!")
