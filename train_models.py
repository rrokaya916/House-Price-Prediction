import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd


print("Loading data & preprocessor...")

# Load saved data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

preprocessor = joblib.load("preprocessor.pkl")


# ============================
# Models to Train
# ============================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        random_state=42
    )
}

results = {}

print("\nTraining started...\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"{name} Results:")
    print(f" MAE = {mae}")
    print(f" MSE = {mse}")
    print(f" RMSE = {rmse}")
    print(f" R² = {r2}\n")


# ============================
# Choose Best Model by R²
# ============================
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = models[best_model_name]
best_r2 = results[best_model_name]["R2"]

print("==============================")
print(f"Best Model = {best_model_name}")
print(f"R² = {best_r2}")
print("==============================")

# Save best model
joblib.dump(best_model, "best_model.pkl")
print("\nSaved best_model.pkl successfully!")
