import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

print("Starting preprocessing...")

# 1. Load dataset
df = pd.read_csv("kc_house_data.csv")

# Target column
target = "price"

# Drop ID & date (غير مفيدين للنموذج)
df = df.drop(columns=["id", "date"], errors="ignore")

# Split features/target
X = df.drop(columns=[target])
y = df[target]

# Identify numeric & categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Numeric Transformer
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical Transformer
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit transformer
preprocessor.fit(X_train)

# Transform data
X_train_trans = preprocessor.transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Save numpy arrays
np.save("X_train.npy", X_train_trans)
np.save("X_test.npy", X_test_trans)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# Save preprocessor
joblib.dump(preprocessor, "preprocessor.pkl")

# Show summary
print("\nPreprocessing completed! Saved all files.\n")
print(f"Train shape: {X_train_trans.shape}")
print(f"Test shape:  {X_test_trans.shape}")
