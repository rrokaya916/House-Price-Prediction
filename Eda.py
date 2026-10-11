import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create folder for saving plots if not exists
os.makedirs("eda_plots", exist_ok=True)

def save_plot(name):
    plt.savefig(f"eda_plots/{name}.png", dpi=300, bbox_inches='tight')
    plt.close()


# 1️⃣ Load Dataset
df = pd.read_csv("kc_house_data.csv")
print("Dataset Loaded Successfully!\n")

# 2️⃣ Remove Non-Numeric & Irrelevant Columns
df = df.drop(columns=["id", "date"], errors="ignore")

print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe().T)
print("\nMissing Values:")
print(df.isnull().sum())


# 3️⃣ Price Distribution
sns.histplot(df['price'], kde=True)
plt.title("Price Distribution")
save_plot("price_distribution")

sns.histplot(np.log1p(df['price']), kde=True)
plt.title("Log Transformed Price Distribution")
save_plot("log_price_distribution")


# 4️⃣ Correlation (Only Numeric Columns)
numeric_df = df.select_dtypes(include=['int64', 'float64'])

corr = numeric_df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
save_plot("correlation_heatmap")

print("\nTop Correlations with Price:")
print(corr['price'].sort_values(ascending=False).head(10))


# 5️⃣ Important Numeric Feature Relationships
important_features = ['sqft_living', 'grade', 'bathrooms', 'sqft_above']
for col in important_features:
    sns.regplot(x=df[col], y=df['price'])
    plt.title(f"{col} vs Price")
    save_plot(f"{col}_vs_price")


# 6️⃣ Categorical Feature Impact
sns.boxplot(x='waterfront', y='price', data=df)
plt.title("Waterfront vs Price")
save_plot("waterfront_vs_price")

sns.boxplot(x='view', y='price', data=df)
plt.title("View vs Price")
save_plot("view_vs_price")

plt.figure(figsize=(12,5))
sns.boxplot(x='grade', y='price', data=df)
plt.title("Grade vs Price")
save_plot("grade_vs_price")


# 7️⃣ Outliers Detection
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['price'] < Q1 - 1.5 * IQR) | (df['price'] > Q3 + 1.5 * IQR)]

print(f"\nOutliers Count: {len(outliers)}")
print(f"Outliers Percentage: {(len(outliers) / len(df)) * 100:.2f}%")

sns.boxplot(df['price'])
plt.title("Price Outliers")
save_plot("price_outliers")


# 8️⃣ Price by Zipcode
df.groupby("zipcode")['price'].mean().sort_values().plot(kind="bar", figsize=(15,6))
plt.title("Average Price by Zipcode")
plt.ylabel("Price")
save_plot("zipcode_average_price")


# 9️⃣ Location Map (Lat-Long)
plt.figure()
sns.scatterplot(data=df, x='long', y='lat', hue='price', palette="viridis", s=30)
plt.title("Location & Price Distribution")
plt.legend([],[], frameon=False)
save_plot("location_price_map")


print("\n🎯 All EDA Completed Successfully!")
print("📁 Plots saved inside folder: eda_plots/\n")
