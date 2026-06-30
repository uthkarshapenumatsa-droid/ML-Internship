import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# HOUSE PRICE PREDICTION USING LINEAR REGRESSION
# ==========================================

print("=" * 50)
print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
print("=" * 50)

# Load Dataset
df = pd.read_csv("train.csv")

print("\nDataset Shape:", df.shape)

# Create Total Bathrooms Feature
df["Bathrooms"] = df["FullBath"] + (0.5 * df["HalfBath"])

# Select Features and Target
X = df[["GrLivArea", "BedroomAbvGr", "Bathrooms"]]
y = df["SalePrice"]

print("\nSelected Features:")
print(X.head())

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nMODEL PERFORMANCE")
print("-" * 30)
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")

# Predict New House Price
new_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "Bathrooms": [2]
})

predicted_price = model.predict(new_house)

print("\nNEW HOUSE PREDICTION")
print("-" * 30)
print("Area      : 2000 sq.ft")
print("Bedrooms  : 3")
print("Bathrooms : 2")
print(f"\nPredicted House Price = ${predicted_price[0]:.2f}")

# Feature Importance
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance")
print(coefficients)

# ==========================================
# GRAPH 1 - Actual vs Predicted
# ==========================================

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.grid(True)
plt.savefig("actual_vs_predicted.png")
plt.show()

# ==========================================
# GRAPH 2 - Feature Importance
# ==========================================

plt.figure(figsize=(8,6))
plt.bar(coefficients["Feature"], coefficients["Coefficient"])
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.grid(True)
plt.savefig("feature_importance.png")
plt.show()

# ==========================================
# GRAPH 3 - Residual Error Plot
# ==========================================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.title("Residual Error Plot")
plt.xlabel("Predicted Price")
plt.ylabel("Residual Error")
plt.grid(True)
plt.savefig("residual_error_plot.png")
plt.show()

print("\nAll graphs have been saved successfully.")
print("Task 1 Completed Successfully!")
