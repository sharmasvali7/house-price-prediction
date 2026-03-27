import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("homeprices.csv")
print("=== Raw Data ===")
print(df)

# ── 2. Handle Missing Values ──────────────────────────────────────────────────
# Fill missing 'bedrooms' with the median (row 3 in your data has NaN)
df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
print("\n=== Data after filling missing values ===")
print(df)

# ── 3. Define Features (X) and Target (y) ─────────────────────────────────────
X = df[["area", "bedrooms", "age"]]   # input features
y = df["price"]                        # target variable

# ── 4. Split into Train / Test sets ──────────────────────────────────────────
# Using 80% for training, 20% for testing
# (with only 6 rows, we keep test_size small)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── 5. Train the Model ────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print("\n=== Model Coefficients ===")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature:10s}: {coef:,.2f}")
print(f"  {'Intercept':10s}: {model.intercept_:,.2f}")

# ── 6. Make Predictions ───────────────────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n=== Predictions vs Actual ===")
results = pd.DataFrame({
    "Actual Price ($)":    y_test.values,
    "Predicted Price ($)": y_pred.round(2)
})
print(results.to_string(index=False))

# ── 7. Evaluate the Model ─────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n=== Model Performance ===")
print(f"  MAE  (Mean Absolute Error)       : ${mae:,.2f}")
print(f"  RMSE (Root Mean Squared Error)   : ${rmse:,.2f}")
print(f"  R²   (Coefficient of Determination): {r2:.4f}")

# ── 8. Predict on New / Unseen Houses ─────────────────────────────────────────
new_houses = pd.DataFrame({
    "area":     [2800, 3500],
    "bedrooms": [3,    4   ],
    "age":      [10,   5   ]
})

predictions = model.predict(new_houses)
print("\n=== Predicting Prices for New Houses ===")
for i, (_, row) in enumerate(new_houses.iterrows()):
    print(f"  House {i+1}: area={int(row.area)} sqft, "
          f"bedrooms={int(row.bedrooms)}, age={int(row.age)} yrs "
          f"→ Predicted Price: ${predictions[i]:,.2f}")
