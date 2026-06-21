import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("Housing_Price.csv")

X = df.drop("price", axis=1)
y = df["price"]

for col in ["mainroad", "guestroom", "basement",
            "hotwaterheating", "airconditioning", "prefarea"]:
    X[col] = X[col].map({"yes": 1, "no": 0})

X["furnishingstatus"] = X["furnishingstatus"].map({
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = ExtraTreesRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("--------------------")
print("MAE:", mae)
print("R2 Score:", r2)

print("\nSample Predictions:")
for i in range(5):
    print(f"Predicted: {y_pred[i]:,.0f} | Actual: {y_test.values[i]:,.0f}")

os.makedirs("outputs", exist_ok=True)

# =========================
# 1st IMAGE - FEATURE IMPORTANCE
# =========================
plt.figure(figsize=(10, 5))
plt.barh(X.columns, model.feature_importances_)
plt.title("Feature Importance")

plt.savefig("outputs/feature_importance.png", bbox_inches="tight")
plt.show()
plt.close()

# =========================
# 2nd IMAGE - ACTUAL VS PREDICTED
# =========================
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.title("Actual vs Predicted Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.savefig("outputs/actual_vs_predicted.png", bbox_inches="tight")
plt.show()
plt.close()