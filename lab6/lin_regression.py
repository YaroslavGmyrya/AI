import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("machine.data")

df = df.drop(columns=["vendor", "model"])

X = df.drop(columns=["PRP"])
y = df["PRP"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.title("Real and predicted (PRP)")
plt.xlabel("Real PRP")
plt.ylabel("Predicted PRP")
plt.grid()
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=25, edgecolor="black")
plt.title("Distribution of Residuals")
plt.xlabel("Error")
plt.ylabel("Count")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("residuals and predicted")
plt.xlabel("Predicted PRP")
plt.ylabel("Residuals")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(X.columns, model.coef_)
plt.title("Linear regression coefficients")
plt.xticks(rotation=45)
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
corr = df.corr()
plt.imshow(corr, cmap="coolwarm", interpolation="none")
plt.colorbar()
plt.title("Correlation matrix")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()

feature = "MMAX" 

X_feature = df[[feature]]
y_feature = df["PRP"]

model_1d = LinearRegression()
model_1d.fit(X_feature, y_feature)

x_vals = np.linspace(X_feature.min(), X_feature.max(), 100)
y_vals = model_1d.predict(x_vals)

plt.figure(figsize=(8, 5))
plt.scatter(X_feature, y_feature, alpha=0.5, label="Data")
plt.plot(x_vals, y_vals, color="red", linewidth=2, label="Regression line")

plt.title(f"Regression line(for PRP and {feature}")
plt.xlabel(feature)
plt.ylabel("PRP")
plt.legend()
plt.grid()
plt.show()