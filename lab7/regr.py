import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import statsmodels.api as sm

def backward_elimination(X, y, sl=0.05):
    X = sm.add_constant(X)
    features = list(X.columns)

    while True:
        X_opt = X[features]
        model = sm.OLS(y, X_opt).fit()

        p_values = model.pvalues
        max_p = p_values.max()

        if max_p > sl:
            excluded_feature = p_values.idxmax()
            print(f"Удаляем: {excluded_feature} (p-value={max_p})")
            features.remove(excluded_feature)
        else:
            break

    return X[features], model


df = pd.read_csv("machine.data")

df = df.drop(columns=["vendor", "model"])

X = df.drop(columns=["PRP"])
y = df["PRP"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_opt_train, sm_model = backward_elimination(X_train, y_train)

print(sm_model.summary())


features_final = X_opt_train.columns

model = LinearRegression()
model.fit(X_train[features_final], y_train)

y_pred = model.predict(X_test[features_final])


mae = mean_absolute_error(y_test, y_pred) / len(y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)


plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.title("Real vs Predicted (PRP)")
plt.xlabel("Real PRP")
plt.ylabel("Predicted PRP")
plt.grid()
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=25, edgecolor="black")
plt.title("Residuals distribution")
plt.grid()
plt.show()


plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals vs Predicted")
plt.grid()
plt.show()


plt.figure(figsize=(10, 5))
plt.bar(features_final, model.coef_)
plt.title("Regression coefficients (after selection)")
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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X['CHMAX'], X['ERP'], y)

ax.set_xlabel('CHMAX')
ax.set_ylabel('ERP')
ax.set_zlabel('PRP')

plt.show()


# Берём только 2 признака
X_plot = X_train[features_final]

x1 = X_plot.iloc[:, 0]
x2 = X_plot.iloc[:, 1]
y_plot = y_train

# Создаём сетку
x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1.min(), x1.max(), 50),
    np.linspace(x2.min(), x2.max(), 50)
)

# Предсказания для сетки
y_grid = model.predict(
    np.c_[x1_grid.ravel(), x2_grid.ravel()]
).reshape(x1_grid.shape)

# График
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# точки
ax.scatter(x1, x2, y_plot, color='blue', alpha=0.5)

# плоскость
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.3)

ax.set_xlabel(features_final[0])
ax.set_ylabel(features_final[1])
ax.set_zlabel("PRP")

plt.title("Regression plane")
plt.show()