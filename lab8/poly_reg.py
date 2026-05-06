import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("machine.data")

X = df[['MMAX']].values
y = df['PRP'].values

for degree in range(2, 6):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)

    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    print(f"N={degree}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print("\n\n")

    sort_idx = np.argsort(X.flatten())
    X_sorted = X.flatten()[sort_idx]
    y_sorted = y_pred[sort_idx]

    plt.figure()
    plt.scatter(X, y)
    plt.plot(X_sorted, y_sorted)
    plt.xlabel("MYCT")
    plt.ylabel("PRP")
    plt.title(f"Regression Line (N = {degree})")
    plt.savefig(f"poly_reg_N={degree}.png")

