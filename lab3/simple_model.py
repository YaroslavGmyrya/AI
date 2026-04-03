import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = "dataset.data"

df = pd.read_csv(file_path)

print("Первые 5 строк:")
print(df.head())

print("\nИнформация о датасете:")
print(df.info())

print("\nРаспределение классов landmass:")
print(df["landmass"].value_counts())

pairplot_features = [
    "area", "population", "bars", "stripes", "colours", "sunstars"
]

pairplot_features = [col for col in pairplot_features if col in df.columns]

sns.pairplot(
    df[pairplot_features + ["landmass"]],
    hue="landmass",
    diag_kind="hist"
)
plt.suptitle("Pairplot признаков по классам landmass", y=1.02)
plt.show()

y = df["landmass"]

X = df.drop(columns=["landmass", "country"], errors="ignore")

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nЧисловые признаки:", numeric_features)
print("Категориальные признаки:", categorical_features)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

knn_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

holdout_acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy на hold-out: {holdout_acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

k_values = range(1, 21)
train_scores = []
test_scores = []

for k in k_values:
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_scores.append(accuracy_score(y_train, y_train_pred))
    test_scores.append(accuracy_score(y_test, y_test_pred))

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_scores, marker="o", label="Train accuracy")
plt.plot(k_values, test_scores, marker="o", label="Test accuracy")
plt.xlabel("Количество соседей (k)")
plt.ylabel("Accuracy")
plt.title("Зависимость качества модели от числа соседей k")
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

best_k = k_values[np.argmax(test_scores)]
best_test_score = max(test_scores)

print(f"\nЛучшее k по hold-out: {best_k}")
print(f"Лучшее качество на тесте: {best_test_score:.4f}")

cv_mean_scores = []
cv_error_scores = []

for k in k_values:
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])
    
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    cv_mean_scores.append(scores.mean())
    cv_error_scores.append(1 - scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_error_scores, marker="o")
plt.xlabel("Количество соседей (k)")
plt.ylabel("CV error = 1 - accuracy")
plt.title("Ошибка cross-validation в зависимости от k")
plt.xticks(k_values)
plt.grid(True)
plt.show()

best_cv_k = k_values[np.argmin(cv_error_scores)]
best_cv_error = min(cv_error_scores)

print(f"\nЛучшее k по cross-validation: {best_cv_k}")
print(f"Минимальная CV error: {best_cv_error:.4f}")


final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=2))
])

final_model.fit(X_train, y_train)
y_final_pred = final_model.predict(X_test)

final_acc = accuracy_score(y_test, y_final_pred)

print("\n=== Итоговая модель с лучшим k ===")
print(f"Лучшее k: {best_k}")
print(f"Accuracy итоговой модели: {final_acc:.4f}")

print("\nClassification report итоговой модели:")
print(classification_report(y_test, y_final_pred))

print("\nConfusion matrix итоговой модели:")
print(confusion_matrix(y_test, y_final_pred))