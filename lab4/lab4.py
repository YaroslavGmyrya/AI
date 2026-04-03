import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, mean_squared_error

file_path = "dataset.data"
df = pd.read_csv(file_path)

print("Первые 5 строк:")
print(df.head())

print("\nИнформация о датасете:")
print(df.info())

print("\nРаспределение классов landmass:")
print(df["landmass"].value_counts())


flag_features = [
    "bars", "stripes", "colours",
    "red", "green", "blue", "gold", "white", "black", "orange",
    "mainhue",
    "circles", "crosses", "saltires", "quarters", "sunstars",
    "crescent", "triangle", "icon", "animate", "text",
    "topleft", "botright"
]

flag_features = [col for col in flag_features if col in df.columns]

X = df[flag_features]
y = df["landmass"]

print("\nИспользуемые признаки флага:")
print(flag_features)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("\nЧисловые признаки:", numeric_features)
print("Категориальные признаки:", categorical_features)


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
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

user_max_depth = 4
user_max_features = 8

tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=user_max_depth,
        max_features=user_max_features,
        random_state=42
    ))
])

tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

print("\n=== Модель с пользовательскими параметрами ===")
print(f"max_depth = {user_max_depth}")
print(f"max_features = {user_max_features}")
print(f"Accuracy на hold-out: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


feature_names = tree_model.named_steps["preprocessor"].get_feature_names_out()
class_names = [str(c) for c in sorted(y.unique())]

plt.figure(figsize=(28, 14))
plot_tree(
    tree_model.named_steps["classifier"],
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title(f"Decision Tree: max_depth={user_max_depth}, max_features={user_max_features}")
plt.savefig("tree_user_params.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nДерево с пользовательскими параметрами сохранено в файл: tree_user_params.png")

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

depth_values = range(1, 16)
depth_mse_scores = []

for depth in depth_values:
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=depth,
            max_features=user_max_features,
            random_state=42
        ))
    ])

    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    depth_mse_scores.append(-scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(depth_values, depth_mse_scores, marker="o")
plt.xlabel("max_depth")
plt.ylabel("Cross-validation MSE")
plt.title("Зависимость CV(MSE) от max_depth")
plt.xticks(depth_values)
plt.grid(True)
plt.show()

best_depth = depth_values[np.argmin(depth_mse_scores)]
best_depth_mse = min(depth_mse_scores)

print("\n=== Результаты по max_depth ===")
print(f"Лучший max_depth: {best_depth}")
print(f"Минимальный CV(MSE): {best_depth_mse:.4f}")


feature_values = range(1, len(flag_features) + 1)
feature_mse_scores = []

for mf in feature_values:
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=user_max_depth,
            max_features=mf,
            random_state=42
        ))
    ])

    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    feature_mse_scores.append(-scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(feature_values, feature_mse_scores, marker="o")
plt.xlabel("max_features")
plt.ylabel("Cross-validation MSE")
plt.title("Зависимость CV(MSE) от max_features")
plt.xticks(feature_values)
plt.grid(True)
plt.show()

best_max_features = feature_values[np.argmin(feature_mse_scores)]
best_features_mse = min(feature_mse_scores)

print("\n=== Результаты по max_features ===")
print(f"Лучший max_features: {best_max_features}")
print(f"Минимальный CV(MSE): {best_features_mse:.4f}")


final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=4,
        max_features=15,
        random_state=42
    ))
])

final_model.fit(X_train, y_train)
y_final_pred = final_model.predict(X_test)

final_acc = accuracy_score(y_test, y_final_pred)

print("\n=== Итоговая модель ===")
print(f"Оптимальный max_depth: {best_depth}")
print(f"Оптимальный max_features: {best_max_features}")
print(f"Accuracy итоговой модели: {final_acc:.4f}")

print("\nClassification report итоговой модели:")
print(classification_report(y_test, y_final_pred))

print("\nConfusion matrix итоговой модели:")
print(confusion_matrix(y_test, y_final_pred))

print("\nОбоснование выбора:")
print("Оптимальные max_depth и max_features выбраны по минимуму CV(MSE).")
print("Это даёт лучший баланс между сложностью дерева и качеством обобщения.")


final_tree = final_model.named_steps["classifier"]
final_feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()

plt.figure(figsize=(30, 15))
plot_tree(
    final_tree,
    feature_names=final_feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title(f"Final Decision Tree: max_depth={best_depth}, max_features={best_max_features}")
plt.savefig("tree_final.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nИтоговое дерево сохранено в файл: tree_final.png")

boundary_features = ["stripes", "colours"]

boundary_features = [col for col in boundary_features if col in df.columns]

if len(boundary_features) == 2:
    X2 = df[boundary_features]
    y2 = df["landmass"]

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2,
        test_size=0.2,
        random_state=42,
        stratify=y2
    )

    boundary_preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]), boundary_features)
        ]
    )

    boundary_model = Pipeline(steps=[
        ("preprocessor", boundary_preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=best_depth,
            max_features=2,
            random_state=42
        ))
    ])

    boundary_model.fit(X2_train, y2_train)

    x_min, x_max = X2.iloc[:, 0].min() - 1, X2.iloc[:, 0].max() + 1
    y_min, y_max = X2.iloc[:, 1].min() - 1, X2.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    grid = pd.DataFrame({
        boundary_features[0]: xx.ravel(),
        boundary_features[1]: yy.ravel()
    })

    Z = boundary_model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(
        xx, yy, Z,
        alpha=0.35,
        levels=np.arange(y2.min(), y2.max() + 2) - 0.5,
        cmap=plt.cm.Set3
    )
    scatter = plt.scatter(
        X2.iloc[:, 0],
        X2.iloc[:, 1],
        c=y2,
        edgecolor="k",
        cmap=plt.cm.Set1
    )

    plt.xlabel(boundary_features[0])
    plt.ylabel(boundary_features[1])
    plt.title("Решающие границы дерева решений по признакам флага")
    plt.colorbar(scatter, label="landmass")
    plt.show()
else:
    print("\nНе удалось построить решающие границы: нет двух подходящих признаков.")