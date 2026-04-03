import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# read
df = pd.read_csv("dataset.data")

# skip values
print("\nКоличество пропущенных значений:")
print(df.isna().sum())

TARGET_COL = "landmass"

# used features (set manually)
flag_features = [
    "bars",
    "stripes",
    "colours",
    "red",
    "green",
    "blue",
    "gold",
    "white",
    "black",
    "orange",
    "mainhue",
    "circles",
    "crosses",
    "saltires",
    "quarters",
    "sunstars",
    "crescent",
    "triangle",
    "icon",
    "animate",
    "text",
    "topleft",
    "botright"
]

X = df[flag_features].copy()
y = df[TARGET_COL].copy()

print("\nИспользуемые признаки флага:")
print(flag_features)

missing_counts = X.isna().sum()
missing_nonzero = missing_counts[missing_counts > 0].sort_values(ascending=False)

if not missing_nonzero.empty:
    plt.figure(figsize=(12, 5))
    missing_nonzero.plot(kind="bar")
    plt.title("Количество пропущенных значений в признаках флага")
    plt.ylabel("Число пропусков")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("\nПропущенных значений среди признаков флага нет.")

plt.figure(figsize=(8, 5))
pd.Series(y).astype(str).value_counts().sort_index().plot(kind="bar")
plt.title("Распределение классов landmass")
plt.ylabel("Количество объектов")
plt.xlabel("Код континента")
plt.tight_layout()
plt.show()

numeric_preview = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
if len(numeric_preview) > 0:
    X[numeric_preview].hist(figsize=(14, 8), bins=15)
    plt.suptitle("Распределения числовых признаков флага", y=1.02)
    plt.tight_layout()
    plt.show()


categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nКатегориальные признаки:")
print(categorical_features)

print("\nЧисловые признаки:")
print(numeric_features)

label_encoder = None

if y.dtype == "object":
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y.astype(str))
else:
    y = y.values

transformers = []

if len(numeric_features) > 0:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    transformers.append(("num", numeric_transformer, numeric_features))

if len(categorical_features) > 0:
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers.append(("cat", categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers)

unique_classes = np.unique(y)
stratify_value = y if len(unique_classes) > 1 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_value
)

print("\nРазмер обучающей выборки:", X_train.shape)
print("Размер тестовой выборки:", X_test.shape)


model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

model.fit(X_train, y_train)



y_pred = model.predict(X_test)


acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(acc, 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))


cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
plt.title("Матрица ошибок для предсказания landmass")
plt.tight_layout()
plt.show()


try:
    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_classifier = model.named_steps["classifier"]

    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Признак": feature_names,
        "Важность": importances
    }).sort_values(by="Важность", ascending=False)

    print("\nТоп-20 самых важных признаков:")
    print(importance_df.head(20))

    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20).sort_values(by="Важность")
    plt.barh(top_features["Признак"], top_features["Важность"])
    plt.title("Топ-20 важных признаков флага")
    plt.xlabel("Важность")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("\nНе удалось построить важность признаков.")
    print("Причина:", e)


try:
    X_train_transformed = model.named_steps["preprocessor"].fit_transform(X_train)

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    transformed_feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    X_train_transformed_df = pd.DataFrame(
        X_train_transformed,
        columns=transformed_feature_names
    )

    print("\nПервые 5 строк обработанных признаков:")
    print(X_train_transformed_df.head())

except Exception as e:
    print("\nНе удалось вывести обработанные данные.")
    print("Причина:", e)