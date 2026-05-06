import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

FILE_PATH = 'dataset.data'  
TARGET_COLUMN = "landmass" 
TEST_SIZE = 0.2 # test size in %
RANDOM_STATE = 42   # random number for spliting on test and teach sequence

AUTO_DETECT_TARGET = True

MISSING_STRATEGY = "median" # drop_rows, median, meam



def safe_read_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    separators = [',', ';', '\t', r'\s+']
    last_error = None

    for sep in separators:
        try:
            df = pd.read_csv(file_path, sep=sep, engine='python')
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_error = e

    raise ValueError(f"Не удалось корректно прочитать файл {file_path}. Последняя ошибка: {last_error}")


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r'\s+', '_', regex=True)
    )
    return df


def try_convert_object_to_numeric(df, threshold=0.8):
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == 'object':
            converted = pd.to_numeric(df[col], errors='coerce')
            non_null_ratio = converted.notna().mean()

            if non_null_ratio >= threshold:
                df[col] = converted

    return df


def detect_target_column(df):
    common_targets = [
        'target', 'label', 'class', 'y', 'output', 'result',
        'survived', 'species', 'diagnosis', 'price'
    ]

    lower_map = {col.lower(): col for col in df.columns}

    for candidate in common_targets:
        if candidate in lower_map:
            return lower_map[candidate]

    candidate_cols = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 20:
            candidate_cols.append(col)

    if candidate_cols:
        return candidate_cols[-1]

    return df.columns[-1]


def choose_task_type(y):
    nunique = y.nunique(dropna=True)
    if y.dtype == 'object' or nunique <= 20:
        return 'classification'
    return 'regression'


def safe_plot_histograms(df):
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        print("\nНет числовых столбцов для гистограмм.")
        return

    numeric_df.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.show()


def safe_plot_missing_heatmap(df):
    if df.empty:
        print("\nПустой DataFrame. Тепловая карта пропусков не построена.")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Карта пропущенных значений')
    plt.tight_layout()
    plt.show()


def safe_plot_target(df, target_col):
    if target_col is None or target_col not in df.columns:
        print("\nЦелевой столбец не найден. График целевой переменной пропущен.")
        return

    series = df[target_col]
    nunique = series.nunique(dropna=True)

    plt.figure(figsize=(8, 5))

    if series.dtype == 'object' or nunique <= 20:
        sns.countplot(x=target_col, data=df)
        plt.title(f'Распределение целевой переменной: {target_col}')
        plt.xticks(rotation=45)
    else:
        sns.histplot(series.dropna(), kde=True)
        plt.title(f'Распределение целевой переменной: {target_col}')

    plt.tight_layout()
    plt.show()


def validate_missing_strategy(strategy):
    allowed = {"drop_rows", "mean", "median", "most_frequent"}
    if strategy not in allowed:
        raise ValueError(
            f"Недопустимая стратегия заполнения пропусков: {strategy}. "
            f"Доступные варианты: {sorted(allowed)}"
        )


def drop_missing_rows(X, y):
    combined = pd.concat([X, y], axis=1)
    before_shape = combined.shape
    combined = combined.dropna()
    after_shape = combined.shape

    X_clean = combined.iloc[:, :-1]
    y_clean = combined.iloc[:, -1]

    print(f"\nУдаление строк с пропусками:")
    print(f"Было строк: {before_shape[0]}")
    print(f"Стало строк: {after_shape[0]}")
    print(f"Удалено строк: {before_shape[0] - after_shape[0]}")

    return X_clean, y_clean


def build_preprocessor(numeric_features, categorical_features, missing_strategy):
    transformers = []

    if missing_strategy == "mean":
        numeric_imputer_strategy = "mean"
        categorical_imputer_strategy = "most_frequent"
    elif missing_strategy == "median":
        numeric_imputer_strategy = "median"
        categorical_imputer_strategy = "most_frequent"
    elif missing_strategy == "most_frequent":
        numeric_imputer_strategy = "most_frequent"
        categorical_imputer_strategy = "most_frequent"
    else:
        numeric_imputer_strategy = None
        categorical_imputer_strategy = None

    if numeric_features:
        numeric_steps = []
        if numeric_imputer_strategy is not None:
            numeric_steps.append(('imputer', SimpleImputer(strategy=numeric_imputer_strategy)))
        numeric_steps.append(('scaler', StandardScaler()))

        numeric_transformer = Pipeline(steps=numeric_steps)
        transformers.append(('num', numeric_transformer, numeric_features))

    if categorical_features:
        categorical_steps = []
        if categorical_imputer_strategy is not None:
            categorical_steps.append(('imputer', SimpleImputer(strategy=categorical_imputer_strategy)))
        categorical_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))

        categorical_transformer = Pipeline(steps=categorical_steps)
        transformers.append(('cat', categorical_transformer, categorical_features))

    return ColumnTransformer(transformers=transformers)


try:
    validate_missing_strategy(MISSING_STRATEGY)
    df = safe_read_csv(FILE_PATH)
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    raise SystemExit(1)

df = clean_columns(df)
df = try_convert_object_to_numeric(df)

print("Первые 5 строк:")
print(df.head())

print("\nРазмер датасета:")
print(df.shape)

print("\nНазвания столбцов:")
print(df.columns.tolist())

print("\nТипы данных:")
print(df.dtypes)

print("\nОбщая информация:")
df.info()

print("\nСтатистика числовых признаков:")
print(df.describe())

print("\nКоличество пропусков:")
print(df.isnull().sum())


safe_plot_histograms(df)
safe_plot_missing_heatmap(df)


target_col = TARGET_COLUMN

if target_col is None and AUTO_DETECT_TARGET:
    target_col = detect_target_column(df)
    print(f"\nАвтоматически выбран целевой столбец: {target_col}")

if target_col not in df.columns:
    print("\nНе удалось определить целевой столбец.")
    print("Укажите TARGET_COLUMN вручную.")
    raise SystemExit(1)

safe_plot_target(df, target_col)


X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

if X.shape[1] == 0:
    print("После удаления целевой переменной не осталось признаков.")
    raise SystemExit(1)

print(f"\nЦелевая переменная: {target_col}")

task_type = choose_task_type(y)
print(f"Предполагаемый тип задачи: {task_type}")


if MISSING_STRATEGY == "drop_rows":
    X, y = drop_missing_rows(X, y)

    if X.empty or len(y) == 0:
        print("После удаления строк с пропусками данные закончились.")
        raise SystemExit(1)


numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nЧисловые признаки:")
print(numeric_features)

print("\nКатегориальные признаки:")
print(categorical_features)

if len(numeric_features) == 0 and len(categorical_features) == 0:
    print("Не удалось определить признаки для обработки.")
    raise SystemExit(1)


preprocessor = build_preprocessor(
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    missing_strategy=MISSING_STRATEGY
)


stratify_value = None
if task_type == 'classification':
    class_counts = y.value_counts(dropna=False)
    if len(class_counts) > 1 and class_counts.min() >= 2:
        stratify_value = y

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_value
    )
except Exception as e:
    print(f"\nОшибка при разделении данных: {e}")
    raise SystemExit(1)


try:
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
except Exception as e:
    print(f"\nОшибка при предварительной обработке: {e}")
    raise SystemExit(1)

print("\nВыбранная стратегия обработки пропусков:", MISSING_STRATEGY)
print("Размер обучающей выборки до обработки:", X_train.shape)
print("Размер тестовой выборки до обработки:", X_test.shape)
print("Размер обучающей выборки после обработки:", X_train_processed.shape)
print("Размер тестовой выборки после обработки:", X_test_processed.shape)

print("\nПредварительная обработка успешно завершена.")