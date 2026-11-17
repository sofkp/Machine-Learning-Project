"""Preprocesamiento para Telco Customer Churn (UTEC, P2).

Los únicos faltantes observados están en
TotalCharges y se sustituyen por cero. No hay imputación por mediana ni moda.

Funciones públicas:
    load_data(path=DATA_URL) -> (X, y)
    split_data(X, y) -> (X_train, X_val, X_test, y_train, y_val, y_test)
    make_preprocessor('logistic' | 'tree') -> ColumnTransformer sin ajustar

Nota: integrar el preprocesador dentro del Pipeline de cada clasificador y
ajustarlo solo con X_train. No ajustar con Validation ni Test.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/sofkp/Machine-Learning-Project/"
    "main/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
RANDOM_STATE = 42

# Las 19 características originales: 3 cuantitativas y 16 categóricas.
NUMERIC_FEATURES = ("tenure", "MonthlyCharges")
TOTALCHARGES_FEATURE = ("TotalCharges",)
CATEGORICAL_FEATURES = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
)
ALL_FEATURES = NUMERIC_FEATURES + TOTALCHARGES_FEATURE + CATEGORICAL_FEATURES


def load_data(path=DATA_URL):
    """Carga el CSV, convierte TotalCharges y codifica Churn como 0/1."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # La imputación cero está justificada únicamente en este patrón observado.
    assert (df.loc[df["TotalCharges"].isna(), "tenure"] == 0).all(), (
        "Hay TotalCharges faltantes con tenure distinto de cero."
    )

    y = df["Churn"].map({"No": 0, "Yes": 1})
    assert y.notna().all(), "Churn contiene etiquetas no reconocidas."

    X = df.drop(columns=["customerID", "Churn"])
    assert len(X.columns) == len(ALL_FEATURES) and set(X.columns) == set(ALL_FEATURES), (
        "Las 19 características no coinciden con las esperadas."
    )
    # No imputamos las otras variables: en este CSV están completas.
    columnas_sin_imputacion = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    assert not X[columnas_sin_imputacion].isna().any().any(), (
        "Se encontraron faltantes fuera de TotalCharges; revisar los datos."
    )
    return X, y


def split_data(X, y, random_state=RANDOM_STATE):
    """Divide de forma estratificada en Train 60 %, Validation 20 %, Test 20 %."""
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=random_state, stratify=y_dev
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_preprocessor(model_type):
    """Crea un ColumnTransformer SIN ajustar para 'logistic' o 'tree'.

    - TotalCharges: faltantes -> 0 en ambos modelos.
    - Categóricas: One-Hot Encoding, sin imputación.
    - tenure y MonthlyCharges: sin imputación.
    - Regresión logística: StandardScaler para las tres numéricas.
    - Árbol de decisión: numéricas sin escalamiento.

    Entrada esperada: DataFrame con las 19 columnas originales.
    """
    if model_type not in ("logistic", "tree"):
        raise ValueError("Use 'logistic' o 'tree'.")

    numeric_transformer = StandardScaler() if model_type == "logistic" else "passthrough"

    totalcharges_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ]
    if model_type == "logistic":
        totalcharges_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(NUMERIC_FEATURES)),
            ("totalcharges", Pipeline(totalcharges_steps), list(TOTALCHARGES_FEATURE)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
