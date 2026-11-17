"""Protocolo experimental compartido de P2.

Este módulo implementa, una sola vez, las reglas acordadas en
``notebooks/P02_baseline_diseno_experimental.ipynb`` para que la regresión
logística, el árbol de decisión y la comparación final utilicen exactamente las
mismas particiones, las mismas métricas y el mismo criterio de selección:

* Particiones estratificadas 60/20/20 de ``preprocessing.split_data`` (semilla 42).
* Cada configuración se entrena solo con Train y se evalúa con Validation
  (validación fija *hold-out*; no se usa K-Fold ni ``GridSearchCV``).
* Criterio de selección: mayor F1-score de ``Churn=1`` en Validation, con las
  reglas de desempate documentadas en P02.
* Umbral fijo de 0,5 mediante ``predict()``; el ROC-AUC se calcula con la
  probabilidad estimada de la clase 1.
* Test se reserva para la evaluación final (P05) y no interviene en ninguna decisión.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import RANDOM_STATE, load_data, split_data  # noqa: E402

DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS_DIR = ROOT / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"

METRICAS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
METRICA_SELECCION = "F1-score"
UMBRAL = 0.5

NOMBRE_DUMMY = "Dummy Classifier"
NOMBRE_LOGISTICA = "Regresión logística"
NOMBRE_ARBOL = "Árbol de decisión"

FILAS_MATRIZ = ["Real No (0)", "Real Sí (1)"]
COLUMNAS_MATRIZ = ["Predicho No (0)", "Predicho Sí (1)"]


# ---------------------------------------------------------------------------
# Datos y particiones
# ---------------------------------------------------------------------------
def cargar_datos():
    """Devuelve ``(X, y)`` con el CSV del repositorio o, si falta, con la URL pública."""
    if DATA_PATH.is_file():
        return load_data(DATA_PATH)
    return load_data()


def cargar_particiones():
    """Particiones 60/20/20 del protocolo: Train, Validation y Test.

    Solo debe llamarse desde la evaluación final (P05). Los notebooks de
    selección de hiperparámetros usan ``cargar_train_validation``.
    """
    X, y = cargar_datos()
    return split_data(X, y)


def cargar_train_validation():
    """Devuelve únicamente Train y Validation.

    Las particiones son las mismas de ``split_data``; Test se descarta dentro de
    esta función para que no esté disponible durante la selección de modelos.
    """
    X_train, X_val, _, y_train, y_val, _ = cargar_particiones()
    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Métricas (mismas definiciones que P02)
# ---------------------------------------------------------------------------
def calcular_metricas(y_real, y_pred, y_proba):
    """Calcula las cinco métricas del protocolo para la clase positiva ``Churn=1``."""
    return {
        "Accuracy": accuracy_score(y_real, y_pred),
        "Precision": precision_score(y_real, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_real, y_pred, pos_label=1, zero_division=0),
        "F1-score": f1_score(y_real, y_pred, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y_real, y_proba),
    }


def predecir(modelo, X):
    """Devuelve ``(y_pred, y_proba)``: clase con umbral 0,5 y probabilidad de ``Churn=1``."""
    y_pred = np.asarray(modelo.predict(X))
    y_proba = np.asarray(modelo.predict_proba(X))[:, 1]
    return y_pred, y_proba


def evaluar_modelo(modelo, X, y):
    """Métricas del protocolo de un modelo ya ajustado sobre ``(X, y)``."""
    y_pred, y_proba = predecir(modelo, X)
    return calcular_metricas(y, y_pred, y_proba)


def matriz_confusion(y_real, y_pred):
    """Matriz de confusión con el formato de P02 (filas = clase real; columnas = predicción)."""
    matriz = pd.DataFrame(
        confusion_matrix(y_real, y_pred, labels=[0, 1]),
        index=FILAS_MATRIZ,
        columns=COLUMNAS_MATRIZ,
    )
    matriz.index.name = "Clase real"
    return matriz


def tabla_metricas(metricas_por_modelo, conjunto):
    """Tabla con el formato de P02: índice ``Modelo``, columna ``Conjunto`` y cinco métricas."""
    tabla = pd.DataFrame.from_dict(metricas_por_modelo, orient="index")[METRICAS]
    tabla.index.name = "Modelo"
    tabla.insert(0, "Conjunto", conjunto)
    return tabla


# ---------------------------------------------------------------------------
# Búsqueda de hiperparámetros con validación fija
# ---------------------------------------------------------------------------
def buscar_en_validacion(crear_modelo, grilla, X_train, y_train, X_val, y_val, describir=None):
    """Evalúa cada configuración de la grilla con un bucle explícito.

    Para cada diccionario de hiperparámetros, en el orden de ``grilla``:

    1. se construye un ``Pipeline`` nuevo con ``crear_modelo(**parametros)``;
    2. se ajusta **solo con Train** (preprocesador y clasificador);
    3. se calculan las cinco métricas en Train y en Validation.

    Las métricas de Train se guardan únicamente para medir la brecha
    Train-Validation (sobreajuste); nunca se usan para elegir.

    Devuelve un DataFrame con una fila por configuración. La columna
    ``Configuración`` es la posición en la grilla (empieza en 1).
    """
    filas = []
    for numero, parametros in enumerate(grilla, start=1):
        modelo = crear_modelo(**parametros)
        modelo.fit(X_train, y_train)

        fila = {"Configuración": numero}
        fila.update({nombre: _texto(valor) for nombre, valor in parametros.items()})
        for conjunto, (X, y) in {"Train": (X_train, y_train), "Validation": (X_val, y_val)}.items():
            metricas = evaluar_modelo(modelo, X, y)
            fila.update({f"{nombre} ({conjunto})": valor for nombre, valor in metricas.items()})
        if describir is not None:
            fila.update(describir(modelo))
        filas.append(fila)
    return pd.DataFrame(filas)


def seleccionar_configuracion(tabla, grilla, desempate):
    """Aplica el criterio de selección de P02 y devuelve la configuración elegida.

    Orden de preferencia:

    1. mayor F1-score de ``Churn=1`` en Validation;
    2. regla de desempate del modelo (``desempate(parametros)`` devuelve una tupla;
       se prefiere la menor), que favorece la configuración más sencilla;
    3. si el empate persiste, la primera configuración según el orden de la grilla.

    Devuelve ``(numero, parametros, empatadas)``, donde ``empatadas`` es la
    cantidad de configuraciones que comparten el mejor F1-score.
    """
    columna = f"{METRICA_SELECCION} (Validation)"
    f1 = tabla.set_index("Configuración")[columna].round(12)
    candidatos = sorted(
        f1.index,
        key=lambda numero: (-f1.loc[numero], desempate(grilla[numero - 1]), numero),
    )
    elegido = int(candidatos[0])
    empatadas = int((f1 == f1.loc[elegido]).sum())
    return elegido, dict(grilla[elegido - 1]), empatadas


def _texto(valor):
    """Representa ``None`` como texto para que no se pierda al exportar a CSV."""
    return "None" if valor is None else valor


# ---------------------------------------------------------------------------
# Exportación de resultados
# ---------------------------------------------------------------------------
def guardar_tabla(tabla, nombre, index=True):
    """Guarda un DataFrame en ``resultados/`` con la misma codificación de P02."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = RESULTS_DIR / nombre
    tabla.to_csv(ruta, encoding="utf-8-sig", index=index)
    return ruta


def guardar_json(datos, nombre):
    """Guarda un diccionario en ``resultados/`` como JSON legible."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = RESULTS_DIR / nombre
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ruta


def leer_json(nombre):
    """Lee un JSON de ``resultados/``."""
    return json.loads((RESULTS_DIR / nombre).read_text(encoding="utf-8"))


def versiones():
    """Versiones del entorno, para registrarlas junto con los resultados."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
