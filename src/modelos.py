"""Pipelines y grillas de hiperparámetros de los dos modelos supervisados de P2.

Cada función ``crear_*`` devuelve un ``Pipeline`` **sin ajustar** formado por el
preprocesador compartido (``preprocessing.make_preprocessor``) y el clasificador.
Al llamar a ``fit`` con Train, el imputador, el escalador y el codificador
aprenden sus parámetros solo de Train, lo que evita fuga de información hacia
Validation o Test.

Las grillas se definen aquí, antes de ejecutar los experimentos, y su orden es
el que se usa como último criterio de desempate.
"""

from __future__ import annotations

import warnings

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.protocolo import RANDOM_STATE  # también añade la raíz del repositorio a sys.path

from preprocessing import make_preprocessor  # noqa: E402

_VERSION_SKLEARN = tuple(int(parte) for parte in sklearn.__version__.split(".")[:2])

# Las versiones de scikit-learn anteriores a 1.8 envían a SciPy la opción de impresión
# ``iprint`` de L-BFGS-B, que las versiones recientes de SciPy ya no reconocen. El aviso
# no afecta al ajuste, así que se silencia únicamente ese mensaje (ningún otro aviso).
warnings.filterwarnings("ignore", message=".*iprint.*")

PASO_PREPROCESAMIENTO = "preprocesamiento"
PASO_MODELO = "modelo"

# ---------------------------------------------------------------------------
# Regresión logística
# ---------------------------------------------------------------------------
# C es el inverso de la fuerza de regularización: valores pequeños regularizan más.
VALORES_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
PENALIZACIONES = ["l2", "l1"]
PESOS_DE_CLASE = [None, "balanced"]

# Orden documentado de la grilla: penalización -> class_weight -> C ascendente.
GRILLA_LOGISTICA = [
    {"penalizacion": penalizacion, "class_weight": class_weight, "C": C}
    for penalizacion in PENALIZACIONES
    for class_weight in PESOS_DE_CLASE
    for C in VALORES_C
]

MAX_ITER_LOGISTICA = 20000
TOLERANCIA_LOGISTICA = 1e-7


def crear_regresion_logistica(penalizacion="l2", class_weight=None, C=1.0):
    """Pipeline de regresión logística regularizada.

    * ``penalizacion='l2'`` (ridge) usa el optimizador ``lbfgs``; ``'l1'`` (lasso)
      usa ``saga``, el optimizador de scikit-learn que admite L1 sin penalizar el
      intercepto. Ambos minimizan la misma pérdida logística.
    * El preprocesador ``'logistic'`` estandariza las tres variables numéricas,
      requisito para que la penalización trate a todas las columnas en la misma
      escala y para que ``saga`` converja.
    * La tolerancia estricta y el máximo de iteraciones alto garantizan que los
      resultados no dependan de una convergencia incompleta.
    """
    if penalizacion not in ("l1", "l2"):
        raise ValueError("Use penalizacion='l1' o 'l2'.")

    comunes = dict(
        C=C,
        class_weight=class_weight,
        solver="lbfgs" if penalizacion == "l2" else "saga",
        max_iter=MAX_ITER_LOGISTICA,
        tol=TOLERANCIA_LOGISTICA,
        random_state=RANDOM_STATE,
    )
    if _VERSION_SKLEARN >= (1, 8):
        # Desde scikit-learn 1.8 la penalización se indica con l1_ratio (0 = L2, 1 = L1).
        clasificador = LogisticRegression(l1_ratio=1.0 if penalizacion == "l1" else 0.0, **comunes)
    else:
        clasificador = LogisticRegression(penalty=penalizacion, **comunes)

    return Pipeline(
        [
            (PASO_PREPROCESAMIENTO, make_preprocessor("logistic")),
            (PASO_MODELO, clasificador),
        ]
    )


def desempate_logistica(parametros):
    """Regla de P02: ante el mismo F1 se prefiere el menor C (mayor regularización)."""
    return (parametros["C"],)


def describir_logistica(modelo):
    """Información adicional de cada ajuste: convergencia y coeficientes distintos de cero."""
    clasificador = modelo.named_steps[PASO_MODELO]
    iteraciones = int(clasificador.n_iter_.max())
    return {
        "Coeficientes no nulos": int((clasificador.coef_ != 0).sum()),
        "Iteraciones": iteraciones,
        "Convergió": iteraciones < MAX_ITER_LOGISTICA,
    }


# ---------------------------------------------------------------------------
# Árbol de decisión
# ---------------------------------------------------------------------------
PROFUNDIDADES = [2, 3, 4, 5, 6, 8, None]  # None = sin límite de profundidad
MINIMOS_POR_HOJA = [1, 20, 50]

# Orden documentado de la grilla: class_weight -> max_depth ascendente -> min_samples_leaf.
GRILLA_ARBOL = [
    {"class_weight": class_weight, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
    for class_weight in PESOS_DE_CLASE
    for max_depth in PROFUNDIDADES
    for min_samples_leaf in MINIMOS_POR_HOJA
]


def crear_arbol_decision(class_weight=None, max_depth=None, min_samples_leaf=1):
    """Pipeline de árbol de decisión (CART, impureza de Gini).

    El sobreajuste se controla con prepoda: ``max_depth`` limita la cantidad de
    preguntas encadenadas y ``min_samples_leaf`` impide hojas con muy pocos
    clientes. El preprocesador ``'tree'`` no escala las variables numéricas
    porque las particiones del árbol solo dependen del orden de los valores.
    """
    clasificador = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        [
            (PASO_PREPROCESAMIENTO, make_preprocessor("tree")),
            (PASO_MODELO, clasificador),
        ]
    )


def desempate_arbol(parametros):
    """Regla de P02: menor ``max_depth`` (``None`` = sin límite) y, luego, mayor ``min_samples_leaf``."""
    profundidad = float("inf") if parametros["max_depth"] is None else parametros["max_depth"]
    return (profundidad, -parametros["min_samples_leaf"])


def describir_arbol(modelo):
    """Tamaño real del árbol ajustado."""
    clasificador = modelo.named_steps[PASO_MODELO]
    return {
        "Profundidad real": int(clasificador.get_depth()),
        "Hojas": int(clasificador.get_n_leaves()),
    }
