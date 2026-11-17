"""Interpretación de los modelos, análisis de errores e intervalos bootstrap (P2).

Todas las funciones reciben modelos ya ajustados y datos ya particionados; ninguna
ajusta modelos ni elige hiperparámetros. Se usan así:

* P03 y P04 (solo Train y Validation): coeficientes, razones de odds, importancias,
  reglas y hojas del árbol.
* P05 (evaluación final): intervalos bootstrap pareados, errores por subgrupo,
  coincidencia de errores entre modelos y perfil de los errores, sobre Test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.inspection import permutation_importance

from src.modelos import PASO_MODELO, PASO_PREPROCESAMIENTO
from src.protocolo import METRICAS, RANDOM_STATE, calcular_metricas

from preprocessing import (  # noqa: E402  (src.protocolo añade la raíz a sys.path)
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TOTALCHARGES_FEATURE,
)

VARIABLES_NUMERICAS = list(NUMERIC_FEATURES + TOTALCHARGES_FEATURE)
VARIABLES_CATEGORICAS = list(CATEGORICAL_FEATURES)

# Categorías que no aportan información nueva: repiten "InternetService = No" o
# "PhoneService = No" dentro de las variables de servicios adicionales.
CATEGORIAS_ESTRUCTURALES = ("No internet service", "No phone service")
SERVICIOS_DE_INTERNET = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Grupos de antigüedad usados en el EDA de P1 (meses).
LIMITES_ANTIGUEDAD = [-1, 12, 24, 48, 72]
ETIQUETAS_ANTIGUEDAD = ["0–12", "13–24", "25–48", "49–72"]

# Orden en que se presentan los grupos en tablas y figuras (el resto, alfabético).
ORDEN_DE_GRUPOS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    "SeniorCitizen": ["0", "1"],
}


# ---------------------------------------------------------------------------
# Columnas que ve cada clasificador después del preprocesamiento
# ---------------------------------------------------------------------------
def columnas_del_modelo(modelo):
    """Relaciona cada columna transformada con su variable original y su categoría.

    Devuelve un DataFrame con ``Columna`` (nombre de scikit-learn), ``Variable``,
    ``Categoría``, ``Etiqueta`` (texto legible) y ``Tipo``.
    """
    preprocesador = modelo.named_steps[PASO_PREPROCESAMIENTO]
    codificador = preprocesador.named_transformers_["cat"]

    filas, esperados = [], []
    for variable in NUMERIC_FEATURES:
        filas.append((variable, "", variable, "numérica"))
        esperados.append(f"num__{variable}")
    for variable in TOTALCHARGES_FEATURE:
        filas.append((variable, "", variable, "numérica"))
        esperados.append(f"totalcharges__{variable}")
    for variable, categorias in zip(CATEGORICAL_FEATURES, codificador.categories_):
        for categoria in categorias:
            filas.append((variable, str(categoria), f"{variable} = {categoria}", "categórica"))
            esperados.append(f"cat__{variable}_{categoria}")

    nombres = list(preprocesador.get_feature_names_out())
    assert nombres == esperados, "El orden de las columnas transformadas no es el esperado."

    tabla = pd.DataFrame(filas, columns=["Variable", "Categoría", "Etiqueta", "Tipo"])
    tabla.insert(0, "Columna", nombres)
    return tabla


# ---------------------------------------------------------------------------
# Regresión logística: coeficientes y razones de odds
# ---------------------------------------------------------------------------
def tabla_coeficientes(modelo):
    """Coeficientes de la regresión logística ordenados por magnitud.

    El coeficiente es el cambio en el logaritmo de los odds de ``Churn=1``:
    por cada desviación estándar en las variables numéricas (están estandarizadas)
    o por pertenecer a la categoría en las columnas One-Hot. Como el codificador
    conserva todas las categorías, cada coeficiente categórico se lee respecto del
    promedio de su variable; para comparar dos categorías use ``razones_de_odds``.
    """
    clasificador = modelo.named_steps[PASO_MODELO]
    tabla = columnas_del_modelo(modelo)
    tabla["Coeficiente"] = clasificador.coef_.ravel()
    tabla["exp(coeficiente)"] = np.exp(tabla["Coeficiente"])
    tabla["Magnitud"] = tabla["Coeficiente"].abs()
    tabla["Categoría estructural"] = tabla["Categoría"].isin(CATEGORIAS_ESTRUCTURALES)
    tabla = tabla.sort_values("Magnitud", ascending=False, kind="mergesort").reset_index(drop=True)
    tabla.index = tabla.index + 1
    tabla.index.name = "Posición"
    return tabla


def escalas_numericas(modelo):
    """Media y desviación estándar aprendidas de Train por el ``StandardScaler``."""
    preprocesador = modelo.named_steps[PASO_PREPROCESAMIENTO]
    escalador = preprocesador.named_transformers_["num"]
    escalador_total = preprocesador.named_transformers_["totalcharges"].named_steps["scaler"]
    tabla = pd.DataFrame(
        {
            "Media (Train)": list(escalador.mean_) + list(escalador_total.mean_),
            "Desviación estándar (Train)": list(escalador.scale_) + list(escalador_total.scale_),
        },
        index=VARIABLES_NUMERICAS,
    )
    tabla.index.name = "Variable"
    return tabla


def cliente_de_referencia(X_train):
    """Cliente hipotético: mediana de cada variable numérica y categoría más frecuente."""
    referencia = X_train.iloc[[0]].copy()
    for variable in VARIABLES_NUMERICAS:
        referencia[variable] = float(X_train[variable].median())
    for variable in VARIABLES_CATEGORICAS:
        referencia[variable] = X_train[variable].mode().iloc[0]
    fila = referencia.iloc[0]
    assert fila["InternetService"] != "No" and fila["PhoneService"] == "Yes", (
        "El cliente de referencia debe tener internet y teléfono para que los contrastes sean coherentes."
    )
    assert not fila[VARIABLES_CATEGORICAS].isin(CATEGORIAS_ESTRUCTURALES).any()
    return referencia.reset_index(drop=True)


def razones_de_odds(modelo, X_train):
    """Razón de odds de cada categoría frente a la más frecuente, con lo demás fijo.

    Para cada variable categórica se toma el cliente de referencia, se cambia solo
    esa variable y se compara el logaritmo de los odds que estima el modelo. Así el
    contraste no depende de las columnas redundantes del One-Hot Encoding:

    * ``InternetService = No`` implica ``No internet service`` en los seis servicios
      adicionales, y ``PhoneService = No`` implica ``MultipleLines = No phone service``;
      esos cambios se aplican juntos porque en los datos siempre ocurren juntos.
    * Las categorías estructurales no se contrastan por separado.

    Una razón de odds mayor que 1 indica mayor propensión estimada al abandono que
    la categoría de referencia; es una asociación condicional, no un efecto causal.
    """
    referencia = cliente_de_referencia(X_train)
    base = float(modelo.decision_function(referencia[list(ALL_FEATURES)])[0])
    preprocesador = modelo.named_steps[PASO_PREPROCESAMIENTO]
    codificador = preprocesador.named_transformers_["cat"]

    filas = []
    for variable, categorias in zip(CATEGORICAL_FEATURES, codificador.categories_):
        actual = referencia.loc[0, variable]
        for categoria in categorias:
            if categoria == actual or categoria in CATEGORIAS_ESTRUCTURALES:
                continue
            cliente = referencia.copy()
            cliente[variable] = categoria
            if variable == "InternetService" and categoria == "No":
                for servicio in SERVICIOS_DE_INTERNET:
                    cliente[servicio] = "No internet service"
            if variable == "PhoneService" and categoria == "No":
                cliente["MultipleLines"] = "No phone service"
            diferencia = float(modelo.decision_function(cliente[list(ALL_FEATURES)])[0]) - base
            filas.append(
                {
                    "Variable": variable,
                    "Categoría": str(categoria),
                    "Referencia": str(actual),
                    "Diferencia de log-odds": diferencia,
                    "Razón de odds": float(np.exp(diferencia)),
                }
            )
    tabla = pd.DataFrame(filas)
    orden = tabla["Diferencia de log-odds"].abs().sort_values(ascending=False, kind="mergesort").index
    return tabla.loc[orden].reset_index(drop=True)


def efectos_numericos(modelo, X_train, meses=12, cargo=20.0):
    """Lectura de las tres variables numéricas en sus unidades originales.

    ``tenure``, ``MonthlyCharges`` y ``TotalCharges`` están ligadas
    (``TotalCharges`` ≈ ``tenure`` × ``MonthlyCharges``), por lo que mover una sola
    "con las demás fijas" describe un cliente poco realista. Además del coeficiente
    por desviación estándar se evalúan dos escenarios coherentes a partir del
    cliente de referencia:

    * ``meses`` más de antigüedad con el mismo cargo mensual
      (``TotalCharges`` aumenta ``meses`` × ``MonthlyCharges``);
    * ``cargo`` dólares más de cargo mensual durante toda la antigüedad
      (``TotalCharges`` aumenta ``cargo`` × ``tenure``).
    """
    referencia = cliente_de_referencia(X_train)
    columnas = list(ALL_FEATURES)
    base = float(modelo.decision_function(referencia[columnas])[0])
    antiguedad = float(referencia.loc[0, "tenure"])
    cargo_mensual = float(referencia.loc[0, "MonthlyCharges"])

    def contraste(cambios):
        cliente = referencia.copy()
        for variable, valor in cambios.items():
            cliente[variable] = valor
        return float(modelo.decision_function(cliente[columnas])[0]) - base

    escalas = escalas_numericas(modelo)["Desviación estándar (Train)"]
    coeficientes = tabla_coeficientes(modelo).set_index("Columna")["Coeficiente"]
    filas = []
    for variable, columna in zip(
        VARIABLES_NUMERICAS,
        ["num__tenure", "num__MonthlyCharges", "totalcharges__TotalCharges"],
    ):
        filas.append(
            {
                "Escenario": f"{variable}: +1 desviación estándar ({escalas[variable]:.1f}), resto fijo",
                "Diferencia de log-odds": float(coeficientes[columna]),
            }
        )
    filas.append(
        {
            "Escenario": f"+{meses} meses de antigüedad con el mismo cargo mensual",
            "Diferencia de log-odds": contraste(
                {
                    "tenure": antiguedad + meses,
                    "TotalCharges": float(referencia.loc[0, "TotalCharges"]) + meses * cargo_mensual,
                }
            ),
        }
    )
    filas.append(
        {
            "Escenario": f"+{cargo:.0f} USD de cargo mensual durante toda la antigüedad",
            "Diferencia de log-odds": contraste(
                {
                    "MonthlyCharges": cargo_mensual + cargo,
                    "TotalCharges": float(referencia.loc[0, "TotalCharges"]) + cargo * antiguedad,
                }
            ),
        }
    )
    tabla = pd.DataFrame(filas)
    tabla["Razón de odds"] = np.exp(tabla["Diferencia de log-odds"])
    return tabla


# ---------------------------------------------------------------------------
# Importancia por permutación (ambos modelos, sobre las 19 variables originales)
# ---------------------------------------------------------------------------
def importancia_por_permutacion(modelo, X, y, n_repeticiones=30):
    """Caída media de F1 y de ROC-AUC al permutar cada variable original.

    Se permuta la columna original (antes del One-Hot Encoding), de modo que todas
    las categorías de una variable se alteran a la vez. Valores cercanos a cero o
    negativos indican que el modelo no depende de esa variable en estos datos.
    Con variables correlacionadas la importancia se reparte y puede subestimarse.
    """
    resultado = permutation_importance(
        modelo,
        X,
        y,
        scoring={"F1-score": "f1", "ROC-AUC": "roc_auc"},
        n_repeats=n_repeticiones,
        random_state=RANDOM_STATE,
    )
    tabla = pd.DataFrame({"Variable": list(X.columns)})
    for metrica in ("F1-score", "ROC-AUC"):
        tabla[f"Caída de {metrica} (media)"] = resultado[metrica].importances_mean
        tabla[f"Caída de {metrica} (DE)"] = resultado[metrica].importances_std
    tabla = tabla.sort_values("Caída de F1-score (media)", ascending=False, kind="mergesort")
    return tabla.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Árbol de decisión: importancias, reglas y hojas
# ---------------------------------------------------------------------------
def importancias_arbol(modelo):
    """Importancia por reducción de impureza (MDI), por columna y sumada por variable.

    La MDI se calcula solo con Train y favorece a las variables con muchos puntos
    de corte posibles (las numéricas); por eso se contrasta con la permutación.
    """
    clasificador = modelo.named_steps[PASO_MODELO]
    columnas = columnas_del_modelo(modelo)
    columnas["Importancia (MDI)"] = clasificador.feature_importances_
    por_columna = columnas.sort_values("Importancia (MDI)", ascending=False, kind="mergesort")
    por_columna = por_columna.reset_index(drop=True)

    por_variable = (
        columnas.groupby("Variable", sort=False)["Importancia (MDI)"].sum()
        .sort_values(ascending=False, kind="mergesort")
        .reset_index()
    )
    usos = pd.Series(clasificador.tree_.feature[clasificador.tree_.feature >= 0]).map(
        columnas["Variable"]
    ).value_counts()
    por_variable["Divisiones que la usan"] = por_variable["Variable"].map(usos).fillna(0).astype(int)
    return por_columna, por_variable


def _condicion(columnas, indice, umbral, rama):
    """Texto legible de una condición del árbol (``rama``: 'izquierda' = ≤ umbral)."""
    fila = columnas.iloc[indice]
    if fila["Tipo"] == "categórica":
        signo = "≠" if rama == "izquierda" else "="
        return f"{fila['Variable']} {signo} {fila['Categoría']}"
    signo = "≤" if rama == "izquierda" else ">"
    return f"{fila['Variable']} {signo} {umbral:.2f}"


def estructura_arbol(modelo, X, y):
    """Descripción de cada nodo con datos reales (sin ponderar) de ``(X, y)``.

    Devuelve un DataFrame indexado por el identificador del nodo con su profundidad,
    padre, hijos, pregunta, cantidad de clientes de ``X`` que pasan por él, cuántos
    abandonaron y la tasa de abandono observada. Se calcula recorriendo el árbol
    con los datos, por lo que no depende de cómo cada versión de scikit-learn
    almacena los valores ponderados de los nodos.
    """
    preprocesador = modelo.named_steps[PASO_PREPROCESAMIENTO]
    clasificador = modelo.named_steps[PASO_MODELO]
    arbol = clasificador.tree_
    columnas = columnas_del_modelo(modelo)

    X_transformado = preprocesador.transform(X)
    recorrido = clasificador.decision_path(X_transformado)
    clientes = np.asarray(recorrido.sum(axis=0)).ravel()
    abandonos = np.asarray(recorrido.T @ np.asarray(y, dtype=float)).ravel()

    padre = np.full(arbol.node_count, -1)
    profundidad = np.zeros(arbol.node_count, dtype=int)
    rama = np.array([""] * arbol.node_count, dtype=object)
    pila = [0]
    while pila:
        nodo = pila.pop()
        for hijo, lado in ((arbol.children_left[nodo], "izquierda"), (arbol.children_right[nodo], "derecha")):
            if hijo != -1:
                padre[hijo] = nodo
                profundidad[hijo] = profundidad[nodo] + 1
                rama[hijo] = lado
                pila.append(hijo)

    es_hoja = arbol.children_left == -1
    preguntas = [
        "" if es_hoja[nodo] else _condicion(columnas, arbol.feature[nodo], arbol.threshold[nodo], "izquierda")
        for nodo in range(arbol.node_count)
    ]
    tabla = pd.DataFrame(
        {
            "Profundidad": profundidad,
            "Padre": padre,
            "Rama": rama,
            "Hijo izquierdo": arbol.children_left,
            "Hijo derecho": arbol.children_right,
            "Es hoja": es_hoja,
            "Pregunta (sí = izquierda)": preguntas,
            "Clientes": clientes.astype(int),
            "Abandonos": abandonos.round().astype(int),
        }
    )
    tabla["Tasa de abandono"] = np.where(tabla["Clientes"] > 0, tabla["Abandonos"] / tabla["Clientes"].clip(lower=1), np.nan)
    tabla.index.name = "Nodo"
    return tabla


def _camino(modelo, nodo, estructura, columnas):
    """Condiciones (variable, tipo, detalle) desde la raíz hasta ``nodo``."""
    arbol = modelo.named_steps[PASO_MODELO].tree_
    condiciones = []
    while estructura.loc[nodo, "Padre"] != -1:
        padre = int(estructura.loc[nodo, "Padre"])
        condiciones.append((arbol.feature[padre], float(arbol.threshold[padre]), estructura.loc[nodo, "Rama"]))
        nodo = padre
    return condiciones[::-1]


def _regla_simplificada(condiciones, columnas):
    """Une las condiciones de un camino eliminando las que son redundantes."""
    limites = {}      # variable numérica -> [inferior, superior]
    iguales = {}      # variable categórica -> categoría exigida
    distintas = {}    # variable categórica -> categorías excluidas
    orden = []
    for indice, umbral, rama in condiciones:
        fila = columnas.iloc[indice]
        variable = fila["Variable"]
        if variable not in orden:
            orden.append(variable)
        if fila["Tipo"] == "numérica":
            inferior, superior = limites.get(variable, (None, None))
            if rama == "izquierda":
                superior = umbral if superior is None else min(superior, umbral)
            else:
                inferior = umbral if inferior is None else max(inferior, umbral)
            limites[variable] = (inferior, superior)
        elif rama == "derecha":
            iguales[variable] = fila["Categoría"]
        else:
            distintas.setdefault(variable, [])
            if fila["Categoría"] not in distintas[variable]:
                distintas[variable].append(fila["Categoría"])

    partes = []
    for variable in orden:
        if variable in limites:
            inferior, superior = limites[variable]
            if inferior is not None and superior is not None:
                partes.append(f"{inferior:.2f} < {variable} ≤ {superior:.2f}")
            elif superior is not None:
                partes.append(f"{variable} ≤ {superior:.2f}")
            else:
                partes.append(f"{variable} > {inferior:.2f}")
        elif variable in iguales:
            partes.append(f"{variable} = {iguales[variable]}")
        else:
            excluidas = distintas[variable]
            if len(excluidas) == 1:
                partes.append(f"{variable} ≠ {excluidas[0]}")
            else:
                partes.append(f"{variable} ∉ {{{', '.join(excluidas)}}}")
    return " y ".join(partes) if partes else "(todos los clientes)"


def tabla_hojas(modelo, X_train, y_train, X_val, y_val):
    """Una fila por hoja: regla, tamaño y tasa de abandono real en Train y Validation.

    ``Probabilidad estimada`` es la que devuelve ``predict_proba`` (con
    ``class_weight='balanced'`` está inflada respecto de la tasa real de la hoja).
    Comparar la tasa de Train con la de Validation muestra qué reglas se sostienen
    fuera de los datos de entrenamiento.
    """
    preprocesador = modelo.named_steps[PASO_PREPROCESAMIENTO]
    clasificador = modelo.named_steps[PASO_MODELO]
    columnas = columnas_del_modelo(modelo)
    estructura_train = estructura_arbol(modelo, X_train, y_train)
    estructura_val = estructura_arbol(modelo, X_val, y_val)

    hojas_train = clasificador.apply(preprocesador.transform(X_train))
    probabilidad = pd.Series(modelo.predict_proba(X_train)[:, 1]).groupby(hojas_train).first()

    filas = []
    for nodo in estructura_train.index[estructura_train["Es hoja"]]:
        condiciones = _camino(modelo, nodo, estructura_train, columnas)
        filas.append(
            {
                "Hoja": int(nodo),
                "Regla": _regla_simplificada(condiciones, columnas),
                "Condiciones": len(condiciones),
                "Clientes (Train)": int(estructura_train.loc[nodo, "Clientes"]),
                "Abandonos (Train)": int(estructura_train.loc[nodo, "Abandonos"]),
                "Tasa de abandono (Train)": float(estructura_train.loc[nodo, "Tasa de abandono"]),
                "Probabilidad estimada": float(probabilidad.loc[nodo]),
                "Predicción": int(probabilidad.loc[nodo] > 0.5),
                "Clientes (Validation)": int(estructura_val.loc[nodo, "Clientes"]),
                "Abandonos (Validation)": int(estructura_val.loc[nodo, "Abandonos"]),
                "Tasa de abandono (Validation)": float(estructura_val.loc[nodo, "Tasa de abandono"]),
            }
        )
    tabla = pd.DataFrame(filas).sort_values(
        ["Probabilidad estimada", "Clientes (Train)"], ascending=[False, False], kind="mergesort"
    )
    return tabla.reset_index(drop=True)


def hojas_por_banda(hojas, tasa_global):
    """Agrupa las hojas según su tasa de abandono en Train y la compara con Validation.

    Cada hoja recibe pocos clientes de Validation, así que su tasa individual es
    muy ruidosa; al agrupar hojas de riesgo parecido se ve si el orden que aprendió
    el árbol se mantiene fuera de Train. ``tasa_global`` (tasa de abandono de Train)
    es uno de los límites porque, con ``class_weight='balanced'``, una hoja predice
    abandono exactamente cuando su tasa en Train la supera.
    """
    limites = [-0.001, 0.10, tasa_global, 0.50, 0.75, 1.0]
    etiquetas = [
        "0–10 %",
        f"10–{100 * tasa_global:.1f} %",
        f"{100 * tasa_global:.1f}–50 %",
        "50–75 %",
        "75–100 %",
    ]
    banda = pd.cut(hojas["Tasa de abandono (Train)"], bins=limites, labels=etiquetas)
    grupos = hojas.groupby(banda, observed=True)
    tabla = pd.DataFrame(
        {
            "Hojas": grupos.size(),
            "Predicen abandono": grupos["Predicción"].sum(),
            "Clientes (Train)": grupos["Clientes (Train)"].sum(),
            "Tasa de abandono (Train)": grupos["Abandonos (Train)"].sum() / grupos["Clientes (Train)"].sum(),
            "Clientes (Validation)": grupos["Clientes (Validation)"].sum(),
            "Tasa de abandono (Validation)": grupos["Abandonos (Validation)"].sum()
            / grupos["Clientes (Validation)"].sum(),
        }
    )
    tabla.index.name = "Tasa de abandono de la hoja en Train"
    return tabla


def reglas_arbol(modelo, X_train, y_train, profundidad_maxima=None):
    """Árbol en texto con condiciones legibles y la tasa real de abandono de cada nodo."""
    arbol = modelo.named_steps[PASO_MODELO].tree_
    columnas = columnas_del_modelo(modelo)
    estructura = estructura_arbol(modelo, X_train, y_train)
    probabilidad = pd.Series(modelo.predict_proba(X_train)[:, 1]).groupby(
        modelo.named_steps[PASO_MODELO].apply(modelo.named_steps[PASO_PREPROCESAMIENTO].transform(X_train))
    ).first()
    lineas = []

    def resumen(nodo):
        fila = estructura.loc[nodo]
        return f"n={fila['Clientes']}, abandono={100 * fila['Tasa de abandono']:.1f} %"

    def recorrer(nodo, sangria):
        if estructura.loc[nodo, "Es hoja"]:
            clase = "Churn" if probabilidad.loc[nodo] > 0.5 else "No churn"
            lineas.append(f"{sangria}→ {clase} ({resumen(nodo)})")
            return
        if profundidad_maxima is not None and estructura.loc[nodo, "Profundidad"] >= profundidad_maxima:
            lineas.append(f"{sangria}… subárbol omitido ({resumen(nodo)})")
            return
        for hijo, lado in ((arbol.children_left[nodo], "izquierda"), (arbol.children_right[nodo], "derecha")):
            condicion = _condicion(columnas, arbol.feature[nodo], arbol.threshold[nodo], lado)
            lineas.append(f"{sangria}Si {condicion}  [{resumen(hijo)}]")
            recorrer(hijo, sangria + "|   ")

    lineas.append(f"Raíz: {resumen(0)}")
    recorrer(0, "")
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Evaluación final: intervalos bootstrap pareados
# ---------------------------------------------------------------------------
def _metricas_vectorizadas(Y, P, S):
    """Las cinco métricas del protocolo para cada fila (remuestreo) de las matrices."""
    positivos = Y == 1
    predichos = P == 1
    vp = (positivos & predichos).sum(axis=1).astype(float)
    fp = (~positivos & predichos).sum(axis=1).astype(float)
    fn = (positivos & ~predichos).sum(axis=1).astype(float)
    vn = (~positivos & ~predichos).sum(axis=1).astype(float)

    def division(numerador, denominador):
        return np.divide(numerador, denominador, out=np.zeros_like(numerador), where=denominador > 0)

    precision = division(vp, vp + fp)
    recall = division(vp, vp + fn)
    f1 = division(2 * vp, 2 * vp + fp + fn)
    accuracy = (vp + vn) / Y.shape[1]

    # ROC-AUC = estadístico de Mann-Whitney con rangos promedio (maneja empates).
    rangos = rankdata(S, axis=1)
    n_pos = positivos.sum(axis=1).astype(float)
    n_neg = Y.shape[1] - n_pos
    auc = ((rangos * positivos).sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1, "ROC-AUC": auc}


def bootstrap_pareado(y, predicciones, comparaciones, n_remuestreos=2000, confianza=0.95, semilla=RANDOM_STATE):
    """Intervalos de confianza por percentiles con remuestreo bootstrap pareado.

    En cada remuestreo se eligen, con reemplazo, los mismos clientes para todos los
    modelos; así la diferencia entre dos modelos se calcula sobre exactamente los
    mismos casos y el intervalo de la diferencia refleja solo lo que los distingue.

    Parámetros
    ----------
    y : etiquetas reales.
    predicciones : ``{nombre: (y_pred, y_proba)}``.
    comparaciones : lista de pares ``(modelo_a, modelo_b)``; se reporta a − b.

    Devuelve ``(intervalos, diferencias)``. El intervalo de una diferencia que no
    contiene el cero indica que la diferencia no se explica por el muestreo de este
    conjunto; no reemplaza una validación con datos nuevos.
    """
    y = np.asarray(y)
    n = len(y)
    generador = np.random.default_rng(semilla)
    indices = generador.integers(0, n, size=(n_remuestreos, n))
    alfa = (1 - confianza) / 2
    cuantiles = [100 * alfa, 100 * (1 - alfa)]

    Y = y[indices]
    muestras, puntuales = {}, {}
    for nombre, (y_pred, y_proba) in predicciones.items():
        y_pred, y_proba = np.asarray(y_pred), np.asarray(y_proba)
        muestras[nombre] = _metricas_vectorizadas(Y, y_pred[indices], y_proba[indices])
        puntuales[nombre] = calcular_metricas(y, y_pred, y_proba)
        propias = _metricas_vectorizadas(y[None, :], y_pred[None, :], y_proba[None, :])
        for metrica in METRICAS:  # la versión vectorizada debe coincidir con la del protocolo
            assert abs(propias[metrica][0] - puntuales[nombre][metrica]) < 1e-9, (nombre, metrica)

    filas = []
    for nombre in predicciones:
        for metrica in METRICAS:
            inferior, superior = np.percentile(muestras[nombre][metrica], cuantiles)
            filas.append(
                {
                    "Modelo": nombre,
                    "Métrica": metrica,
                    "Estimación": puntuales[nombre][metrica],
                    "IC inferior": inferior,
                    "IC superior": superior,
                }
            )
    intervalos = pd.DataFrame(filas)

    filas = []
    for modelo_a, modelo_b in comparaciones:
        for metrica in METRICAS:
            diferencias = muestras[modelo_a][metrica] - muestras[modelo_b][metrica]
            inferior, superior = np.percentile(diferencias, cuantiles)
            filas.append(
                {
                    "Comparación": f"{modelo_a} − {modelo_b}",
                    "Métrica": metrica,
                    "Diferencia": puntuales[modelo_a][metrica] - puntuales[modelo_b][metrica],
                    "IC inferior": inferior,
                    "IC superior": superior,
                    "El IC excluye 0": bool(inferior > 0 or superior < 0),
                }
            )
    return intervalos, pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Evaluación final: análisis de errores
# ---------------------------------------------------------------------------
def grupo_antiguedad(tenure):
    """Grupos de antigüedad del EDA: 0–12, 13–24, 25–48 y 49–72 meses."""
    return pd.cut(tenure, bins=LIMITES_ANTIGUEDAD, labels=ETIQUETAS_ANTIGUEDAD).astype(str)


def _conteos(y, y_pred):
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    vp = int(((y == 1) & (y_pred == 1)).sum())
    fn = int(((y == 1) & (y_pred == 0)).sum())
    fp = int(((y == 0) & (y_pred == 1)).sum())
    vn = int(((y == 0) & (y_pred == 0)).sum())
    return vp, fn, fp, vn


def errores_por_subgrupo(X, y, predicciones, variables):
    """Errores de cada modelo dentro de los grupos de clientes indicados.

    ``variables`` es un diccionario ``{nombre: serie}`` con el grupo de cada cliente.
    Por grupo y modelo se reportan los conteos de la matriz de confusión
    (VP, FN, FP, VN), Recall (abandonos detectados), tasa de falsos positivos
    (clientes que se quedan y reciben alerta) y Precision. Un valor ausente
    significa que la métrica no está definida en ese grupo (denominador cero).
    """
    y = np.asarray(y)
    filas = []
    for variable, serie in variables.items():
        serie = pd.Series(np.asarray(serie), name=variable).astype(str)
        presentes = sorted(pd.unique(serie))
        orden = [g for g in ORDEN_DE_GRUPOS.get(variable, presentes) if g in presentes]
        orden += [g for g in presentes if g not in orden]
        for grupo in orden:
            mascara = (serie == grupo).to_numpy()
            for nombre, (y_pred, _) in predicciones.items():
                vp, fn, fp, vn = _conteos(y[mascara], np.asarray(y_pred)[mascara])
                filas.append(
                    {
                        "Variable": variable,
                        "Grupo": str(grupo),
                        "Clientes": int(mascara.sum()),
                        "Abandonos": vp + fn,
                        "Tasa de abandono": (vp + fn) / mascara.sum(),
                        "Modelo": nombre,
                        "VP": vp, "FN": fn, "FP": fp, "VN": vn,
                        "Recall": vp / (vp + fn) if (vp + fn) else np.nan,
                        "Tasa de falsos positivos": fp / (fp + vn) if (fp + vn) else np.nan,
                        "Precision": vp / (vp + fp) if (vp + fp) else np.nan,
                        "Alertas (proporción)": (vp + fp) / mascara.sum(),
                    }
                )
    return pd.DataFrame(filas)


def resumen_por_subgrupo(tabla):
    """Versión compacta de ``errores_por_subgrupo``: una fila por grupo de clientes.

    Para cada modelo deja dos columnas: Recall (abandonos detectados) y TFP (tasa de
    falsos positivos: clientes que se quedan y reciben una alerta).
    """
    claves = ["Variable", "Grupo"]
    resumen = tabla.drop_duplicates(claves).set_index(claves)[["Clientes", "Abandonos", "Tasa de abandono"]]
    for metrica, etiqueta in (("Recall", "Recall"), ("Tasa de falsos positivos", "TFP")):
        for modelo in dict.fromkeys(tabla["Modelo"]):
            resumen[f"{etiqueta} · {modelo}"] = tabla[tabla["Modelo"] == modelo].set_index(claves)[metrica]
    return resumen


def coincidencia_de_errores(y, pred_a, pred_b, nombre_a, nombre_b):
    """Compara, cliente por cliente, los aciertos y errores de dos modelos."""
    y, pred_a, pred_b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    filas = []
    for etiqueta, clase, descripcion in (
        ("Clientes que abandonaron (Churn=1)", 1, "detecta"),
        ("Clientes que se quedaron (Churn=0)", 0, "da falsa alarma"),
    ):
        mascara = y == clase
        a, b = pred_a[mascara] == 1, pred_b[mascara] == 1
        total = int(mascara.sum())
        for situacion, cantidad in (
            (f"Ambos modelos: {descripcion}", int((a & b).sum())),
            (f"Solo {nombre_a}: {descripcion}", int((a & ~b).sum())),
            (f"Solo {nombre_b}: {descripcion}", int((~a & b).sum())),
            (f"Ninguno: {descripcion}", int((~a & ~b).sum())),
        ):
            filas.append({"Clase real": etiqueta, "Situación": situacion, "Clientes": cantidad, "Proporción": cantidad / total})
    return pd.DataFrame(filas)


def perfil_de_resultados(X, y, y_pred):
    """Perfil de los clientes según el resultado de la predicción (VP, FN, FP, VN).

    Sirve para describir a qué clientes que abandonaron no detecta el modelo (FN) y
    qué clientes que se quedaron reciben una alerta (FP).
    """
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    resultado = np.select(
        [(y == 1) & (y_pred == 1), (y == 1) & (y_pred == 0), (y == 0) & (y_pred == 1)],
        ["VP (abandono detectado)", "FN (abandono no detectado)", "FP (falsa alarma)"],
        default="VN (se queda, sin alerta)",
    )
    datos = X.reset_index(drop=True).copy()
    datos["Resultado"] = resultado
    orden = ["VP (abandono detectado)", "FN (abandono no detectado)", "FP (falsa alarma)", "VN (se queda, sin alerta)"]
    grupos = datos.groupby("Resultado", sort=False)
    perfil = pd.DataFrame(
        {
            "Clientes": grupos.size(),
            "tenure (mediana)": grupos["tenure"].median(),
            "MonthlyCharges (mediana)": grupos["MonthlyCharges"].median(),
            "Contract = Month-to-month (%)": 100 * grupos["Contract"].apply(lambda s: (s == "Month-to-month").mean()),
            "InternetService = Fiber optic (%)": 100 * grupos["InternetService"].apply(lambda s: (s == "Fiber optic").mean()),
            "PaymentMethod = Electronic check (%)": 100 * grupos["PaymentMethod"].apply(lambda s: (s == "Electronic check").mean()),
            "SeniorCitizen = 1 (%)": 100 * grupos["SeniorCitizen"].apply(lambda s: (s == 1).mean()),
        }
    )
    perfil = perfil.reindex([fila for fila in orden if fila in perfil.index])
    perfil.index.name = "Resultado"
    return perfil


def resumen_de_probabilidades(y, predicciones):
    """Compara la probabilidad media estimada con la tasa real de abandono.

    Con ``class_weight='balanced'`` los modelos aprenden como si ambas clases
    fueran igual de frecuentes; sus probabilidades sirven para ordenar clientes,
    pero sobrestiman la frecuencia real del abandono (no están calibradas).
    """
    y = np.asarray(y)
    filas = []
    for nombre, (y_pred, y_proba) in predicciones.items():
        y_proba = np.asarray(y_proba)
        filas.append(
            {
                "Modelo": nombre,
                "Tasa real de abandono": y.mean(),
                "Probabilidad media estimada": y_proba.mean(),
                "Proporción de alertas (umbral 0,5)": np.asarray(y_pred).mean(),
                "Probabilidad media si Churn=1": y_proba[y == 1].mean(),
                "Probabilidad media si Churn=0": y_proba[y == 0].mean(),
            }
        )
    return pd.DataFrame(filas).set_index("Modelo")
