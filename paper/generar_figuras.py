"""Genera en ``paper/figures/`` las figuras de las secciones de P2 del informe.

Uso, desde la raíz del repositorio y después de ejecutar P03, P04 y P05:

    python paper/generar_figuras.py

El script no selecciona ni evalúa nada nuevo. Lee las tablas que los notebooks
guardan en ``resultados/`` y reutiliza el estilo de ``src/graficos.py``, con el
tamaño exacto del formato IEEE (una columna: 3.5 in; dos columnas: 7.16 in) para
que la letra se lea al tamaño final. Solo el diagrama del árbol vuelve a ajustar,
con Train, la configuración guardada en ``configuracion_arbol_decision.json``.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src import graficos  # noqa: E402
from src.modelos import PROFUNDIDADES, VALORES_C, crear_arbol_decision  # noqa: E402
from src.protocolo import (  # noqa: E402
    NOMBRE_ARBOL,
    NOMBRE_DUMMY,
    NOMBRE_LOGISTICA,
    RESULTS_DIR,
    cargar_train_validation,
    leer_json,
)

DESTINO = RAIZ / "paper" / "figures"


def leer_tabla(nombre):
    """Lee un CSV de ``resultados/`` conservando el texto "None" de las grillas."""
    return pd.read_csv(RESULTS_DIR / nombre, keep_default_na=False, na_values=[""])


def guardar(figura, nombre):
    DESTINO.mkdir(parents=True, exist_ok=True)
    ruta = DESTINO / nombre
    figura.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figura)
    print("  ", ruta.relative_to(RAIZ))


def figura_busqueda():
    """F1 en Train y Validation de las configuraciones ponderadas de ambos modelos."""
    logistica = leer_tabla("resultados_regresion_logistica_grilla_validacion.csv")
    arbol = leer_tabla("resultados_arbol_decision_grilla_validacion.csv")
    elegida_logistica = leer_json("configuracion_regresion_logistica.json")["configuracion"]
    elegida_arbol = leer_json("configuracion_arbol_decision.json")["configuracion"]

    figura, (eje_l, eje_a) = plt.subplots(1, 2, figsize=(graficos.ANCHO_PAGINA, 2.6))

    # Regresión logística: fuerza de la regularización (C) y tipo de penalización.
    datos = logistica[logistica["class_weight"] == "balanced"]
    for penalizacion, color, marcador, etiqueta in (("l2", "#0072B2", "o", "L2"), ("l1", "#CC79A7", "s", "L1")):
        serie = datos[datos["penalizacion"] == penalizacion].sort_values("C")
        eje_l.plot(serie["C"], serie["F1-score (Validation)"], color=color, marker=marcador, markersize=4,
                   linewidth=1.6, label=f"{etiqueta} · Validation")
        eje_l.plot(serie["C"], serie["F1-score (Train)"], color=color, marker=marcador, markersize=3,
                   linewidth=1.0, linestyle="--", alpha=0.75, label=f"{etiqueta} · Train")
    eje_l.set_xscale("log")
    eje_l.set_xticks(VALORES_C)
    eje_l.set_xticklabels([f"{c:g}" for c in VALORES_C], rotation=45)
    eje_l.minorticks_off()
    eje_l.set_ylim(0.38, 0.72)
    eje_l.set_xlabel("C (inverso de la regularización)")
    eje_l.set_ylabel("F1-score de Churn=1")
    eje_l.set_title("(a) Regresión logística")
    fila = logistica.set_index("Configuración").loc[elegida_logistica]
    graficos._marcar_seleccion(eje_l, fila["C"], fila["F1-score (Validation)"], desplazamiento=(6, -15))
    eje_l.legend(loc="lower right", frameon=True, fontsize=7.5)

    # Árbol de decisión: profundidad máxima y tamaño mínimo de hoja.
    posiciones = {("None" if p is None else str(p)): i for i, p in enumerate(PROFUNDIDADES)}
    datos = arbol[arbol["class_weight"] == "balanced"].copy()
    datos["x"] = datos["max_depth"].astype(str).map(posiciones)
    for minimo, color, marcador in ((1, "#D55E00", "o"), (20, "#0072B2", "s"), (50, "#009E73", "^")):
        serie = datos[datos["min_samples_leaf"] == minimo].sort_values("x")
        eje_a.plot(serie["x"], serie["F1-score (Validation)"], color=color, marker=marcador, markersize=4,
                   linewidth=1.6, label=f"Hoja ≥ {minimo} · Validation")
        eje_a.plot(serie["x"], serie["F1-score (Train)"], color=color, marker=marcador, markersize=3,
                   linewidth=1.0, linestyle="--", alpha=0.75, label=f"Hoja ≥ {minimo} · Train")
    eje_a.set_xticks(range(len(PROFUNDIDADES)))
    eje_a.set_xticklabels(["Sin límite" if p is None else str(p) for p in PROFUNDIDADES])
    eje_a.set_xlabel("Profundidad máxima (max_depth)")
    eje_a.set_ylabel("F1-score de Churn=1")
    eje_a.set_title("(b) Árbol de decisión")
    fila = datos.set_index("Configuración").loc[elegida_arbol]
    graficos._marcar_seleccion(eje_a, fila["x"], fila["F1-score (Validation)"], desplazamiento=(-62, -14))
    eje_a.legend(loc="upper left", frameon=True, fontsize=7.5)

    figura.tight_layout()
    return figura


def figura_arbol():
    """Primeros dos niveles del árbol seleccionado, con las tasas observadas en Train.

    El diagrama de tres niveles, a dos columnas, queda en
    ``resultados/figuras/p04_arbol_decision_primeros_niveles.png``.
    """
    X_train, _, y_train, _ = cargar_train_validation()
    configuracion = leer_json("configuracion_arbol_decision.json")
    modelo = crear_arbol_decision(**configuracion["hiperparametros"]).fit(X_train, y_train)
    estructura = modelo.named_steps["modelo"].tree_
    assert estructura.n_leaves == configuracion["hojas"], "El árbol reajustado no coincide con el de P04."
    return graficos.figura_arbol(modelo, X_train, y_train, niveles=2, ancho=graficos.ANCHO_COLUMNA)


def figura_importancias():
    """Importancia por permutación (Validation) de ambos modelos."""
    figura = graficos.figura_importancias(
        leer_tabla("importancia_permutacion_regresion_logistica_validacion.csv"),
        leer_tabla("importancia_permutacion_arbol_decision_validacion.csv"),
        cantidad=8,
    )
    figura.set_size_inches(graficos.ANCHO_COLUMNA, 2.5)  # más compacta que en el notebook P05
    figura.tight_layout()
    return figura


def figura_curvas():
    """Curvas ROC y Precision-Recall en Test, en el ancho de una columna."""
    predicciones = leer_tabla("predicciones_test.csv")
    y = predicciones["Churn real"].to_numpy()
    figura, (eje_roc, eje_pr) = plt.subplots(1, 2, figsize=(graficos.ANCHO_COLUMNA, 1.95))
    for nombre in (NOMBRE_LOGISTICA, NOMBRE_ARBOL):
        color = graficos.COLOR_MODELO[nombre]
        probabilidad = predicciones[f"Probabilidad · {nombre}"].to_numpy()
        prediccion = predicciones[f"Predicción · {nombre}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, probabilidad)
        precision, recall, _ = precision_recall_curve(y, probabilidad)
        eje_roc.plot(fpr, tpr, color=color, linewidth=1.4)
        eje_pr.plot(recall, precision, color=color, linewidth=1.4)
        vp = ((y == 1) & (prediccion == 1)).sum()
        fp = ((y == 0) & (prediccion == 1)).sum()
        eje_roc.scatter([fp / (y == 0).sum()], [vp / (y == 1).sum()], color=color, edgecolor="black", s=26, zorder=5)
        eje_pr.scatter([vp / (y == 1).sum()], [vp / (vp + fp)], color=color, edgecolor="black", s=26, zorder=5)
    gris = graficos.COLOR_MODELO[NOMBRE_DUMMY]
    eje_roc.plot([0, 1], [0, 1], color=gris, linestyle="--", linewidth=1.0)
    eje_pr.axhline(y.mean(), color=gris, linestyle="--", linewidth=1.0)
    eje_roc.set_xlabel("Tasa de falsos positivos")
    eje_roc.set_ylabel("Recall")
    eje_roc.set_title("ROC")
    eje_pr.set_xlabel("Recall")
    eje_pr.set_ylabel("Precision")
    eje_pr.set_title("Precision-Recall")
    for eje in (eje_roc, eje_pr):
        eje.set_xlim(-0.02, 1.02)
        eje.set_ylim(-0.02, 1.02)
        eje.set_xticks([0, 0.5, 1])
        eje.set_yticks([0, 0.25, 0.5, 0.75, 1])
        eje.set_aspect("equal")
    leyenda = [
        Line2D([], [], color=graficos.COLOR_MODELO[NOMBRE_LOGISTICA], linewidth=1.6, label="Reg. logística"),
        Line2D([], [], color=graficos.COLOR_MODELO[NOMBRE_ARBOL], linewidth=1.6, label="Árbol de decisión"),
        Line2D([], [], color=gris, linestyle="--", linewidth=1.0, label="Referencia sin modelo"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", markersize=5,
               label="Umbral 0.5"),
    ]
    figura.tight_layout(pad=0.3, w_pad=0.8)
    figura.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, 0.10), ncol=2, frameon=False,
                  fontsize=7.5, columnspacing=1.2, handlelength=1.8)
    return figura


def main():
    graficos.aplicar_estilo()
    print("Figuras del informe:")
    guardar(figura_busqueda(), "busqueda_hiperparametros.png")
    guardar(figura_arbol(), "arbol_primeros_niveles.png")
    guardar(figura_importancias(), "importancia_permutacion.png")
    guardar(figura_curvas(), "curvas_roc_pr.png")


if __name__ == "__main__":
    main()
