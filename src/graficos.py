"""Estilo y figuras de P2 (notebooks e informe).

Convenciones:

* Mismo estilo que las figuras de P1 (seaborn ``whitegrid``) y mismos colores de
  clase: verde para ``Churn = No`` y naranja para ``Churn = Yes``.
* Un color fijo por modelo en todas las figuras.
* Tamaños pensados para el formato IEEE: 3.5 in (una columna) o 7.16 in (dos
  columnas), con letra de 8 pt o más al tamaño final.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve, roc_curve

from src.analisis import CATEGORIAS_ESTRUCTURALES, estructura_arbol
from src.modelos import PASO_MODELO, PROFUNDIDADES, VALORES_C
from src.protocolo import (
    FIGURES_DIR,
    METRICAS,
    NOMBRE_ARBOL,
    NOMBRE_DUMMY,
    NOMBRE_LOGISTICA,
    UMBRAL,
)

ANCHO_COLUMNA = 3.5
ANCHO_PAGINA = 7.16

COLOR_NO = "#66c2a5"
COLOR_SI = "#fc8d62"
COLOR_MODELO = {
    NOMBRE_DUMMY: "#7f7f7f",
    NOMBRE_LOGISTICA: "#0072B2",
    NOMBRE_ARBOL: "#D55E00",
}
ABREVIATURA = {
    NOMBRE_DUMMY: "Dummy",
    NOMBRE_LOGISTICA: "Reg. logística",
    NOMBRE_ARBOL: "Árbol de decisión",
}


def aplicar_estilo():
    """Activa el estilo común de las figuras."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )


def guardar_figura(figura, nombre):
    """Guarda la figura en ``resultados/figuras`` (PNG a 300 dpi) y devuelve la ruta."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / nombre
    figura.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
    return ruta


def _marcar_seleccion(eje, x, y, texto="Seleccionada", desplazamiento=(8, -16)):
    eje.scatter([x], [y], marker="*", s=170, color="#F0E442", edgecolor="black", linewidth=0.8, zorder=6)
    eje.annotate(
        texto, (x, y), xytext=desplazamiento, textcoords="offset points", fontsize=8,
        ha="left", va="center", zorder=7,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
    )


# ---------------------------------------------------------------------------
# Regresión logística
# ---------------------------------------------------------------------------
def figura_busqueda_logistica(tabla, elegido):
    """F1 en Train y Validation de las 36 configuraciones de la regresión logística."""
    figura, ejes = plt.subplots(1, 2, figsize=(ANCHO_PAGINA, 2.75), sharey=True)
    estilos = {"l2": ("#0072B2", "o", "L2"), "l1": ("#CC79A7", "s", "L1")}
    titulos = {"None": "Sin ponderar (class_weight=None)", "balanced": "Ponderado (class_weight='balanced')"}
    for eje, (peso, titulo) in zip(ejes, titulos.items()):
        for penalizacion, (color, marcador, etiqueta) in estilos.items():
            datos = tabla[(tabla["class_weight"] == peso) & (tabla["penalizacion"] == penalizacion)].sort_values("C")
            eje.plot(datos["C"], datos["F1-score (Validation)"], color=color, marker=marcador, markersize=4,
                     linewidth=1.6, label=f"{etiqueta} · Validation")
            eje.plot(datos["C"], datos["F1-score (Train)"], color=color, marker=marcador, markersize=3,
                     linewidth=1.0, linestyle="--", alpha=0.75, label=f"{etiqueta} · Train")
        eje.set_xscale("log")
        eje.set_xticks(VALORES_C)
        eje.set_xticklabels([f"{c:g}" for c in VALORES_C], rotation=45)
        eje.minorticks_off()
        eje.set_xlabel("C (inverso de la regularización)")
        eje.set_title(titulo)
    ejes[0].set_ylabel("F1-score de Churn=1")
    fila = tabla.set_index("Configuración").loc[elegido]
    eje = ejes[0] if fila["class_weight"] == "None" else ejes[1]
    _marcar_seleccion(eje, fila["C"], fila["F1-score (Validation)"])
    ejes[0].legend(loc="lower right", frameon=True, ncol=1)
    figura.tight_layout()
    return figura


def figura_coeficientes(tabla_coeficientes, cantidad=15):
    """Coeficientes de mayor magnitud (sin las columnas estructurales repetidas)."""
    datos = tabla_coeficientes[~tabla_coeficientes["Categoría"].isin(CATEGORIAS_ESTRUCTURALES)].head(cantidad)
    datos = datos.iloc[::-1]
    etiquetas = [
        f"{fila['Variable']} (por 1 DE)" if fila["Tipo"] == "numérica" else fila["Etiqueta"]
        for _, fila in datos.iterrows()
    ]
    colores = [COLOR_SI if valor > 0 else COLOR_NO for valor in datos["Coeficiente"]]
    figura, eje = plt.subplots(figsize=(ANCHO_COLUMNA, 0.22 * len(datos) + 0.9))
    eje.barh(etiquetas, datos["Coeficiente"], color=colores, edgecolor="black", linewidth=0.4)
    eje.axvline(0, color="black", linewidth=0.8)
    eje.set_xlabel("Coeficiente (log-odds)")
    eje.grid(axis="y", visible=False)
    limite = 1.75 * datos["Coeficiente"].abs().max()
    eje.set_xlim(-limite, limite)
    for posicion, valor in enumerate(datos["Coeficiente"]):
        eje.text(valor + (0.02 if valor > 0 else -0.02) * limite, posicion, f"{valor:+.2f}",
                 va="center", ha="left" if valor > 0 else "right", fontsize=7.5)
    figura.tight_layout()
    return figura


def figura_razones_de_odds(tabla_odds, cantidad=14):
    """Razones de odds frente a la categoría más frecuente (escala logarítmica)."""
    datos = tabla_odds.head(cantidad).iloc[::-1]
    etiquetas = [f"{f['Variable']}: {f['Categoría']} vs. {f['Referencia']}" for _, f in datos.iterrows()]
    colores = [COLOR_SI if valor > 1 else COLOR_NO for valor in datos["Razón de odds"]]
    figura, eje = plt.subplots(figsize=(ANCHO_PAGINA, 0.23 * len(datos) + 0.9))
    eje.hlines(range(len(datos)), 1, datos["Razón de odds"], color=colores, linewidth=2.2)
    eje.scatter(datos["Razón de odds"], range(len(datos)), color=colores, edgecolor="black", linewidth=0.5, s=38, zorder=3)
    eje.axvline(1, color="black", linewidth=0.8)
    eje.set_xscale("log")
    marcas = [0.2, 0.3, 0.5, 0.7, 1, 1.5, 2]
    eje.set_xticks(marcas)
    eje.set_xticklabels([f"{m:g}" for m in marcas])
    eje.minorticks_off()
    eje.set_yticks(range(len(datos)))
    eje.set_yticklabels(etiquetas)
    eje.set_xlabel("Razón de odds de Churn=1 (escala logarítmica)")
    eje.grid(axis="y", visible=False)
    for posicion, valor in enumerate(datos["Razón de odds"]):
        eje.annotate(f"{valor:.2f}", (valor, posicion), xytext=(6 if valor > 1 else -6, 0), textcoords="offset points",
                     va="center", ha="left" if valor > 1 else "right", fontsize=7.5)
    eje.set_xlim(min(0.14, 0.8 * datos["Razón de odds"].min()), max(2.6, 1.3 * datos["Razón de odds"].max()))
    figura.tight_layout()
    return figura


# ---------------------------------------------------------------------------
# Árbol de decisión
# ---------------------------------------------------------------------------
def figura_busqueda_arbol(tabla, elegido):
    """F1 en Train y Validation de las 42 configuraciones del árbol de decisión."""
    figura, ejes = plt.subplots(1, 2, figsize=(ANCHO_PAGINA, 2.75), sharey=True)
    etiquetas_x = ["Sin límite" if p is None else str(p) for p in PROFUNDIDADES]
    posiciones = {("None" if p is None else p): i for i, p in enumerate(PROFUNDIDADES)}
    estilos = {1: ("#D55E00", "o"), 20: ("#0072B2", "s"), 50: ("#009E73", "^")}
    titulos = {"None": "Sin ponderar (class_weight=None)", "balanced": "Ponderado (class_weight='balanced')"}
    for eje, (peso, titulo) in zip(ejes, titulos.items()):
        for minimo, (color, marcador) in estilos.items():
            datos = tabla[(tabla["class_weight"] == peso) & (tabla["min_samples_leaf"] == minimo)].copy()
            datos["x"] = datos["max_depth"].map(posiciones)
            datos = datos.sort_values("x")
            eje.plot(datos["x"], datos["F1-score (Validation)"], color=color, marker=marcador, markersize=4,
                     linewidth=1.6, label=f"Hoja ≥ {minimo} · Validation")
            eje.plot(datos["x"], datos["F1-score (Train)"], color=color, marker=marcador, markersize=3,
                     linewidth=1.0, linestyle="--", alpha=0.75, label=f"Hoja ≥ {minimo} · Train")
        eje.set_xticks(range(len(etiquetas_x)))
        eje.set_xticklabels(etiquetas_x)
        eje.set_xlabel("Profundidad máxima (max_depth)")
        eje.set_title(titulo)
    ejes[0].set_ylabel("F1-score de Churn=1")
    fila = tabla.set_index("Configuración").loc[elegido]
    eje = ejes[0] if fila["class_weight"] == "None" else ejes[1]
    _marcar_seleccion(eje, posiciones[fila["max_depth"]], fila["F1-score (Validation)"], desplazamiento=(-62, -14))
    ejes[0].legend(loc="upper left", frameon=True, ncol=1, fontsize=7.5)
    figura.tight_layout()
    return figura


def _pregunta_positiva(texto, largo_maximo=27):
    """Convierte la condición de la rama izquierda en una pregunta cuya respuesta 'sí' va a la derecha."""
    if " ≠ " in texto:
        variable, categoria = texto.split(" ≠ ")
        pregunta = f"¿{variable} = {categoria}?"
        return pregunta if len(pregunta) <= largo_maximo else f"¿{variable} =\n{categoria}?"
    variable, umbral = texto.split(" ≤ ")
    return f"¿{variable} > {float(umbral):g}?"


def figura_arbol(modelo, X_train, y_train, niveles=3, ancho=ANCHO_PAGINA):
    """Diagrama de los primeros ``niveles`` del árbol con tasas reales de abandono en Train.

    En cada pregunta, la rama izquierda corresponde a "no" y la derecha a "sí".
    El color de cada caja indica la tasa de abandono observada en ese nodo. Las
    cajas del último nivel resumen el subárbol que cuelga de ellas. ``ancho`` es
    el ancho de la figura en pulgadas: con ``ANCHO_COLUMNA`` caben dos niveles.
    """
    estructura = estructura_arbol(modelo, X_train, y_train)
    arbol = modelo.named_steps[PASO_MODELO].tree_

    def hojas_bajo(nodo):
        if arbol.children_left[nodo] == -1:
            return 1
        return hojas_bajo(arbol.children_left[nodo]) + hojas_bajo(arbol.children_right[nodo])

    posicion, contador = {}, [0]

    def ubicar(nodo):
        fila = estructura.loc[nodo]
        if fila["Es hoja"] or fila["Profundidad"] == niveles:
            posicion[nodo] = float(contador[0])
            contador[0] += 1
        else:
            izquierda, derecha = int(fila["Hijo izquierdo"]), int(fila["Hijo derecho"])
            ubicar(izquierda)
            ubicar(derecha)
            posicion[nodo] = (posicion[izquierda] + posicion[derecha]) / 2

    ubicar(0)
    terminales = contador[0]
    figura, eje = plt.subplots(figsize=(ancho, 0.98 * (niveles + 1) + 0.2))
    mapa = plt.get_cmap("OrRd")
    alto_caja = 0.62
    # Ancho aproximado de un carácter en unidades del eje x (una unidad = una caja terminal).
    unidad_en_pulgadas = (ancho - 0.1) / (terminales + 0.1)
    ancho_caracter = {7.6: 0.058 / unidad_en_pulgadas, 6.9: 0.053 / unidad_en_pulgadas}

    for nodo in posicion:
        fila = estructura.loc[nodo]
        x, y = posicion[nodo], -float(fila["Profundidad"])
        terminal = bool(fila["Es hoja"] or fila["Profundidad"] == niveles)
        tasa = float(fila["Tasa de abandono"])
        if terminal:
            lineas = [f"n = {int(fila['Clientes'])}", "abandono", f"{100 * tasa:.1f} %"]
            lineas.append("(hoja)" if fila["Es hoja"] else f"({hojas_bajo(nodo)} hojas)")
            letra = 6.9
        else:
            lineas = _pregunta_positiva(fila["Pregunta (sí = izquierda)"]).split("\n")
            lineas.append(f"n = {int(fila['Clientes'])} · abandono {100 * tasa:.1f} %")
            letra = 7.6
            for hijo, respuesta in ((int(fila["Hijo izquierdo"]), "No"), (int(fila["Hijo derecho"]), "Sí")):
                xh, yh = posicion[hijo], -float(estructura.loc[hijo, "Profundidad"])
                eje.plot([x, xh], [y - alto_caja / 2, yh + alto_caja / 2], color="#555555", linewidth=0.9, zorder=1)
                eje.text((x + xh) / 2, (y + yh) / 2, respuesta, fontsize=7.5, ha="center", va="center",
                         color="#333333", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))
        ancho_caja = ancho_caracter[letra] * max(len(linea) for linea in lineas) + 0.16
        if terminal:
            ancho_caja = max(ancho_caja, 0.8)
        eje.add_patch(
            FancyBboxPatch(
                (x - ancho_caja / 2, y - alto_caja / 2), ancho_caja, alto_caja,
                boxstyle="round,pad=0.0,rounding_size=0.05", linewidth=0.8, edgecolor="#444444",
                facecolor=mapa(0.06 + 0.7 * tasa), zorder=2,
            )
        )
        eje.text(x, y, "\n".join(lineas), ha="center", va="center", fontsize=letra, zorder=3,
                 color="white" if tasa > 0.66 else "black", linespacing=1.25)

    eje.set_xlim(-0.55, terminales - 0.45)
    eje.set_ylim(-niveles - 0.4, 0.4)
    eje.axis("off")
    figura.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    return figura


def figura_importancias(permutacion_logistica, permutacion_arbol, cantidad=10, metrica="F1-score"):
    """Importancia por permutación de ambos modelos sobre las variables originales."""
    columna, error = f"Caída de {metrica} (media)", f"Caída de {metrica} (DE)"
    a = permutacion_logistica.set_index("Variable")
    b = permutacion_arbol.set_index("Variable")
    orden = pd.concat([a[columna], b[columna]], axis=1).max(axis=1).sort_values(ascending=False).head(cantidad).index[::-1]
    figura, eje = plt.subplots(figsize=(ANCHO_COLUMNA, 0.3 * cantidad + 1.0))
    alto = 0.38
    posiciones = np.arange(len(orden))
    for desplazamiento, datos, nombre in ((alto / 2, a, NOMBRE_LOGISTICA), (-alto / 2, b, NOMBRE_ARBOL)):
        eje.barh(posiciones + desplazamiento, datos.loc[orden, columna], height=alto, xerr=datos.loc[orden, error],
                 color=COLOR_MODELO[nombre], edgecolor="black", linewidth=0.4, label=ABREVIATURA[nombre],
                 error_kw=dict(ecolor="#333333", elinewidth=0.8, capsize=1.5))
    eje.set_yticks(posiciones)
    eje.set_yticklabels(orden)
    eje.axvline(0, color="black", linewidth=0.8)
    eje.set_xlabel(f"Caída de {metrica} al permutar la variable")
    eje.grid(axis="y", visible=False)
    eje.legend(loc="lower right", frameon=True)
    figura.tight_layout()
    return figura


def figura_importancias_arbol(importancias_por_variable, permutacion_arbol, cantidad=10):
    """Compara la importancia MDI (Train) con la permutación (Validation) en el árbol."""
    mdi = importancias_por_variable.set_index("Variable")["Importancia (MDI)"]
    permutacion = permutacion_arbol.set_index("Variable")["Caída de F1-score (media)"]
    orden = mdi.sort_values(ascending=False).head(cantidad).index[::-1]
    figura, ejes = plt.subplots(1, 2, figsize=(ANCHO_PAGINA, 0.26 * cantidad + 1.0), sharey=True)
    ejes[0].barh(orden, mdi.loc[orden], color=COLOR_MODELO[NOMBRE_ARBOL], edgecolor="black", linewidth=0.4)
    ejes[0].set_xlabel("Importancia MDI (Train)")
    ejes[0].set_title("Reducción de impureza")
    ejes[1].barh(orden, permutacion.loc[orden], color="#E69F00", edgecolor="black", linewidth=0.4)
    ejes[1].set_xlabel("Caída de F1-score (Validation)")
    ejes[1].set_title("Permutación")
    for eje in ejes:
        eje.grid(axis="y", visible=False)
    figura.tight_layout()
    return figura


# ---------------------------------------------------------------------------
# Evaluación comparativa
# ---------------------------------------------------------------------------
def figura_metricas(intervalos, titulo=None):
    """Métricas de los tres modelos con su intervalo bootstrap del 95 %."""
    figura, eje = plt.subplots(figsize=(ANCHO_PAGINA, 2.7))
    modelos = list(dict.fromkeys(intervalos["Modelo"]))
    ancho = 0.8 / len(modelos)
    for i, modelo in enumerate(modelos):
        datos = intervalos[intervalos["Modelo"] == modelo].set_index("Métrica").loc[METRICAS]
        x = np.arange(len(METRICAS)) + (i - (len(modelos) - 1) / 2) * ancho
        errores = np.vstack([datos["Estimación"] - datos["IC inferior"], datos["IC superior"] - datos["Estimación"]])
        eje.bar(x, datos["Estimación"], width=ancho, color=COLOR_MODELO[modelo], edgecolor="black", linewidth=0.4,
                yerr=errores, error_kw=dict(ecolor="black", elinewidth=0.9, capsize=2), label=modelo)
        for xi, valor, superior in zip(x, datos["Estimación"], datos["IC superior"]):
            eje.text(xi, superior + 0.015, f"{valor:.3f}", ha="center", va="bottom", fontsize=7.2)
    eje.set_xticks(np.arange(len(METRICAS)))
    eje.set_xticklabels(METRICAS)
    eje.set_ylim(0, 1.08)
    eje.set_ylabel("Valor de la métrica")
    eje.grid(axis="x", visible=False)
    eje.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=len(modelos), frameon=False)
    if titulo:
        eje.set_title(titulo, pad=22)
    figura.tight_layout()
    return figura


def figura_matrices(matrices):
    """Matrices de confusión (conteos y porcentaje respecto de la clase real)."""
    figura, ejes = plt.subplots(1, len(matrices), figsize=(ANCHO_PAGINA, 2.45))
    for eje, (nombre, matriz) in zip(np.atleast_1d(ejes), matrices.items()):
        valores = matriz.to_numpy()
        proporciones = valores / valores.sum(axis=1, keepdims=True)
        eje.imshow(proporciones, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                eje.text(j, i, f"{valores[i, j]}\n({100 * proporciones[i, j]:.1f} %)", ha="center", va="center",
                         fontsize=8.5, color="white" if proporciones[i, j] > 0.55 else "black")
        eje.set_xticks([0, 1])
        eje.set_xticklabels(["No (0)", "Sí (1)"])
        eje.set_yticks([0, 1])
        eje.set_yticklabels(["No (0)", "Sí (1)"])
        eje.set_xlabel("Predicción")
        eje.set_title(nombre, color=COLOR_MODELO.get(nombre, "black"))
        eje.grid(False)
    np.atleast_1d(ejes)[0].set_ylabel("Clase real")
    figura.tight_layout()
    return figura


def figura_roc_pr(y, predicciones):
    """Curvas ROC y Precision-Recall con el punto de operación del umbral 0,5."""
    y = np.asarray(y)
    figura, (eje_roc, eje_pr) = plt.subplots(1, 2, figsize=(ANCHO_PAGINA, 3.0))
    prevalencia = y.mean()
    for nombre, (y_pred, y_proba) in predicciones.items():
        y_pred, y_proba = np.asarray(y_pred), np.asarray(y_proba)
        color = COLOR_MODELO[nombre]
        if np.unique(y_proba).size == 1:  # puntuaciones constantes (Dummy)
            continue
        fpr, tpr, _ = roc_curve(y, y_proba)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        eje_roc.plot(fpr, tpr, color=color, linewidth=1.6, label=nombre)
        eje_pr.plot(recall, precision, color=color, linewidth=1.6, label=nombre)
        vp = ((y == 1) & (y_pred == 1)).sum()
        fp = ((y == 0) & (y_pred == 1)).sum()
        eje_roc.scatter([fp / (y == 0).sum()], [vp / (y == 1).sum()], color=color, edgecolor="black", s=42, zorder=5)
        eje_pr.scatter([vp / (y == 1).sum()], [vp / max(vp + fp, 1)], color=color, edgecolor="black", s=42, zorder=5)
    eje_roc.plot([0, 1], [0, 1], color=COLOR_MODELO[NOMBRE_DUMMY], linestyle="--", linewidth=1.2, label="Dummy / azar")
    eje_pr.axhline(prevalencia, color=COLOR_MODELO[NOMBRE_DUMMY], linestyle="--", linewidth=1.2,
                   label=f"Tasa de abandono ({prevalencia:.3f})")
    eje_roc.scatter([], [], color="white", edgecolor="black", s=42, label=f"Umbral {UMBRAL:g}".replace(".", ","))
    eje_roc.set_xlabel("Tasa de falsos positivos")
    eje_roc.set_ylabel("Recall (tasa de verdaderos positivos)")
    eje_roc.set_title("Curva ROC")
    eje_pr.set_xlabel("Recall")
    eje_pr.set_ylabel("Precision")
    eje_pr.set_title("Curva Precision-Recall")
    for eje in (eje_roc, eje_pr):
        eje.set_xlim(-0.02, 1.02)
        eje.set_ylim(-0.02, 1.02)
        eje.set_aspect("equal")
    eje_roc.legend(loc="lower right", frameon=True)
    eje_pr.legend(loc="lower left", frameon=True)
    figura.tight_layout()
    return figura


def _partir(texto):
    """Parte en dos líneas las etiquetas largas de los grupos."""
    texto = str(texto)
    if len(texto) <= 8:
        return texto
    for separador in ("-to-", " "):
        if separador in texto:
            inicio, fin = texto.split(separador, 1)
            return f"{inicio}{separador.rstrip()}\n{fin}"
    return texto


def figura_errores_subgrupo(tabla, variables=None):
    """Recall y tasa de falsos positivos de cada modelo por grupo de clientes."""
    variables = variables or list(dict.fromkeys(tabla["Variable"]))
    modelos = list(dict.fromkeys(tabla["Modelo"]))
    anchos = [max(2, tabla[tabla["Variable"] == v]["Grupo"].nunique()) for v in variables]
    figura, ejes = plt.subplots(2, len(variables), figsize=(ANCHO_PAGINA, 3.9), sharey="row",
                                gridspec_kw=dict(width_ratios=anchos))
    metricas = [("Recall", "Recall\n(abandonos detectados)"), ("Tasa de falsos positivos", "Tasa de falsos\npositivos")]
    for columna, variable in enumerate(variables):
        datos = tabla[tabla["Variable"] == variable]
        grupos = list(dict.fromkeys(datos["Grupo"]))
        for fila, (metrica, etiqueta) in enumerate(metricas):
            eje = ejes[fila, columna]
            ancho = 0.8 / len(modelos)
            for i, modelo in enumerate(modelos):
                valores = datos[datos["Modelo"] == modelo].set_index("Grupo").loc[grupos, metrica]
                x = np.arange(len(grupos)) + (i - (len(modelos) - 1) / 2) * ancho
                eje.bar(x, valores, width=ancho, color=COLOR_MODELO[modelo], edgecolor="black", linewidth=0.4,
                        label=ABREVIATURA[modelo])
            eje.set_xticks(np.arange(len(grupos)))
            eje.set_ylim(0, 1.0)
            eje.grid(axis="x", visible=False)
            if fila == 0:
                eje.set_title(variable)
                eje.set_xticklabels([])
            else:
                resumen = datos.drop_duplicates("Grupo").set_index("Grupo").loc[grupos]
                eje.set_xticklabels(
                    [f"{_partir(g)}\n(n={int(n)})" for g, n in zip(grupos, resumen["Clientes"])], fontsize=7
                )
            if columna == 0:
                eje.set_ylabel(etiqueta)
    manijas, etiquetas = ejes[0, 0].get_legend_handles_labels()
    figura.tight_layout()
    figura.legend(manijas, etiquetas, loc="lower center", bbox_to_anchor=(0.5, 0.995), ncol=len(modelos), frameon=False)
    return figura


def figura_probabilidades(y, predicciones):
    """Distribución de la probabilidad estimada de abandono según la clase real."""
    y = np.asarray(y)
    nombres = [n for n, (_, p) in predicciones.items() if np.unique(np.asarray(p)).size > 1]
    figura, ejes = plt.subplots(1, len(nombres), figsize=(ANCHO_PAGINA, 2.6), sharey=True)
    cortes = np.linspace(0, 1, 21)
    for eje, nombre in zip(np.atleast_1d(ejes), nombres):
        probabilidad = np.asarray(predicciones[nombre][1])
        eje.hist(probabilidad[y == 0], bins=cortes, color=COLOR_NO, alpha=0.8, edgecolor="white", linewidth=0.4,
                 label="Churn = No")
        eje.hist(probabilidad[y == 1], bins=cortes, color=COLOR_SI, alpha=0.8, edgecolor="white", linewidth=0.4,
                 label="Churn = Yes")
        eje.axvline(UMBRAL, color="black", linestyle="--", linewidth=1.0)
        eje.set_title(nombre, color=COLOR_MODELO[nombre])
        eje.set_xlabel("Probabilidad estimada de Churn=1")
    np.atleast_1d(ejes)[0].set_ylabel("Clientes")
    np.atleast_1d(ejes)[0].legend(loc="upper right", frameon=True)
    figura.tight_layout()
    return figura
