"""Ejecuta, en orden, los notebooks de la etapa P2 desde un entorno limpio.

Uso, desde la raíz del repositorio:

    python ejecutar_p2.py              # P02 -> P03 -> P04 -> P05 -> P06
    python ejecutar_p2.py P03 P04      # solo los notebooks indicados (requiere P02 antes)

Cada notebook se ejecuta de principio a fin en un kernel nuevo, con ``notebooks/``
como carpeta de trabajo, y se guarda con sus salidas. Las tablas y figuras se
escriben en ``resultados/``.

El orden importa: P05 lee las configuraciones que P03 y P04 guardan en
``resultados/configuracion_*.json`` y es el único notebook que usa Test.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
CARPETA = ROOT / "notebooks"

NOTEBOOKS = {
    "P02": "P02_baseline_diseno_experimental.ipynb",
    "P03": "P03_regresion_logistica.ipynb",
    "P04": "P04_arbol_decision.ipynb",
    "P05": "P05_evaluacion_comparativa.ipynb",
    "P06": "P06_discusion_conclusiones.ipynb",
}


def ejecutar(nombre):
    """Ejecuta un notebook completo y lo sobrescribe con las salidas nuevas."""
    ruta = CARPETA / nombre
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el notebook requerido: {ruta}")
    notebook = nbformat.read(ruta, as_version=4)
    inicio = time.time()
    cliente = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(CARPETA)}},
    )
    cliente.execute()  # se detiene con un error si alguna celda falla
    nbformat.write(notebook, ruta)
    print(f"  {nombre}: ejecutado en {time.time() - inicio:.0f} s")


def main(argumentos):
    claves = [argumento.upper() for argumento in argumentos] or list(NOTEBOOKS)
    desconocidas = [clave for clave in claves if clave not in NOTEBOOKS]
    if desconocidas:
        raise SystemExit(f"Notebooks no reconocidos: {desconocidas}. Opciones: {list(NOTEBOOKS)}")

    # Se respeta siempre el orden P02 -> P03 -> P04 -> P05 -> P06.
    resultados = ROOT / "resultados"
    dependencias = {
        "P03": ["resultados_baseline_validacion.csv"],
        "P04": ["resultados_baseline_validacion.csv"],
        "P05": ["configuracion_regresion_logistica.json", "configuracion_arbol_decision.json",
                "resultados_baseline_validacion.csv"],
        "P06": ["comparacion_modelos_validacion_test.csv", "predicciones_test.csv",
                "diferencias_bootstrap_test.csv"],
    }
    print("Ejecutando notebooks de P2:")
    for clave in NOTEBOOKS:
        if clave in claves:
            faltan = [nombre for nombre in dependencias.get(clave, [])
                      if not (resultados / nombre).is_file()]
            if faltan:
                raise FileNotFoundError(
                    f"{clave} requiere estos artefactos previos en resultados/: {faltan}. "
                    "Ejecuta el flujo completo con python ejecutar_p2.py"
                )
            ejecutar(NOTEBOOKS[clave])
    print("Listo. Resultados en resultados/ y figuras en resultados/figuras/.")


if __name__ == "__main__":
    main(sys.argv[1:])
