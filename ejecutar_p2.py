"""Ejecuta, en orden, los notebooks de modelado y evaluación de P2.

Uso, desde la raíz del repositorio:

    python ejecutar_p2.py              # P03 -> P04 -> P05
    python ejecutar_p2.py P03 P04      # solo los notebooks indicados

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
    "P03": "P03_regresion_logistica.ipynb",
    "P04": "P04_arbol_decision.ipynb",
    "P05": "P05_evaluacion_comparativa.ipynb",
}


def ejecutar(nombre):
    """Ejecuta un notebook completo y lo sobrescribe con las salidas nuevas."""
    ruta = CARPETA / nombre
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

    # Se respeta siempre el orden P03 -> P04 -> P05, sin importar cómo se pidan.
    print("Ejecutando notebooks de P2:")
    for clave in NOTEBOOKS:
        if clave in claves:
            ejecutar(NOTEBOOKS[clave])
    print("Listo. Resultados en resultados/ y figuras en resultados/figuras/.")


if __name__ == "__main__":
    main(sys.argv[1:])
