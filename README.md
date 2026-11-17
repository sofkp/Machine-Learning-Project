# Análisis y clasificación del abandono de clientes de telecomunicaciones

Proyecto final del curso **Machine Learning** — Universidad de Ingeniería y Tecnología (UTEC), 2026-2.

## Descripción y objetivo

El abandono de clientes (*customer churn*) representa un problema de interés para las empresas de telecomunicaciones, pues puede afectar la continuidad de los ingresos y orientar decisiones de retención. Este proyecto estudia la relación entre las características de los clientes y el abandono registrado, y compara modelos de clasificación binaria para identificar a quienes abandonan el servicio.

El trabajo comprende dos etapas:

- **P1 — Formulación del problema y análisis exploratorio (EDA):** caracterización de variables, distribución del abandono, calidad de los datos, asociaciones y limitaciones.
- **P2 — Preprocesamiento, modelado y evaluación:** preparación reproducible de las características, definición de un modelo de referencia (*baseline*), entrenamiento y comparación de regresión logística y árbol de decisión, y análisis de sus resultados.

**Objetivo general:** analizar el abandono de clientes mediante técnicas de análisis exploratorio y aprendizaje automático supervisado, identificando patrones asociados al abandono y comparando el desempeño e interpretabilidad de modelos de clasificación.

Los datos corresponden a un **conjunto público de ejemplo de una empresa ficticia**. Los resultados son válidos para el ejercicio experimental planteado, pero no demuestran por sí solos capacidad de predicción prospectiva ni generalización a empresas reales.

## Integrantes

- Sofia Valentina Ku Paredes
- Farid Espinoza
- Sebastian Chu
- Marcia Arce
- Alessandra Lopez-Ameri Castro

## Datos

Se utiliza el conjunto **Telco Customer Churn**, disponible en [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). El CSV empleado se encuentra en:

`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Características principales:

- **7 043 registros** y **21 columnas** en el archivo original.
- Variable objetivo: `Churn` (`No` = 0; `Yes` = 1).
- **19 características predictoras**, una vez excluidos `customerID` y `Churn`: tres numéricas y dieciséis categóricas.
- Distribución del objetivo: **5 174 clientes sin abandono (73,46 %)** y **1 869 con abandono (26,54 %)**.
- `TotalCharges` contiene **11 valores vacíos** que pasan a faltantes al convertir la columna a formato numérico. Todos corresponden a registros con `tenure = 0`.

El archivo `preprocessing.py` permite cargar este CSV desde una ruta local y, si no se especifica ninguna, desde la copia pública del repositorio. Para la imputación de `TotalCharges` se adopta el valor 0 como **supuesto de modelado** para dichos registros; no se afirma que el importe real haya sido verificado.

## Estructura del repositorio

```text
Machine-Learning-Project/
├── README.md
├── requirements.txt
├── preprocessing.py          # Carga de datos, particiones y preprocesadores compartidos
├── ejecutar_p2.py            # Ejecuta P03, P04 y P05 en orden desde la terminal
├── .gitignore
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── docs/                     # Guía del proyecto y discusión de resultados (P06)
├── notebooks/
│   ├── 01_eda_telco_churn.ipynb
│   ├── P01Pre_procesamiento_actualizado.ipynb
│   ├── P02_baseline_diseno_experimental.ipynb
│   ├── P03_regresion_logistica.ipynb
│   ├── P04_arbol_decision.ipynb
│   └── P05_evaluacion_comparativa.ipynb
├── src/                      # Código reutilizado por P03, P04 y P05
│   ├── protocolo.py          # Particiones, métricas y búsqueda en Validation
│   ├── modelos.py            # Pipelines y grillas de hiperparámetros
│   ├── analisis.py           # Interpretación, bootstrap y análisis de errores
│   └── graficos.py           # Figuras
├── resultados/               # Tablas generadas por los notebooks (CSV, JSON y TXT)
│   └── figuras/              # Figuras generadas por P03, P04 y P05
└── paper/                    # Informe académico en LaTeX (P1 y P2)
    ├── main.tex
    ├── sections/
    ├── figures/
    └── generar_figuras.py    # Figuras de P2 al tamaño del formato IEEE
```

Los notebooks P03, P04 y P05 contienen la **regresión logística**, el **árbol de decisión** y la **evaluación comparativa**. Los tres importan `preprocessing.py` y el código de `src/`, de modo que usan las mismas particiones, métricas y criterios definidos en P02. Sus tablas se guardan en `resultados/` y sus figuras en `resultados/figuras/`. La carpeta `docs/` reúne la guía del proyecto y la discusión de resultados (`P06_discusion_limitaciones_conclusiones.md`); `paper/` contiene el informe académico.

## Requisitos del entorno

- Python **3.10 o superior** recomendado.
- Google Colab, JupyterLab o un entorno equivalente compatible con notebooks.

Las dependencias del proyecto se especifican en `requirements.txt`: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn` e `ipython`, además de `nbformat`, `nbclient` e `ipykernel`, que se usan para ejecutar los notebooks desde la terminal. Para instalarlas localmente:

```bash
python -m pip install -r requirements.txt
```

Si se necesita una interfaz local para abrir los notebooks:

```bash
python -m pip install jupyterlab
```

El preprocesamiento utiliza `OneHotEncoder(sparse_output=False)`, por lo que requiere una versión compatible de **scikit-learn (1.2 o posterior)**. Las dependencias declaradas son versiones mínimas; para una reproducción estricta entre entornos conviene registrar también las versiones exactas utilizadas.

**Versiones usadas en P03, P04 y P05.** Los resultados guardados en el repositorio se generaron con Python 3.12.13, numpy 2.4.6, pandas 2.2.3 y scikit-learn 1.6.1. Cada notebook imprime sus versiones y quedan registradas en `resultados/configuracion_*.json`. Como comprobación, los tres notebooks se ejecutaron también con scikit-learn 1.9.1 (numpy 2.5.5 y pandas 3.1.1): se seleccionan las mismas configuraciones, las predicciones de clase sobre Test son idénticas y las métricas de Test coinciden. Solo cambian detalles en árboles con divisiones empatadas (por ejemplo, el árbol sin restricciones queda con 855 hojas en lugar de 859), que no afectan al modelo elegido.

## Ejecución con Google Colab

1. Abrir [Google Colab](https://colab.research.google.com/).
2. Elegir **Archivo → Abrir cuaderno → GitHub** y buscar o pegar `https://github.com/sofkp/Machine-Learning-Project`.
3. Abrir el notebook que se desea ejecutar desde `notebooks/`.
4. Seleccionar **Entorno de ejecución → Ejecutar todas**.

**Notebooks de P2:**

- `P01Pre_procesamiento_actualizado.ipynb` descarga el CSV público directamente desde GitHub para revisar las transformaciones y sus verificaciones.
- `P02_baseline_diseno_experimental.ipynb` busca `preprocessing.py` en el entorno. Si está abierto en Colab y no encuentra el módulo, intenta clonar automáticamente este repositorio en `/content/Machine-Learning-Project/`. Por tanto, es necesario que el repositorio sea público y que `preprocessing.py` esté en su raíz.
- `P03_regresion_logistica.ipynb`, `P04_arbol_decision.ipynb` y `P05_evaluacion_comparativa.ipynb` localizan el repositorio con la misma lógica que P02 y, en Colab, lo clonan si no lo encuentran, porque necesitan `preprocessing.py` y la carpeta `src/`. P05 lee las configuraciones elegidas en `resultados/configuracion_*.json`: usa las del repositorio, salvo que P03 y P04 se hayan ejecutado antes en la misma sesión.

**Importante:** los archivos generados en la sesión de Colab son temporales. Si se generan nuevos resultados, deben descargarse o incorporarse explícitamente al repositorio; ejecutarlos en Colab no actualiza GitHub de forma automática.

## Ejecución local con Jupyter

Clonar el repositorio y preparar el entorno:

```bash
git clone https://github.com/sofkp/Machine-Learning-Project.git
cd Machine-Learning-Project
python -m pip install -r requirements.txt
python -m pip install jupyterlab
jupyter lab
```

Abrir los notebooks en la carpeta `notebooks/` y ejecutar las celdas en orden. Se recomienda seguir esta secuencia:

1. `01_eda_telco_churn.ipynb`: análisis exploratorio de P1.
2. `P01Pre_procesamiento_actualizado.ipynb`: revisión de datos, particiones y transformaciones.
3. `P02_baseline_diseno_experimental.ipynb`: ejecución del baseline y exportación de métricas de validación.
4. `P03_regresion_logistica.ipynb`: búsqueda de hiperparámetros en Validation, regularización, tratamiento del desbalance e interpretación del modelo.
5. `P04_arbol_decision.ipynb`: búsqueda de hiperparámetros en Validation, control del sobreajuste, tratamiento del desbalance e interpretación del árbol.
6. `P05_evaluacion_comparativa.ipynb`: evaluación única de los tres modelos sobre Test, intervalos de confianza y análisis de errores.

P03 y P04 no usan Test. P05 debe ejecutarse al final, porque lee las configuraciones que guardan los dos anteriores.

Los tres notebooks de modelado también se pueden ejecutar desde la terminal, en ese orden y con un kernel nuevo para cada uno:

```bash
python ejecutar_p2.py              # P03 -> P04 -> P05; guarda tablas y figuras en resultados/
python ejecutar_p2.py P03          # solo el notebook indicado
python paper/generar_figuras.py    # regenera en paper/figures/ las figuras de P2 del informe
```

**Ruta del CSV en el EDA original:** el archivo se encuentra en `data/`, **no** en la raíz del repositorio. Si el notebook `01_eda_telco_churn.ipynb` utiliza todavía una ruta antigua, debe actualizarse antes de su ejecución. Desde `notebooks/`, la ruta relativa al CSV es `../data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (cuando el directorio de trabajo coincide con esa carpeta); desde la raíz, es `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

Para P02 no es necesario modificar la ubicación del CSV: el notebook prioriza `data/` dentro del repositorio y utiliza la URL pública configurada en `preprocessing.py` si no encuentra el archivo local.

## Protocolo experimental de P2

El archivo `preprocessing.py` centraliza las funciones compartidas:

- `load_data()`: lectura del CSV, conversión de `TotalCharges` y codificación de `Churn`.
- `split_data(X, y)`: división estratificada en **Train (60 %, 4 225 clientes)**, **Validation (20 %, 1 409)** y **Test (20 %, 1 409)**, con `random_state=42`.
- `make_preprocessor("logistic")`: imputación de `TotalCharges` con cero, codificación *one-hot* de variables categóricas y estandarización de las variables numéricas.
- `make_preprocessor("tree")`: imputación de `TotalCharges` con cero y codificación *one-hot*, sin estandarización de las variables numéricas.

Los preprocesadores se incorporan en un `Pipeline` junto con cada clasificador y se ajustan **solo con Train**, evitando aprender transformaciones de Validation o Test. El Dummy Classifier no requiere transformar las características.

La selección de hiperparámetros utiliza un único conjunto fijo de **Validation (*hold-out*)**, sin validación cruzada K-Fold. Se comparan, entre otras configuraciones, `class_weight=None` y `class_weight="balanced"`. El criterio principal es el **F1-score de `Churn=1`**. Se reportan además Accuracy, Precision, Recall, ROC-AUC y matrices de confusión. El umbral binario se mantiene en **0,5**, sin optimización posterior.

Una vez seleccionados los hiperparámetros, se conservan los modelos ajustados con Train; **Test** queda reservado para evaluar al final el Dummy, la regresión logística y el árbol de decisión sobre exactamente las mismas observaciones. No se reajustan configuraciones después de consultar Test. Los detalles completos y la regla de desempate se documentan en `P02_baseline_diseno_experimental.ipynb`.

## Resultados disponibles

### Baseline: Dummy Classifier (`strategy="most_frequent"`)

Resultados obtenidos en **Validation**:

| Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0,7346 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |

Matriz de confusión (filas = clase real; columnas = clase predicha):

| | Predicho No (0) | Predicho Sí (1) |
| --- | ---: | ---: |
| Real No (0) | 1 035 | 0 |
| Real Sí (1) | 374 | 0 |

El baseline predice siempre la clase mayoritaria: alcanza aproximadamente **73,46 % de Accuracy**, pero **no detecta abandonos** (Recall y F1-score iguales a 0). El ROC-AUC es 0,5 porque asigna una puntuación constante a la clase positiva. Estos valores corresponden exclusivamente a Validation, **no a Test**.

Los archivos generados son:

- `resultados/resultados_baseline_validacion.csv`
- `resultados/matriz_confusion_baseline_validacion.csv`

### Regresión logística y árbol de decisión (selección en Validation)

Se evaluaron 36 configuraciones de regresión logística (penalización L1 o L2, nueve valores de `C` y `class_weight`) y 42 de árbol de decisión (`max_depth`, `min_samples_leaf` y `class_weight`). Cada una se ajustó con Train y se midió en **Validation**. Configuraciones con mayor F1-score de `Churn=1`:

| Modelo | Configuración seleccionada | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | L2, `C=0.1`, `class_weight="balanced"` | 0,7480 | 0,5166 | 0,7914 | 0,6251 | 0,8360 |
| Árbol de decisión | `max_depth=None`, `min_samples_leaf=20`, `class_weight="balanced"` | 0,7573 | 0,5305 | 0,7433 | 0,6192 | 0,8191 |

- **Regularización:** con L2 y ponderación de clases, el F1-score varía entre 0,610 y 0,625 en todo el rango de `C`; la fuerza de la regularización influye poco.
- **Desbalance:** sin `class_weight="balanced"`, el mejor F1-score es 0,593 en la regresión logística y 0,579 en el árbol. La ponderación aumenta el Recall a cambio de más falsas alarmas.
- **Sobreajuste del árbol:** sin restricciones, el árbol alcanza F1-score de 0,996 en Train y 0,496 en Validation (859 hojas). El árbol seleccionado tiene 132 hojas y 12 niveles; uno de solo 8 hojas obtiene 0,608.

### Evaluación final en Test

Los tres modelos se evaluaron **una sola vez** sobre Test (1 409 clientes, 374 abandonos), con las configuraciones ya cerradas:

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy Classifier | 0,7346 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |
| Regresión logística | 0,7431 | 0,5105 | 0,7807 | 0,6173 | 0,8415 |
| Árbol de decisión | 0,7438 | 0,5123 | 0,7219 | 0,5993 | 0,8184 |

Matrices de confusión en Test (VN: verdaderos negativos; FP: falsos positivos; FN: falsos negativos; VP: verdaderos positivos):

| Modelo | VN | FP | FN | VP |
| --- | ---: | ---: | ---: | ---: |
| Dummy Classifier | 1 035 | 0 | 374 | 0 |
| Regresión logística | 755 | 280 | 82 | 292 |
| Árbol de decisión | 778 | 257 | 104 | 270 |

Lectura de los resultados (intervalos bootstrap del 95 %, 2 000 remuestreos):

- El baseline no detecta ningún abandono. La regresión logística detecta 292 de 374 y el árbol, 270. En ambos, cerca de la mitad de las alertas corresponde a clientes que realmente abandonan.
- La regresión logística supera al árbol en **Recall** (+0,059; intervalo [0,016; 0,103]) y en **ROC-AUC** (+0,023; [0,009; 0,040]). En **F1-score** la diferencia es de +0,018, con intervalo [−0,009; 0,049]: no es concluyente.
- El Accuracy de los dos modelos no se distingue del baseline (+0,009), porque cambian aciertos sobre clientes que permanecen por aciertos sobre clientes que abandonan. Por eso no se usa como criterio.
- `Contract`, `tenure` e `InternetService` son las variables más importantes en ambos modelos.
- Las probabilidades no están calibradas, como consecuencia de la ponderación de clases: sirven para ordenar clientes por riesgo, no como probabilidad real de abandono.

Archivos generados en `resultados/`:

- Selección: `resultados_*_grilla_validacion.csv` (todas las configuraciones), `resultados_*_validacion.csv` y `configuracion_*.json` (configuración elegida, métricas y versiones).
- Interpretación: `razones_de_odds_regresion_logistica.csv`, `coeficientes_regresion_logistica.csv`, `reglas_arbol_decision.txt`, `hojas_arbol_decision.csv`, `importancias_arbol_decision.csv` e `importancia_permutacion_*_validacion.csv`.
- Evaluación final: `comparacion_modelos_validacion_test.csv`, `intervalos_bootstrap_test.csv`, `diferencias_bootstrap_test.csv`, `matriz_confusion_*_test.csv` y `predicciones_test.csv`.
- Análisis de errores: `errores_por_subgrupo_test.csv`, `coincidencia_errores_test.csv` y `perfil_errores_test.csv`.
- Figuras: `figuras/p03_*.png`, `figuras/p04_*.png` y `figuras/p05_*.png`.

La interpretación completa, las limitaciones y las conclusiones están en `docs/P06_discusion_limitaciones_conclusiones.md` y en el informe (`paper/`).

## Limitaciones y alcance

- Los datos son un ejemplo de una empresa ficticia y no necesariamente representan a una empresa o población específica.
- Las asociaciones entre características y abandono no demuestran causalidad.
- La información no incluye un historial temporal suficiente para verificar que todas las variables estuvieran disponibles antes del abandono; por ello, no puede descartarse totalmente una fuga temporal en el origen de los datos.
- Una única partición Train/Validation/Test facilita la reproducibilidad, pero las diferencias pequeñas entre modelos pueden depender de la muestra utilizada.
- El F1-score no constituye una estimación de costos empresariales, ya que no se dispone de costos conocidos de falsos positivos o negativos.
- Las probabilidades estimadas con `class_weight="balanced"` no están calibradas y el umbral de 0,5 no se ajustó.
- Los errores no se reparten por igual entre grupos de clientes: por ejemplo, los adultos mayores que permanecen reciben más falsas alarmas. El detalle está en `docs/P06_discusion_limitaciones_conclusiones.md`.

## Referencias y documentación
- [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- Guía del proyecto e informes disponibles en las carpetas `docs/` y `paper/`.
