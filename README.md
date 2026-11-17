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
├── preprocessing.py
├── .gitignore
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── docs/                     # Guía y documentación del proyecto
├── notebooks/
│   ├── 01_eda_telco_churn.ipynb
│   ├── P01Pre_procesamiento_actualizado.ipynb
│   └── P02_baseline_diseno_experimental.ipynb
├── resultados/
│   ├── resultados_baseline_validacion.csv
│   └── matriz_confusion_baseline_validacion.csv
└── paper/                    # Informes académicos de P1 y P2
```

Los notebooks de **regresión logística**, **árbol de decisión** y **comparación final** se incorporarán a `notebooks/` cuando estén terminados y revisados. Sus resultados se añadirán a `resultados/`. La carpeta `docs/` reúne material de documentación y `paper/` contiene los informes académicos.

## Requisitos del entorno

- Python **3.10 o superior** recomendado.
- Google Colab, JupyterLab o un entorno equivalente compatible con notebooks.

Las dependencias del proyecto se especifican en `requirements.txt`: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn` e `ipython`. Para instalarlas localmente:

```bash
python -m pip install -r requirements.txt
```

Si se necesita una interfaz local para abrir los notebooks:

```bash
python -m pip install jupyterlab
```

El preprocesamiento utiliza `OneHotEncoder(sparse_output=False)`, por lo que requiere una versión compatible de **scikit-learn (1.2 o posterior)**. Las dependencias declaradas son versiones mínimas; para una reproducción estricta entre entornos conviene registrar también las versiones exactas utilizadas.

## Ejecución con Google Colab

1. Abrir [Google Colab](https://colab.research.google.com/).
2. Elegir **Archivo → Abrir cuaderno → GitHub** y buscar o pegar `https://github.com/sofkp/Machine-Learning-Project`.
3. Abrir el notebook que se desea ejecutar desde `notebooks/`.
4. Seleccionar **Entorno de ejecución → Ejecutar todas**.

**Notebooks de P2:**

- `P01Pre_procesamiento_actualizado.ipynb` descarga el CSV público directamente desde GitHub para revisar las transformaciones y sus verificaciones.
- `P02_baseline_diseno_experimental.ipynb` busca `preprocessing.py` en el entorno. Si está abierto en Colab y no encuentra el módulo, intenta clonar automáticamente este repositorio en `/content/Machine-Learning-Project/`. Por tanto, es necesario que el repositorio sea público y que `preprocessing.py` esté en su raíz.

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
4. Notebooks de regresión logística y árbol de decisión.
5. Comparación final de modelos sobre Test.

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

Los resultados de regresión logística, árbol de decisión y evaluación final sobre Test **aún no se incluyen en esta versión del README**. 

## Limitaciones y alcance

- Los datos son un ejemplo de una empresa ficticia y no necesariamente representan a una empresa o población específica.
- Las asociaciones entre características y abandono no demuestran causalidad.
- La información no incluye un historial temporal suficiente para verificar que todas las variables estuvieran disponibles antes del abandono; por ello, no puede descartarse totalmente una fuga temporal en el origen de los datos.
- Una única partición Train/Validation/Test facilita la reproducibilidad, pero las diferencias pequeñas entre modelos pueden depender de la muestra utilizada.
- El F1-score no constituye una estimación de costos empresariales, ya que no se dispone de costos conocidos de falsos positivos o negativos.

## Referencias y documentación
- [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- Guía del proyecto e informes disponibles en las carpetas `docs/` y `paper/`.


- [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- Guía del proyecto e informes disponibles en las carpetas `docs/` y `paper/`.
