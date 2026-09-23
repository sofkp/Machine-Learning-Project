# Análisis de abandono de clientes de telecomunicaciones

Proyecto del curso de Machine Learning. En esta etapa se formula el problema de abandono de clientes (*churn*), se describe el dataset y se realiza un análisis exploratorio de datos (EDA).

## Objetivo

Explorar las características de los clientes y sus servicios para identificar patrones asociados al abandono. Los hallazgos de esta etapa servirán como base para preparar los datos y desarrollar modelos predictivos en las siguientes fases del proyecto.

## Integrantes

- Sofia Valentina Ku Paredes
- Farid Espinoza
- Sebastian Chu
- Marcia Sofia Arce Montesinos
- Alessandra Lopez-Ameri Castro

## Estructura del repositorio

```text
Machine-Learning-Project/
├── README.md
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── docs/
│   └── Guia_Proyecto_Final_Machine_Learning_ES.pdf
├── notebooks/
│   └── 01_eda_telco_churn.ipynb
└── paper/
    └── Grupo10_informe_P1.pdf
    └── ...
```

- **`WA_Fn-UseC_-Telco-Customer-Churn.csv`**: datos utilizados en el análisis.
- **`01_eda_telco_churn.ipynb`**: código, gráficos y resultados del EDA.
- **`Grupo10_informe_P1.pdf`**: informe de la etapa P1.

## Dataset

El archivo `WA_Fn-UseC_-Telco-Customer-Churn.csv` está incluido en este repositorio. También se puede obtener desde [Telco Customer Churn en Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

Si se prefiere descargarlo desde el propio notebook, se pueden seguir las instrucciones de Kaggle para utilizar `kagglehub` y ajustar la celda de carga de datos para que lea el archivo descargado. En cualquiera de los casos, el notebook debe apuntar a la ubicación real del CSV antes de ejecutarse.

## Entorno y dependencias

Se recomienda utilizar Python 3.10 o superior y un entorno que permita abrir notebooks de Jupyter, como JupyterLab o Visual Studio Code. El análisis utiliza `pandas`, `numpy`, `matplotlib` y `seaborn`.

Para instalar estas dependencias:

```bash
python -m pip install pandas numpy matplotlib seaborn jupyter ipykernel
```

También el el propio notebook se puede crear un nuevo bloque para instalar las dependencias:

```bash
%pip install pandas numpy matplotlib seaborn jupyter ipykernel
```

## Ejecución del análisis

1. Clona el repositorio o descarga sus archivos:

   ```bash
   git clone https://github.com/sofkp/Machine-Learning-Project.git
   cd Machine-Learning-Project
   ```

2. Abre `notebooks/01_eda_telco_churn.ipynb` en Jupyter o Visual Studio Code.

3. Selecciona un **kernel de Python** que tenga instaladas las dependencias indicadas arriba o instalarlas en el propio notebook antes de correr el código.

4. Verifica que la celda que carga el dataset apunte al archivo CSV. Como el notebook está dentro de `notebooks/` y el CSV está en la raíz del repositorio, la ruta relativa sería:

   ```python
   ../WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

5. Ejecuta el notebook de principio a fin con **«Ejecutar todo»**. También puedes ejecutar las celdas **una por una y en orden** si deseas seguir cada paso del análisis.

Los resultados, tablas y visualizaciones se muestran en el propio notebook. El informe en `paper/` presenta la formulación del problema y los principales hallazgos de esta etapa.
