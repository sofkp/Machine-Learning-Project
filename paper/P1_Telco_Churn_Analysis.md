# Proyecto 1: Predicción de churn en clientes de telecomunicaciones

## 1. Resumen

Este proyecto aborda el problema de churn en clientes de telecomunicaciones, una situación crítica para empresas del sector porque impacta directamente la retención, la rentabilidad y los costos asociados a captar nuevos clientes. El objetivo principal es analizar el conjunto de datos Telco Customer Churn para comprender los factores asociados con la salida de clientes y preparar la base para un modelo predictivo de clasificación binaria.

El dataset utilizado corresponde a registros de clientes con información demográfica, servicios contratados, uso, facturación y una variable objetivo llamada `Churn`, que indica si el cliente abandonó el servicio (`Yes`) o no (`No`). El análisis exploratorio de datos (EDA) muestra que el problema no es balanceado, que la duración del contrato y el tipo de servicio de internet parecen estar fuertemente relacionados con la deserción, y que la variable `TotalCharges` requiere limpieza y tratamiento previo antes de cualquier modelado.

## 2. Introducción y formulación del problema

La competencia en el mercado de telecomunicaciones es alta y la retención de clientes es un componente estratégico para el negocio. El churn representa la pérdida de clientes y suele traducirse en una reducción de ingresos, mayores costos de adquisición y menor estabilidad del portafolio de clientes. En este contexto, entender por qué un cliente abandona un servicio puede ayudar a diseñar políticas de retención, promociones, mejor servicio o ajustes de planes.

El problema de este proyecto se formaliza como una tarea de clasificación supervisada binaria:

- Entrada: variables relacionadas con el cliente, su historial de servicio, servicios adicionales, contrato y facturación.
- Salida: `Churn` con dos clases: `Yes` (abandona) y `No` (permanece).

Las preguntas de investigación del proyecto son las siguientes:

1. ¿Qué características están más asociadas con la deserción de clientes?
2. ¿El tipo de contrato y la duración del servicio influyen significativamente en la tasa de churn?
3. ¿Los clientes con servicios adicionales o planes más estables presentan menor probabilidad de abandono?
4. ¿Existen problemas de calidad de datos o variables potencialmente contaminadas que deban atenderse antes del modelado?

## 3. Datos y análisis exploratorio de datos (EDA)

### 3.1 Dataset

El conjunto de datos utilizado es `WA_Fn-UseC_-Telco-Customer-Churn.csv`. Se trata de un dataset público de clientes de telecomunicaciones con registros de 7,043 clientes y 21 columnas en total.

La variable objetivo es:

- `Churn`: indica si el cliente abandonó el servicio (`Yes`) o no (`No`).

Las variables del dataset pueden agruparse de la siguiente manera:

- Identificador: `customerID`
- Demográficas: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Servicios: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- Cuenta y facturación: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

### 3.2 Carga inicial y estructura

Se cargó el dataset con `pandas` y se verificó la dimensión del mismo:

- Filas: 7,043
- Columnas: 21

Se confirmó además que no existen clientes duplicados y que la estructura general del dataframe es consistente con un problema clásico de clasificación tabular.

### 3.3 Calidad de datos y limpieza inicial

Durante la inspección inicial se detectaron dos problemas relevantes:

1. `TotalCharges` se leía como texto en lugar de numérico.
2. En algunas filas la columna `TotalCharges` presenta cadenas vacías o valores no numéricos, lo cual requiere conversión explícita a tipo numérico con manejo de valores faltantes.

Además, la variable `SeniorCitizen` usa codificación binaria (`0` y `1`), mientras que otras variables categóricas binarias usan valores `Yes`/`No`. Esto es relevante para el preprocesamiento posterior, especialmente si se pretende estandarizar la representación de atributos para modelos de Machine Learning.

### 3.4 Distribución de la variable objetivo

La variable objetivo presenta un desbalance claro:

- `No`: 5,174 clientes (73.46%)
- `Yes`: 1,869 clientes (26.54%)

Este desbalance es importante porque afecta la elección de métricas de evaluación y la interpretación de resultados. En problemas de churn, no basta con usar accuracy como métrica principal, ya que un modelo que predice siempre la clase mayoritaria podría aparecer con alta precisión y aun así no ser útil para detectar clientes en riesgo real.

### 3.5 Variables numéricas

Las variables numéricas más relevantes del dataset son:

- `tenure`
- `MonthlyCharges`
- `TotalCharges`

El análisis descriptivo mostró que:

- `tenure` tiene una distribución con varios clientes con permanencia corta y una fracción significativa con contratos de larga duración.
- `MonthlyCharges` presenta rangos amplios, con clientes pagando más en planes con mayor velocidad y servicios adicionales.
- `TotalCharges` está muy relacionado con la antigüedad del cliente y su monto mensual acumulado.

La inspección por `Churn` sugiere que los clientes con mayor permanencia y con mayores montos acumulados tienden a tener menores tasas de abandono, mientras que los clientes con menos tiempo de permanencia y con planes más costosos pueden estar más expuestos al churn.

### 3.6 Variables categóricas

La mayor parte del dataset es categórico, y varias variables muestran relación directa con la deserción:

- `Contract`: es una de las variables más importantes. La tasa de churn es mucho mayor en contratos de tipo `Month-to-month` que en contratos anuales o bianuales.
- `InternetService`: clientes con `Fiber optic` presentan una tasa de churn significativamente mayor que clientes con `DSL` o sin servicio de internet.
- `PaymentMethod`: el pago por `Electronic check` está asociado con una mayor tasa de abandono.
- `PaperlessBilling`: la facturación electrónica parece estar asociada a una mayor proporción de churn.
- Servicios adicionales como `OnlineSecurity`, `OnlineBackup`, `TechSupport` y `DeviceProtection` muestran diferencias importantes entre clientes que se quedan y los que abandonan.

Se observa un patrón consistente: los clientes con contratos a más largo plazo, mejores niveles de servicio y menor uso de servicios intensivos tienden a quedarse. Por el contrario, aquellos con contratos cortos y servicios más costosos tienen mayor riesgo de churn.

### 3.7 Valores faltantes, outliers y calidad de datos

La revisión de valores faltantes muestra que `TotalCharges` fue la variable con valores nulos (11 registros) tras la conversión a numérico. Estos faltantes deben manejarse adecuadamente en las etapas posteriores. En el resto de columnas no se detectan faltantes relevantes.

Respecto a outliers, los cálculos con el criterio IQR no muestran outliers significativos en las variables numéricas principales, lo que sugiere que los datos son relativamente estables y que no hay valores extremos artificialmente contaminantes en la mayoría de las variables numéricas.

### 3.8 Posibles problemas de data leakage y sesgos

Durante el EDA también se analizó el riesgo de data leakage:

- `customerID` es un identificador único y no aporta valor predictivo al modelo; debe descartarse del entrenamiento.
- `TotalCharges` puede estar correlacionado con `MonthlyCharges` y `tenure`, y aunque no es necesariamente leak, debe tratarse con precaución para evitar introducir información redundante o indirectamente derivada.
- Variables de servicios adicionales y tipo de contrato pueden estar altamente correlacionadas entre sí, lo que potencialmente genera multicolinealidad y debe considerarse al momento de seleccionar modelos.

## 4. Hallazgos clave

Los hallazgos principales del EDA son los siguientes:

- El churn es una variable desbalanceada, por lo que se requerirá una evaluación orientada a métricas como recall, precision, F1-score o ROC-AUC.
- El tipo de contrato es una de las señales más fuertes: clientes con contratos mensuales abandonan con mayor frecuencia.
- `InternetService` y los servicios adicionales parecen estar asociados con el riesgo de churn.
- La facturación y los costos mensuales también son factores relevantes.
- La limpieza de `TotalCharges` es obligatoria antes del modelado.
- `customerID` no debe incluirse en el conjunto de entrenamiento.

## 5. Propuesta de pasos siguientes

Para la siguiente etapa del proyecto, se propone:

1. Generar un conjunto de datos final para modelado, eliminando identificadores y tratando correctamente `TotalCharges`.
2. Realizar preprocesamiento: codificación de variables categóricas, manejo de valores faltantes y normalización o escalado cuando aplique.
3. Definir un baseline simple como referencia.
4. Evaluar al menos dos modelos de clasificación (por ejemplo, regresión logística, random forest o gradient boosting).
5. Medir desempeño con métricas apropiadas para datos desbalanceados.
6. Analizar la importancia de variables y discutir resultados, limitaciones y conclusiones.

## 6. Referencias

- Dataset Telco Customer Churn (fuente pública de telecomunicaciones / IBM sample data) utilizado para el análisis de churn.
- Guía del proyecto: `docs/guia_proyecto.md`.
- Documentación de pandas, matplotlib y seaborn para análisis exploratorio y visualización de datos.

## 7. Conclusión preliminar

El análisis exploratorio confirma que el churn en clientes de telecomunicaciones es un problema de clasificación relevante y con señales claras. El comportamiento del cliente, el tipo de contrato, la permanencia, el servicio de internet y la facturación emergen como variables clave. El siguiente paso natural es convertir este análisis exploratorio en un pipeline reproducible de modelado predictivo con validación y evaluación de desempeño.
