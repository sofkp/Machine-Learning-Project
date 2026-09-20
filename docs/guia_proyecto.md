Guía para el Proyecto Final de Machine Learning
(Proyecto 1 y Proyecto 2)

Trabajo en equipos de 3 a 5 estudiantes

1  DESCRIPCIÓN GENERAL

El proyecto final consiste en desarrollar un trabajo con formato de artículo académico en el que los
estudiantes planteen y resuelvan un problema utilizando técnicas de Machine Learning (métodos de
Deep Learning también son aplicables). El objetivo es demostrar no solo la capacidad de entrenar
modelos, sino también la comprensión del problema, los datos, la metodología, la evaluación y la
interpretación de los resultados.

El proyecto se realizará en equipos de 3 a 5 estudiantes. Todos los integrantes deben participar
activamente en el desarrollo y ser capaces de explicar las decisiones metodológicas tomadas durante el
proyecto.

2  DESARROLLO EN DOS ETAPAS

El proyecto se desarrollará en dos etapas. La primera entrega estará enfocada en la comprensión y
formulación del problema y en el análisis de los datos. La segunda completará el desarrollo
metodológico, experimental y analítico del proyecto. Ambas etapas incluirán los mismos artefactos,
que son descritos en las próximas secciones (paper, código en repositorio, y presentación). Las
presentaciones se realizarán durante las clases de laboratorio de la semana correspondiente a la
entrega de cada una de éstas. Para ambas etapas, cada exposición tendrá un tiempo máximo de 10
minutos, seguidos de una ronda de preguntas.

2.1  ETAPA 1 (PROYECTO 1 – P1) — PROBLEMA, DATASET Y EDA
La primera etapa debe demostrar que el equipo comprende adecuadamente el problema antes de
comenzar con el entrenamiento de modelos. La entrega de esta etapa será realizada en la semana 7
del curso.

•  Descripción y motivación del problema.
•  Definición clara de la tarea de Machine Learning.
•  Preguntas de investigación y/o hipótesis, cuando corresponda.
•  Descripción del dataset: fuente, tamaño, variables y variable objetivo.
•  Análisis exploratorio de datos (EDA), acompañado de visualizaciones relevantes.
•

Identificación de valores faltantes, outliers, desbalance, relaciones entre variables y posibles
problemas de data leakage.

•  Propuesta de los siguientes pasos del proyecto.

2.2  ETAPA 2 (PROYECTO 2 – P2)  — MODELADO, EVALUACIÓN Y DISCUSIÓN
La segunda etapa, además del contenido de la primera etapa incluye los tópicos presentados a
continuación. La entrega de esta etapa será realizada en la semana 16 del curso.

Proyecto Final — Machine Learning

Implementación y comparación de al menos dos modelos de Machine Learning.

•  Preprocesamiento e ingeniería de características.
•  Definición de un baseline.
•
•  Diseño experimental, validación y ajuste de hiperparámetros.
•  Evaluación mediante métricas apropiadas.
•  Análisis e interpretación de resultados y errores.
•  Discusión, limitaciones, consideraciones éticas y conclusiones.

3  ESTRUCTURA DEL TRABAJO ESCRITO

El documento final deberá seguir, como mínimo, la estructura indicada en esta guía y deberá utilizar el
template oficial de conferencias IEEE ( https://template-
selector.ieee.org/secure/templateSelector/format?publicationTypeId=3&titleId=1&articleId=1 ). La
entrega se hará en formato PDF y se podrá considerar edición en Latex o Word.

El documento tendrá una extensión máxima de 12 páginas, para ambas entregas P1 y P2. El trabajo
podrá ser redactado en español o en inglés.

1. Resumen [P1 y P2]

•  Presentar brevemente el problema, el conjunto de datos, la metodología utilizada [P2] y los

principales resultados y conclusiones [P2].

2. Introducción y formulación del problema [P1 y P2]

•  Describir el problema que se desea resolver y su relevancia.
•  Definir claramente el objetivo del proyecto y la tarea de Machine Learning.
•  Plantear, cuando sea apropiado, una o más preguntas de investigación o hipótesis.

3. Datos y Análisis Exploratorio de Datos (EDA) [P1 y P2]

Indicar número de observaciones, variables, tipos de datos y variable objetivo.

•  Describir el origen y las características principales del dataset.
•
•  Analizar valores faltantes, distribución de clases, posibles outliers y problemas de calidad.
•
•

Incluir visualizaciones y análisis exploratorio relevantes.
Identificar posibles problemas de sesgo o data leakage.

4. Preprocesamiento e ingeniería de características [P2]

•  Explicar las transformaciones realizadas sobre los datos.
•

Justificar decisiones relacionadas con valores faltantes, codificación, escalamiento, selección o
creación de variables, entre otras.

5. Metodología y modelos [P2]

•  Presentar los modelos de Machine Learning utilizados.
•

Incluir un modelo baseline y al menos dos enfoques de modelado que permitan realizar una
comparación.

•  Explicar y justificar la selección de los modelos y sus principales hiperparámetros.

6. Diseño experimental y evaluación [P2]

•  Describir la estrategia de entrenamiento, validación y prueba.

Proyecto Final — Machine Learning

•  Explicar el uso de validación cruzada y/o búsqueda de hiperparámetros cuando corresponda.
•  Definir y justificar las métricas utilizadas.

7. Resultados y análisis [P2]

•  Presentar los resultados de manera clara mediante tablas y visualizaciones.
•  Comparar el desempeño de los modelos.
•  Analizar errores, predicciones incorrectas, importancia de características u otras técnicas de

interpretación pertinentes.

8. Discusión y limitaciones [P2]

•

Interpretar los resultados y explicar por qué algunos modelos pueden haber funcionado mejor que
otros.

•  Discutir las principales limitaciones del dataset, metodología y resultados.
•  Considerar aspectos éticos, sesgos o problemas de generalización cuando sean relevantes.

9. Conclusiones [P2]

•  Responder las preguntas de investigación planteadas.
•  Resumir los principales aprendizajes y resultados.
•  Proponer posibles líneas de trabajo futuro.

10. Referencias [P1 y P2]

•

Incluir todas las fuentes utilizadas siguiendo un formato de citación consistente.

4  REPOSITORIO DE CÓDIGO Y REPRODUCIBILIDAD

El proyecto deberá incluir un repositorio público de código asociado al trabajo. El repositorio debe
estar organizado, ser comprensible para una persona externa al equipo y permitir reproducir, en la
medida de lo posible, los experimentos presentados en el documento.

Como mínimo, el repositorio deberá incluir:

•  README.md con una descripción del proyecto, objetivo, integrantes, estructura del repositorio e

instrucciones para ejecutar el código.

•  Código fuente organizado en carpetas y archivos con nombres claros.
•  Archivos de configuración de dependencias o instrucciones para crear el entorno de ejecución.
•  Scripts o notebooks necesarios para reproducir el análisis y los experimentos.
•  Carpeta o sección de documentación pública del proyecto, cuando corresponda.
•

Información sobre cómo obtener o preparar el dataset, respetando las restricciones de
distribución y licencias.

•  Resultados, figuras o archivos generados que sean necesarios para comprender el proyecto,

evitando incluir archivos innecesariamente grandes.

La calidad y organización del repositorio serán consideradas parte de la evaluación. No se espera
únicamente que el código funcione: debe estar estructurado de forma clara, documentada y
reproducible.

Proyecto Final — Machine Learning

5  PRESENTACIÓN

Cada equipo deberá realizar una presentación de cada etapa del proyecto. La presentación debe
resumir el contenido de cada etapa. El material usado para presentar también deberá ser entregado en
formato PDF.

•  Todos los integrantes del equipo deben participar en la presentación.
•  Solo un integrante deberá compartir pantalla durante la presentación.
•  El equipo debe estar preparado para responder preguntas sobre el dataset, los modelos, la

evaluación y las decisiones tomadas.

6  CRITERIOS GENERALES DE EVALUACIÓN

Se valorará especialmente la capacidad del equipo para justificar sus decisiones y demostrar
comprensión de los conceptos de Machine Learning, además del desempeño obtenido por los modelos.
Las rúbricas para cada etapa del proyecto se encuentran en la plataforma Canvas.

Proyecto Final — Machine Learning


