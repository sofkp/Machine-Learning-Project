# P06 · Discusión, limitaciones y conclusiones

- **Proyecto:** Abandono de clientes de telecomunicaciones (Telco Customer Churn) — P2
- **Corresponde a:** secciones 8 (Discusión y limitaciones) y 9 (Conclusiones) del informe.

> **Estado: propuesta.** Estas secciones están asignadas a Farid, Sebastian y Sofia. Este documento las
> desarrolla a partir de los resultados de P03, P04 y P05 para que puedan usarlo como base, recortarlo o
> reemplazarlo. La versión resumida, ya en formato del informe, está en `paper/sections/discussion.tex` y
> `paper/sections/conclusions.tex`.
>
> **De dónde salen las cifras.** Todas provienen de las tablas guardadas en `resultados/` por los
> notebooks P03, P04 y P05 (la sección 8 de este documento indica el archivo de cada una). Los decimales
> se escriben con punto, igual que en los notebooks y en el informe.

---

## 1. Qué se obtuvo

Los tres modelos se entrenaron solo con Train (4225 clientes), se eligieron con Validation (1409) y se
evaluaron una sola vez en Test (1409 clientes, 374 abandonos). El criterio de selección fue el F1-score
de `Churn=1` en Validation, con el umbral fijo en 0.5, tal como define el protocolo de P02.

| Modelo | Configuración elegida | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dummy Classifier | `most_frequent` | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Regresión logística | L2, `C=0.1`, `class_weight="balanced"` | 0.743 | 0.510 | 0.781 | 0.617 | 0.841 |
| Árbol de decisión | sin límite de profundidad, `min_samples_leaf=20`, `class_weight="balanced"` | 0.744 | 0.512 | 0.722 | 0.599 | 0.818 |

Intervalos bootstrap del 95 % en Test (2000 remuestreos):

| Métrica | Regresión logística | Árbol de decisión | Diferencia (RL − AD) | ¿Se distingue de cero? |
| --- | --- | --- | --- | --- |
| Recall | [0.736, 0.820] | [0.675, 0.767] | +0.059 [0.016, 0.103] | Sí |
| ROC-AUC | [0.818, 0.863] | [0.793, 0.841] | +0.023 [0.009, 0.040] | Sí |
| F1-score | [0.579, 0.654] | [0.559, 0.635] | +0.018 [−0.009, 0.049] | No |
| Precision | [0.470, 0.551] | [0.469, 0.554] | −0.002 [−0.027, 0.025] | No |
| Accuracy | [0.720, 0.766] | [0.720, 0.765] | −0.001 [−0.021, 0.020] | No |

En una línea: **los dos modelos detectan entre el 72 % y el 78 % de los abandonos que el baseline no
detecta en absoluto; la regresión logística es mejor en Recall y ROC-AUC, y en F1 la ventaja no es
concluyente.**

---

## 2. Interpretación de los resultados

### 2.1 Los modelos sí aportan, aunque la exactitud casi no cambie

El Dummy acierta en el 73.5 % de los clientes sin mirar ninguna característica, porque esa es la
proporción de clientes que permanecen. La regresión logística llega a 74.3 % y el árbol a 74.4 %. La
diferencia con el Dummy es de 0.009 y su intervalo incluye el cero ([−0.025, 0.041] y [−0.023, 0.040]).

Eso no quiere decir que los modelos no sirvan. Lo que pasa se ve en las matrices de confusión:

| Modelo | VN | FP | FN | VP | Aciertos totales |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy Classifier | 1035 | 0 | 374 | 0 | 1035 |
| Regresión logística | 755 | 280 | 82 | 292 | 1047 |
| Árbol de decisión | 778 | 257 | 104 | 270 | 1048 |

La regresión logística gana 292 aciertos sobre clientes que abandonan y pierde 280 sobre clientes que
permanecen. El saldo es de solo 12 aciertos más en 1409 clientes, por eso la exactitud no se mueve. Pero
el contenido de esos aciertos es completamente distinto: pasa de detectar 0 abandonos a detectar 292.
Es la razón por la que el protocolo usa F1 de la clase positiva y no Accuracy.

Sobre la calidad de las alertas: cerca de la mitad corresponde a clientes que realmente abandonan
(Precision ≈ 0.51). Parece poco, pero la tasa base es 26.5 %, así que una alerta del modelo señala a
alguien con casi el doble de probabilidad de irse que un cliente tomado al azar.

### 2.2 Por qué la regresión logística iguala o supera al árbol

Hay tres razones, y las tres se pueden verificar con los resultados.

1. **La señal de estos datos es, en buena parte, aditiva y monótona.** En el EDA la diferencia por
   contrato se mantenía dentro de cada grupo de antigüedad, y el abandono bajaba de forma sostenida con
   los meses de permanencia. Un modelo lineal en los *log-odds* representa justo ese tipo de efecto: cada
   variable suma o resta riesgo, sin importar el valor de las demás. Además lo hace con pocos parámetros
   (46 coeficientes y un intercepto) y cada uno se estima con los 4225 clientes de Train.

2. **El árbol reparte los datos.** Cada división deja menos clientes en cada rama, y al final cada hoja
   estima su tasa de abandono con pocos casos (a veces solo 20). Esto tiene dos consecuencias medibles:
   - produce solo **unas 100 probabilidades distintas** en Test (como máximo una por hoja), frente a
     1407 de la regresión logística, así que ordena a los clientes de forma más gruesa (de ahí el
     menor ROC-AUC);
   - la brecha entre Train y Validation es mayor: **0.066** de F1 (0.685 frente a 0.619) contra **0.012**
     en la regresión logística (0.637 frente a 0.625).

3. **Las interacciones aportan poco aquí.** La ventaja teórica del árbol es que capta combinaciones de
   variables sin que nadie se las indique. En estos datos eso se nota en un solo lugar: entre los
   clientes sin internet el árbol detecta 11 de los 25 abandonos y la regresión logística ninguno. Son
   pocos casos y no alcanzan para compensar lo anterior.

Un dato que refuerza la primera razón: un árbol de **8 hojas** (profundidad 3) obtiene en Validation
F1 = 0.608, a 0.011 del árbol elegido de 132 hojas, y esa diferencia no se distingue de cero (intervalo
[−0.014, 0.035]). Lo que ese árbol pequeño decide cabe en una línea: **da la alerta si el contrato es
mensual y, además, el cliente tiene fibra óptica o lleva 5 meses o menos**. Es decir, casi toda la señal
está en tres variables: contrato, tipo de internet y antigüedad.

### 2.3 Qué limita el desempeño

La diferencia entre los dos modelos es modesta. Lo que parece poner el techo no es el algoritmo sino la
información disponible. Los dos modelos se equivocan, en gran parte, con los mismos clientes: coinciden
en el 84.7 % de las predicciones y hay **61 abandonos (16.3 %) que ninguno detecta**.

Esos 61 clientes tienen el perfil de alguien que se queda:

| Grupo (Test) | n | Antigüedad mediana | Contrato mensual | Fibra óptica | Cargo mensual mediano |
| --- | ---: | ---: | ---: | ---: | ---: |
| Abandonan y ningún modelo los detecta | 61 | 35 meses | 44.3 % | 34.4 % | 64.4 |
| Abandonan y ambos modelos los detectan | 249 | 4 meses | 99.6 % | 80.7 % | 79.0 |
| Permanecen | 1035 | 38 meses | 42.9 % | 34.9 % | 64.7 |

Con las 19 variables disponibles, estos clientes son prácticamente indistinguibles de los que
permanecen. Para detectarlos harían falta variables que el conjunto de datos no tiene: reclamos, calidad
del servicio, cambios recientes de plan u ofertas de la competencia.

<details>
<summary>Código para reproducir la tabla anterior (desde la raíz del repositorio)</summary>

```python
import pandas as pd

datos = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
pred = pd.read_csv("resultados/predicciones_test.csv")
test = datos.loc[pred["Fila en el CSV original"]].reset_index(drop=True)
rl = pred["Predicción · Regresión logística"]
ad = pred["Predicción · Árbol de decisión"]
abandona = pred["Churn real"] == 1

grupos = {
    "Abandonan, ningún modelo los detecta": abandona & (rl == 0) & (ad == 0),
    "Abandonan, ambos modelos los detectan": abandona & (rl == 1) & (ad == 1),
    "Permanecen": ~abandona,
}
for nombre, filtro in grupos.items():
    g = test[filtro]
    print(f"{nombre}: n={len(g)}, tenure mediana={g['tenure'].median():.0f}, "
          f"contrato mensual={100 * (g['Contract'] == 'Month-to-month').mean():.1f} %, "
          f"fibra={100 * (g['InternetService'] == 'Fiber optic').mean():.1f} %, "
          f"cargo mensual mediano={g['MonthlyCharges'].median():.1f}")
```

Es un análisis descriptivo de errores sobre predicciones ya guardadas: no vuelve a entrenar ni a
seleccionar nada.
</details>

### 2.4 La decisión que más pesó: el tratamiento del desbalance

La fuerza de la regularización casi no cambió el resultado: con L2 y ponderación de clases, el F1 de
Validation va de 0.610 a 0.625 en todo el rango de `C`, y las cinco mejores configuraciones difieren en
menos de 0.002. En cambio, `class_weight="balanced"` sí lo cambió:

| Modelo (Validation) | Sin ponderar (mejor configuración) | Con ponderación (elegida) |
| --- | --- | --- |
| Regresión logística | F1 0.593 · Precision 0.66 · Recall 0.53 · Accuracy 0.80 | F1 0.625 · Precision 0.52 · Recall 0.79 · Accuracy 0.75 |
| Árbol de decisión | F1 0.579 | F1 0.619 |

Con los pesos (1.884 para `Churn=1` y 0.681 para `Churn=0`), un error sobre un cliente que abandona
cuesta 2.77 veces más que uno sobre un cliente que permanece. El modelo responde bajando la exigencia
para dar una alerta: detecta muchos más abandonos y, a cambio, genera más falsas alarmas y pierde
exactitud.

El costo es concreto. La regresión logística señala a **572 clientes de Test (40.6 %)** y la mitad son
falsas alarmas. Si ese intercambio conviene depende de cuánto cuesta una acción de retención frente a
cuánto cuesta perder un cliente, y ese dato no está en el conjunto. Por eso el resultado se presenta
como un punto de operación razonable y no como el óptimo para un negocio.

Hay un efecto secundario: las probabilidades dejan de estar calibradas. En Test la probabilidad media
estimada es 0.414 en la regresión logística y 0.389 en el árbol, frente a una tasa real de 0.265. Las
salidas sirven para ordenar clientes por riesgo y para clasificar con el umbral 0.5, pero no se deben
leer como "este cliente tiene 41 % de probabilidad de irse".

### 2.5 Los modelos confirman lo que mostraba el EDA

- Las tres variables más importantes por permutación son las mismas en ambos modelos: `Contract`,
  `tenure` e `InternetService`. Cambia el reparto: la regresión logística se apoya sobre todo en
  `tenure` (el F1 cae 0.126 al desordenarla) y el árbol en `Contract` (0.180) e `InternetService` (0.138).
- Las razones de *odds* van en la misma dirección que las tasas del EDA: contrato de dos años frente a
  mensual, 0.24; contrato de un año, 0.46; DSL frente a fibra óptica, 0.50; sin internet frente a fibra,
  0.19; `OnlineSecurity`, 0.66; `TechSupport`, 0.69.
- El género no aporta (OR = 1.05, importancia por permutación cercana a cero).
- Un caso para leer con cuidado: no tener telefonía aparece con OR = 1.90, pero en términos brutos esos
  clientes no abandonan más (24.9 % frente a 26.7 %). El OR es condicional: compara clientes con el mismo
  cargo mensual y los mismos servicios. Una lectura posible es que, en esa comparación, quien no tiene
  telefonía paga lo mismo por menos. No contradice al EDA, pero tampoco es una regla que se pueda
  presentar suelta.

---

## 3. Limitaciones

### 3.1 Del conjunto de datos

- **Es una muestra de ejemplo de una empresa ficticia.** No se sabe cómo se seleccionaron los clientes,
  así que los resultados no se pueden generalizar a un operador real ni al mercado peruano.
- **No tiene fechas.** Es una fotografía: no se sabe cuándo se registró cada variable respecto del
  abandono. Por eso no fue posible una validación temporal (entrenar con el pasado y probar con el
  futuro), que es como se usaría el modelo en la práctica, ni descartar por completo una fuga temporal
  en el origen.
- **Sesgo de supervivencia en la antigüedad.** Los clientes con muchos meses son los que no se fueron
  antes. El efecto de `tenure` describe a quienes llegaron hasta ahí, no lo que le pasaría a un cliente
  nuevo si se quedara más tiempo.
- **Imputación por supuesto.** Los 11 faltantes de `TotalCharges` (clientes con `tenure = 0`) se
  imputaron con cero. Es coherente con el significado de la variable, pero no se pudo verificar.

### 3.2 De la metodología

- **Una sola partición y validación *hold-out*.** Con 1409 clientes en Validation, diferencias de 0.002
  a 0.01 de F1 entre configuraciones están dentro del ruido. La configuración elegida podría cambiar con
  otra semilla. Una validación cruzada daría una selección más estable.
- **El F1 de Validation del modelo elegido es algo optimista.** Se eligió justamente por ser el mayor
  entre 36 (o 42) configuraciones. Por eso el desempeño que se reporta como final es el de Test, que baja
  un poco: de 0.625 a 0.617 en la regresión logística y de 0.619 a 0.599 en el árbol.
- **Umbral fijo en 0.5.** No se ajustó porque el protocolo lo fijó así y porque no hay costos con los
  cuales elegir otro. Con otro umbral cambiarían Precision, Recall y F1 (no el ROC-AUC).
- **Solo dos familias de modelos**, sin variables derivadas, interacciones explícitas ni ensambles.
- **Coeficientes individuales no únicos.** La codificación *one-hot* conserva todas las categorías, así
  que hay columnas repetidas (por ejemplo, las siete que indican "sin servicio de internet"). Las
  predicciones no se ven afectadas, pero un coeficiente suelto no se puede interpretar. Por eso se
  reportan contrastes entre categorías de una misma variable, que sí están bien definidos.
- **Importancia por permutación con variables correlacionadas.** `tenure` y `TotalCharges` comparten
  información (correlación de 0.83 en Train). Al desordenar una, el modelo todavía se apoya en la otra,
  y la importancia de cada una puede quedar subestimada.

### 3.3 De los resultados

- **Los intervalos son amplios**: cerca de ±0.04 en F1 con 374 abandonos en Test.
- **Subgrupos pequeños.** Hay 9 abandonos con contrato de dos años y 25 sin internet. Las tasas de esos
  grupos son imprecisas y sirven como señal, no como medida.
- **Asociación, no causalidad.** Las razones de *odds* y las reglas del árbol describen patrones en datos
  observacionales. Que el contrato de dos años tenga OR = 0.24 no demuestra que pasar a un cliente a ese
  contrato reduzca su probabilidad de abandono: quien elige un contrato largo probablemente ya tenía
  menos intención de irse.
- **Probabilidades no calibradas** (ver 2.4).

---

## 4. Sesgos, ética y generalización

**Género.** Prácticamente no influye. El OR es 1.05 y los errores son casi iguales: en la regresión
logística, Recall de 0.782 (mujeres) y 0.779 (hombres), y tasa de falsos positivos de 0.271 y 0.270.

**Adultos mayores (`SeniorCitizen`).** Aquí sí hay una diferencia que conviene declarar:

| Regresión logística (Test) | `SeniorCitizen=0` | `SeniorCitizen=1` |
| --- | ---: | ---: |
| Clientes / abandonos | 1187 / 276 | 222 / 98 |
| Recall | 0.728 | 0.929 |
| Tasa de falsos positivos | 0.243 | 0.476 |
| Precision de las alertas | 0.476 | 0.607 |

Un adulto mayor que permanece recibe una alerta casi el doble de veces que el resto. El efecto directo
de la variable es pequeño (OR = 1.18). La diferencia viene sobre todo de la composición del grupo: en el
conjunto completo, los adultos mayores tienen con más frecuencia contrato mensual (70.7 % frente a
52.0 %) y fibra óptica (72.8 % frente a 38.4 %), que son justo los perfiles de mayor riesgo. Por eso
quitar la variable del modelo probablemente no eliminaría la diferencia: el contrato y el tipo de
internet seguirían actuando como sustitutos.

También hay que notar que, cuando la alerta se da, acierta más en los adultos mayores (0.607 frente a
0.476), porque su tasa de abandono es más alta (44.1 % frente a 23.3 % en Test). Es decir, el resultado
depende del criterio con el que se mida la equidad, y no se pueden igualar todos a la vez cuando las
tasas base son distintas.

**Que esto sea un problema depende del uso.**

- Si la alerta activa un descuento o una llamada de retención, el grupo señalado se beneficia. Los
  perjudicados serían los clientes con contratos largos, a quienes los modelos casi nunca señalan: ningún
  modelo detecta a los 9 que abandonaron con contrato de dos años.
- Si la alerta activara un trato menos favorable, el perjuicio recaería en los adultos mayores y en
  quienes tienen contrato mensual.

**Uso de atributos personales.** El modelo recibe `gender`, `SeniorCitizen`, `Partner` y `Dependents`
como predictoras, siguiendo el protocolo común de usar las 19 características. En un uso real habría que
evaluar si corresponde usarlas en decisiones sobre clientes.

**Generalización.** Antes de cualquier uso real habría que validar los modelos con datos del operador,
separados en el tiempo, y monitorear los errores por grupo, porque el comportamiento de los clientes y
las ofertas cambian.

---

## 5. Respuestas a las preguntas de investigación

1. **¿Qué características presentan diferencias en la proporción de abandono observado?**
   El contrato, la antigüedad, el tipo de internet, el método de pago y los servicios de seguridad y
   soporte. Los modelos lo confirman: `Contract`, `tenure` e `InternetService` son las tres variables más
   importantes en ambos.

2. **¿Las diferencias por contrato se mantienen al comparar grupos de antigüedad similares?**
   Sí. En la regresión logística, con las demás variables fijas (incluida la antigüedad), un contrato de
   dos años multiplica los *odds* de abandono por 0.24 frente al mensual, y uno de un año por 0.46. En el
   árbol, el contrato es la primera división: 43.1 % de abandono con contrato mensual frente a 6.3 % en
   el resto (Train).

3. **¿Cómo se relacionan el tipo de internet y los servicios adicionales con el abandono, y qué
   diferencias permanecen dentro de cada contrato?**
   La fibra óptica se asocia con más abandono incluso dentro del contrato mensual (55.5 % frente a 28.1 %
   en el árbol; OR de DSL frente a fibra: 0.50). Contar con `OnlineSecurity` o `TechSupport` se asocia
   con menos abandono (OR de 0.66 y 0.69).

4. **¿Qué problemas de calidad, representatividad y fuga de información deben considerarse antes del
   modelado?**
   Los 11 faltantes de `TotalCharges` se imputaron con cero; se excluyó `customerID`; los preprocesadores
   se ajustaron solo con Train y Test se usó una sola vez. La representatividad es la limitación principal
   y no se resuelve con modelado, porque es una muestra ficticia y sin fechas.

5. **¿En qué medida la regresión logística y el árbol de decisión permiten identificar clientes que
   abandonan, en comparación con el clasificador de referencia?**
   El Dummy no detecta ninguno de los 374 abandonos de Test. La regresión logística detecta 292
   (Recall 0.781, F1 0.617, ROC-AUC 0.841) y el árbol 270 (0.722, 0.599 y 0.818), con una Precision
   cercana a 0.51. La exactitud casi no cambia (0.74 frente a 0.735), lo que confirma que no es la métrica
   adecuada para este problema.

6. **¿Qué diferencias presentan ambos modelos en su desempeño e interpretabilidad, y qué características
   resultan relevantes?**
   La regresión logística supera al árbol en Recall y ROC-AUC; en F1 la diferencia no es concluyente. La
   regresión se interpreta con razones de *odds* que valen para todos los clientes. El árbol se expresa
   en reglas, pero el elegido tiene 132 hojas y solo se pueden leer sus primeros niveles; uno de 8 hojas
   rinde casi igual. Las características relevantes son `Contract`, `tenure` e `InternetService`.

---

## 6. Aprendizajes

- **El desbalance pesó más que la regularización.** Mover `C` casi no cambió el F1; activar la
  ponderación de clases sí.
- **Cuantificar la incertidumbre cambia las conclusiones.** Sin intervalos se habría dicho que la
  regresión logística es mejor en F1 (0.617 frente a 0.599). Con ellos, lo correcto es decir que es mejor
  en Recall y ROC-AUC y que en F1 no se puede afirmar.
- **Un modelo más complejo no fue mejor.** El árbol sin restricciones memoriza Train (F1 = 0.996 con
  859 hojas) y cae a 0.496 en Validation. Controlar el tamaño de las hojas fue lo que lo hizo útil.
- **Fijar el protocolo antes de entrenar** (particiones, métrica, umbral y uso de Test) hizo que los
  resultados de todos los integrantes fueran comparables y evitó decidir mirando Test.
- **La exactitud engaña con clases desbalanceadas.** Tres modelos muy distintos quedan entre 0.735 y
  0.744.

---

## 7. Trabajo futuro

1. Seleccionar hiperparámetros con **validación cruzada estratificada** para que la elección no dependa
   de una sola partición.
2. **Elegir el umbral** a partir de los costos de una acción de retención y de perder un cliente, en
   lugar de fijarlo en 0.5.
3. **Calibrar las probabilidades** (por ejemplo, con regresión isotónica o de Platt) si se quieren usar
   como estimaciones de riesgo.
4. Evaluar **interacciones y variables derivadas**, y modelos de ensamble como *random forest* o
   *gradient boosting*, que reducen la varianza de un solo árbol.
5. **Validar con datos reales separados en el tiempo** y revisar periódicamente los errores por grupo.

---

## 8. Trazabilidad: dónde está cada cifra

| Dato | Archivo en `resultados/` |
| --- | --- |
| Configuraciones elegidas, métricas de Train y Validation, versiones | `configuracion_regresion_logistica.json`, `configuracion_arbol_decision.json` |
| Las 36 y 42 configuraciones evaluadas | `resultados_regresion_logistica_grilla_validacion.csv`, `resultados_arbol_decision_grilla_validacion.csv` |
| Métricas de los tres modelos en Validation y Test | `comparacion_modelos_validacion_test.csv` |
| Intervalos bootstrap y diferencias entre modelos | `intervalos_bootstrap_test.csv`, `diferencias_bootstrap_test.csv` |
| Matrices de confusión | `matriz_confusion_*_test.csv` |
| Razones de *odds* y coeficientes | `razones_de_odds_regresion_logistica.csv`, `coeficientes_regresion_logistica.csv` |
| Reglas, hojas e importancias del árbol | `reglas_arbol_decision.txt`, `hojas_arbol_decision.csv`, `importancias_arbol_decision.csv` |
| Importancia por permutación (Validation) | `importancia_permutacion_*_validacion.csv` |
| Errores por subgrupo, coincidencia y perfil de errores | `errores_por_subgrupo_test.csv`, `coincidencia_errores_test.csv`, `perfil_errores_test.csv` |
| Predicciones y probabilidades cliente por cliente | `predicciones_test.csv` |

Tres datos no están en esas tablas:

- La comparación entre el árbol elegido y el de 8 hojas (diferencia de 0.011, intervalo
  [−0.014, 0.035]) se calcula en `notebooks/P04_arbol_decision.ipynb`. El árbol de 8 hojas es la
  configuración 26 de la grilla (`max_depth=3`, `min_samples_leaf=20`, `class_weight="balanced"`); la
  regla de una línea de la sección 2.2 sale de sus divisiones y, aplicada tal cual a Validation, da el
  mismo F1 de 0.608.
- El número de probabilidades distintas se imprime en `notebooks/P05_evaluacion_comparativa.ipynb`
  (1407 y 101). En el árbol, varias hojas tienen la misma proporción y solo difieren en el último
  decimal de punto flotante; al redondear quedan 95 valores. Por eso aquí se dice "unas 100".
- Los porcentajes de contrato mensual y fibra óptica por `SeniorCitizen` son un cálculo descriptivo
  sobre el conjunto completo:

```python
import pandas as pd

datos = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
resumen = datos.assign(
    mensual=datos["Contract"].eq("Month-to-month"),
    fibra=datos["InternetService"].eq("Fiber optic"),
).groupby("SeniorCitizen")[["mensual", "fibra"]].mean().mul(100).round(1)
print(resumen)  # 0: 52.0 y 38.4 · 1: 70.7 y 72.8
```
