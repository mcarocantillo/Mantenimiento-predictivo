# Protocolo de confirmación con XJTU-SY y FEMTO/PRONOSTIA

Fecha de registro: 12 de noviembre de 2026. Este protocolo se fija **antes** de calcular cualquier resultado en los
conjuntos XJTU-SY y FEMTO. La fecha del commit en el repositorio sirve como constancia. Cualquier cambio posterior se
documentará como desviación, con su motivo.

## 1. Qué se quiere confirmar

En los datos IMS, la mejor alarma fue la **alarma de dispersión de la carta X̄-S sobre el RMS** (en adelante, "carta
S"): avisó con una mediana de 86 horas de anticipación, con 2 % de falsas alarmas y una precisión de 84 % a una semana.
Con solo 4 fallas, el resultado no fue concluyente porque se eligió entre 54 alarmas.

**Hipótesis principal (H1).** La carta S, aplicada sin ningún ajuste a rodamientos nuevos, avisa en el tramo final
antes de la falla más seguido de lo que lo haría por casualidad.

Es **una sola prueba planeada**, así que no se corrige por comparaciones múltiples. Todo lo demás (otras reglas, otras
cartas, la curtosis) se reporta como análisis exploratorio, sin conclusiones fuertes.

## 2. Datos

| Conjunto | Rodamientos | Qué se usa | Señal |
| --- | --- | --- | --- |
| XJTU-SY (Wang et al., 2020) | 15, en 3 condiciones de velocidad y carga | Todos, salvo los excluidos por la regla de la sección 3 | 1,28 s a 25,6 kHz cada minuto; canales horizontal y vertical |
| FEMTO/PRONOSTIA (Nectoux et al., 2012) | 17 (6 de aprendizaje y 11 del conjunto de prueba completo) | Todos, salvo los excluidos por la regla de la sección 3 | 0,1 s a 25,6 kHz cada 10 s; canales horizontal y vertical |

- **Momento de falla:** el último archivo de cada rodamiento, igual que en IMS.
- **Canal principal:** el horizontal. El vertical se reporta como análisis de sensibilidad.
- **Unidad de análisis:** el rodamiento. En estos conjuntos cada prueba tiene un solo rodamiento, así que no hay
  rodamientos sanos en el mismo eje; las falsas alarmas se miden en el tramo sano inicial de cada uno.

## 3. Reglas fijas

1. **Exclusión.** Se excluyen los rodamientos con menos de 100 archivos, porque no dejan tramo suficiente para fijar
   los límites y monitorear. Según la tabla oficial de XJTU-SY, esto excluye Bearing1_5 (52 archivos) y Bearing2_4
   (42 archivos).
2. **Subgrupos.** Cada archivo del canal se divide en **20 trozos de igual longitud**. En cada trozo se calcula el RMS.
   El punto de la carta S es la desviación estándar de los 20 RMS (S); el de la carta X̄, su promedio.
3. **Línea base.** Se usa la misma regla que en IMS, escalada a la duración de cada prueba:
   - Longitud L = el mayor entre 30 archivos y el 10 % de los archivos del rodamiento.
   - Se calcula el coeficiente de variación (CV) del RMS en ventanas móviles de L/4 archivos.
   - La línea base es el bloque de L archivos seguidos, dentro de la primera mitad de la prueba, cuyo CV máximo es el
     más bajo.
4. **Límites de la carta S.** S̄ = promedio de S en la línea base. Límite inferior 0,510·S̄ y superior 1,490·S̄
   (constantes B3 y B4 para subgrupos de 20; Montgomery).
5. **Alarma.** Un **evento** de alarma es S fuera de sus límites en 3 archivos seguidos, igual que en IMS.
6. **Tramo previo a la falla.** El **10 % final** de la duración de cada prueba. Como las pruebas duran entre minutos y
   decenas de horas, la ventana se define como proporción de la vida y no en horas.
7. **Tramo sano de referencia.** Desde el final de la línea base hasta la mitad de la prueba.

## 4. Prueba de la hipótesis principal

1. Para cada rodamiento se registra si hubo **al menos un evento** de alarma en el tramo previo a la falla (acierto).
2. La probabilidad de acertar por casualidad, p₀, se estima con los eventos en los tramos sanos de todos los
   rodamientos del conjunto: con la tasa de eventos por archivo en el tramo sano, λ, y el número de archivos de la
   ventana final de cada rodamiento, nᵢ, la probabilidad de al menos un evento por azar es 1 − e^(−λ·nᵢ). p₀ es el
   promedio sobre los rodamientos.
3. Se aplica una **prueba binomial de una cola**: k aciertos entre n rodamientos frente a p₀, con α = 0,05.
4. Se hace por separado para XJTU-SY y para FEMTO, y con los dos juntos.

**Criterio de confirmación:** H1 se considera confirmada si el p de la prueba binomial es menor que 0,05 en el
análisis conjunto y la proporción de aciertos es mayor que p₀ en cada conjunto por separado.

## 5. Medidas secundarias

Se reportan con las mismas definiciones que en IMS, para la carta S y para el método original (RMS en carta I-MR con
la regla 1):

- sensibilidad en el tramo previo a la falla;
- falsas alarmas: porcentaje del tramo sano con alarma y eventos por hora;
- anticipación: tiempo entre el primer evento sostenido del 30 % final de la prueba y la falla, en minutos y como
  porcentaje de la vida;
- precisión: proporción de eventos que ocurren dentro del tramo previo a la falla.

## 6. Análisis de sensibilidad (exploratorios)

- Canal vertical en lugar del horizontal.
- Tramo previo a la falla de 5 % y 20 % de la vida.
- Por condición de operación (velocidad y carga) y, en XJTU-SY, por elemento que falló.
- Las demás cartas y reglas de Nelson, y la curtosis.

## 7. Desviaciones

Cualquier cambio a este protocolo después de ver los datos se registrará aquí, con la fecha y el motivo.

**12 de noviembre de 2026, después de procesar FEMTO y antes de procesar XJTU-SY. Aclaración de implementación.**
La primera versión del código contaba como acierto solo los eventos que **empezaban** dentro del tramo previo a la
falla. Así, un rodamiento con la alarma sonando el 100 % del tramo final, pero desde antes de que empezara ese tramo,
quedaba como fallo. Se corrigió para que cumpla la definición de la sección 4: hay acierto si dentro del tramo previo
hay al menos una racha de 3 archivos seguidos fuera de los límites, haya empezado antes o no. Como esta corrección
favorece a la hipótesis, se agrega una versión **estricta** de p₀, que suma a la probabilidad del protocolo la fracción
del tramo sano con una alarma ya en curso. Se reportan las dos.

## Referencias

Montgomery, D. C. *Introduction to Statistical Quality Control* (8.ª ed.). Wiley.

Nectoux, P., Gouriveau, R., Medjaher, K., Ramasso, E., Chebel-Morello, B., Zerhouni, N. y Varnier, C. (2012).
PRONOSTIA: An experimental platform for bearings accelerated degradation tests. En *IEEE International Conference on
Prognostics and Health Management (PHM'12)*, Denver. https://hal.science/hal-00719503

Wang, B., Lei, Y., Li, N. y Li, N. (2020). A hybrid prognostics approach for estimating remaining useful life of
rolling element bearings. *IEEE Transactions on Reliability, 69*(1), 401–412. https://doi.org/10.1109/TR.2018.2882682
