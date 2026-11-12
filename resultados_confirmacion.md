# Resultados de la confirmación con XJTU-SY y FEMTO/PRONOSTIA

Fecha: 12 de noviembre de 2026. Resultados obtenidos aplicando sin cambios el
[protocolo de confirmación](protocolo_confirmacion.md), registrado antes de calcularlos (con la aclaración de
implementación documentada en su sección 7). Código: [scripts/confirmacion.py](scripts/confirmacion.py) e
[scripts/indicadores_confirmacion.py](scripts/indicadores_confirmacion.py).

## Resumen

**La hipótesis principal se confirma.** La alarma de dispersión de la carta X̄-S sobre el RMS (carta S), elegida con
los datos IMS y aplicada sin ningún ajuste a 30 rodamientos nuevos, avisó en el 10 % final de la vida en **29 de 30**
rodamientos. Por pura casualidad se esperaría que avisara en el 60 % (65 % con el cálculo estricto). La probabilidad de
obtener 29 de 30 por azar es menor que 0,0001.

## Prueba principal (canal horizontal, 10 % final de la vida)

Se excluyeron, según la regla fijada, los dos rodamientos de XJTU-SY con menos de 100 archivos (Bearing1_5 y
Bearing2_4).

| Conjunto | Rodamientos | Aciertos de la carta S | Aciertos esperados por azar | p | p (estricta) |
| --- | --- | --- | --- | --- | --- |
| XJTU-SY | 13 | 12 | 24 % | < 0,0001 | < 0,0001 |
| FEMTO | 17 | 17 | 78 % | 0,015 | 0,040 |
| **Los dos juntos** | **30** | **29** | **60 %** | **< 0,0001** | **< 0,0001** |

Criterio del protocolo: p < 0,05 en el análisis conjunto y más aciertos que los esperados por azar en cada conjunto.
**Se cumple.**

**Qué quiere decir.** En FEMTO la prueba es más exigente, porque ahí la carta S da bastantes falsas alarmas y por eso
acertar por casualidad es más fácil (78 %); aun así pasa. En XJTU-SY la diferencia es muy clara: por azar acertaría en
1 de cada 4 rodamientos y acertó en 12 de 13.

## Comparación con la carta I-MR del código original (RMS, regla 1)

| Medida (los dos conjuntos juntos) | Carta S | I-MR, regla 1 |
| --- | --- | --- |
| Aciertos | 29 de 30 | 30 de 30 |
| Aciertos esperados por azar | 60 % | 76 % |
| p (estricta) | < 0,0001 | 0,057 (no significativo) |
| Tiempo con falsa alarma en el tramo sano | 17 % | 40 % |
| Precisión (eventos dentro del tramo final) | 19 % | 14 % |
| Anticipación mediana | 18,7 % de la vida | 24,3 % de la vida |

**Qué quiere decir.** La I-MR con la regla 1 también avisa antes de la falla, pero suena tanto en el tramo sano (40 % del
tiempo) que acertar no prueba nada: con el cálculo estricto no se distingue del azar. La carta S suena menos de la
mitad de veces en el tramo sano y su acierto sí es significativo.

## Análisis de sensibilidad (los dos conjuntos juntos)

| Variante | Aciertos de la carta S | p (estricta) | I-MR con regla 1, p (estricta) |
| --- | --- | --- | --- |
| Principal: canal horizontal, 10 % final | 29/30 | < 0,0001 | 0,057 |
| Canal vertical | 30/30 | < 0,0001 | 0,116 |
| Tramo final del 5 % | 28/30 | < 0,0001 | 0,003 |
| Tramo final del 20 % | 29/30 | 0,006 | 0,342 |

La carta S es significativa en todas las variantes. La I-MR con la regla 1 solo lo es en una.

## Limitaciones

- **Falsas alarmas más altas que en IMS.** En el tramo sano, la carta S estuvo en alarma el 17 % del tiempo (14 % en
  FEMTO y 20 % en XJTU-SY), frente al 2 % en IMS. En FEMTO cada archivo dura solo 0,1 s, así que los 20 trozos con los
  que se calcula S son muy cortos y ruidosos. En XJTU-SY, en algunos rodamientos (Bearing1_2 y Bearing2_2) la alarma
  suena desde muy temprano, lo que sugiere que el deterioro empezó antes de la mitad de la prueba.
- **Precisión baja (19 %).** Por cada aviso dentro del tramo final hubo unos 4 fuera de él. Con un umbral fijo y
  pruebas de solo horas, la carta S sirve para identificar la etapa final, pero sola no basta para decidir una parada.
- **Momento de falla supuesto:** el último archivo de cada prueba.
- **Un rodamiento por prueba.** Las falsas alarmas se midieron en el tramo inicial de cada rodamiento, no en
  rodamientos sanos del mismo eje como en IMS.

## Referencias

Nectoux, P., Gouriveau, R., Medjaher, K., Ramasso, E., Chebel-Morello, B., Zerhouni, N. y Varnier, C. (2012).
PRONOSTIA: An experimental platform for bearings accelerated degradation tests. En *IEEE International Conference on
Prognostics and Health Management (PHM'12)*, Denver. https://hal.science/hal-00719503

Wang, B., Lei, Y., Li, N. y Li, N. (2020). A hybrid prognostics approach for estimating remaining useful life of
rolling element bearings. *IEEE Transactions on Reliability, 69*(1), 401–412. https://doi.org/10.1109/TR.2018.2882682
