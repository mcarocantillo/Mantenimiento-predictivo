# Reglas de Nelson y cartas de control para anticipar fallas de rodamientos

Evaluamos si un sistema de alarmas basado en las 8 reglas de Nelson, validado antes en otra industria, anticipa
fallas de rodamientos, y con qué carta de control (I-MR, X̄-R o X̄-S) lo hace mejor. Se usan los 3 ensayos del
conjunto IMS (Universidad de Cincinnati): 4 rodamientos que fallan y 8 sanos como control.

Autores: María Carolina Cantillo Orozco, Andrés Aldana y Jenaan Tarabein (Universidad del Norte).

## Resultados principales

| Alarma (variable · carta · regla) | Sensibilidad 48 h | Falsas alarmas en sanos | J | Anticipación mediana |
| --- | --- | --- | --- | --- |
| RMS · X̄-S · dispersión (carta S) | 57 % | 2 % | 0,55 | 86 h |
| Curtosis · I-MR · N1 | 60 % | 7 % | 0,53 | 117 h |
| RMS · I-MR · N1 (método actual) | 84 % | 49 % | 0,35 | 112 h |

- La carta S sobre el RMS es la que mejor separa los rodamientos que fallan (AUC 0,86 frente a 0,73 de la MR).
- N3, N4, N7 y N8 no se asocian con la falla en ninguna carta.
- El criterio de estabilidad de la industria de origen (CV de 10 % a 30 %) no se transfiere: en estos rodamientos
  el estado sano tiene CV ~1 %.
- Con 4 fallas, ninguna alarma sobrevive la corrección por 54 comparaciones (Holm) en el objetivo 1; hace falta
  un segundo conjunto de datos para confirmar.

## Estructura

| Archivo | Qué hace |
| --- | --- |
| `scripts/ims_features.py` | Indicadores por archivo IMS: RMS, pico, cresta, curtosis, bandas y envolvente (BPFO, BPFI, BSF, FTF en ±5 %) |
| `scripts/ims_huecos.py` | Huecos de registro y saltos de nivel tras los reinicios |
| `scripts/ims_baseline.py` | Fase 2: baseline de 24 h con menor CV y límites I-MR |
| `scripts/ims_subgrupos.py` | 20 subbloques por segundo para X̄-R y X̄-S |
| `scripts/nelson.py` | Las 8 reglas de Nelson, idénticas a la implementación del código original (Code01) |
| `scripts/ims_cartas.py` | Fase 3: cartas I-MR, X̄-R y X̄-S y alarmas |
| `scripts/ims_reglas_falla.py` | Fase 4 (objetivo 1): qué regla corresponde a falla, con permutación y Holm/BH |
| `scripts/ims_correlacion.py` | Fase 5 (objetivo 2): correlación temporal y épocas superpuestas |
| `scripts/ims_comparar_cartas.py` | Fase 6 (objetivo 3): AUC y J por carta con bootstrap por rodamiento |
| `scripts/ims_precursores.py` | Análisis inicial del ensayo 1 con los límites del código original |
| `scripts/construir_eda.py` | Genera `eda_ims.ipynb`, el análisis exploratorio de los 3 ensayos |
| `scripts/features.py`, `scripts/analisis.py`, `scripts/construir_notebook.py` | Ejercicio previo con el dataset hidráulico de UCI (`mantenimiento_predictivo.ipynb`) |
| `scripts/llenar_cimga.py` | Llena el formulario de resumen de CIMGA 2027 (`entrega_cimga/`) |
| `ims*_features.csv` | Indicadores ya calculados de los ensayos 1, 2 y 3 |
| `baseline_ims.csv`, `limites_cartas.csv` | Baseline y límites de cada carta |
| `reglas_falla.csv`, `correlacion_temporal.csv`, `comparacion_cartas.csv` | Resultados de las fases 4, 5 y 6 |
| `fig_epocas.png` | Tasa de alarma en la última semana, rodamientos que fallan frente a sanos |
| `html/eda_ims.html`, `html/mantenimiento_predictivo.html` | Los dos notebooks exportados a HTML, con figuras, para verlos sin Jupyter (descargar y abrir en el navegador) |

## Cómo reproducir

Requiere Python con `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `nbformat` y `nbclient`.

1. Descargar el conjunto IMS (NASA Prognostics Data Repository, "IMS Bearings") y extraer los 3 ensayos.
2. Calcular los indicadores (si no se usan los `ims*_features.csv` incluidos) y los subgrupos:
   `python scripts/ims_features.py <carpeta del ensayo> ims_features.csv` y
   `python scripts/ims_subgrupos.py <carpeta del ensayo> sub1.csv` (igual para los ensayos 2 y 3).
3. Correr en orden: `ims_baseline.py`, `ims_cartas.py`, `ims_reglas_falla.py`, `ims_correlacion.py`,
   `ims_comparar_cartas.py`.
4. Para el EDA: `python scripts/construir_eda.py` y ejecutar `eda_ims.ipynb` (ajustar la ruta `RAIZ` a los datos).

El ejercicio con UCI usa el dataset [Condition monitoring of hydraulic systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems),
extraído en `data/`.

## Notas de calidad de datos

- Se excluyen 3 archivos con señal muerta al final de los ensayos 2 y 3 (máquina detenida).
- El ensayo 1 tiene 24 huecos de registro; solo 341 de sus 828 h tienen datos.
- La frecuencia de defecto real del ensayo 2 es ~231 Hz y no 236,4 Hz; por eso la envolvente busca en ±5 %.
- El Readme oficial del conjunto IMS dice que el ensayo 3 tiene 4448 archivos y termina el 4 de abril de 2004,
  pero el archivo descargado tiene 6324 archivos y llega al 18 de abril de 2004. Aquí se toma como momento de
  la falla el último archivo con señal (18 de abril): la curtosis y el RMS de B3 solo se disparan en sus últimas
  horas y el 4 de abril no muestra ningún cambio.
