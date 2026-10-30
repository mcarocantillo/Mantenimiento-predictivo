"""Genera mantenimiento_predictivo.ipynb a partir de celdas definidas aquí."""
import nbformat as nbf

celdas = []
md = lambda s: celdas.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: celdas.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Mantenimiento predictivo: ¿puede el ML anticiparse a las alarmas por σ?

**Pregunta.** Hoy se usa una alarma cuando la señal del motor sale de μ ± 1.5σ y otra cuando sale de μ ± 3σ.
¿Puede el Machine Learning detectar la degradación **antes**, en sus grados leves, cuando las alarmas σ todavía no se disparan?

**Contexto.** Este fue un ejercicio previo, con los datos hidráulicos que compartió el profesor Aldana, para
probar la idea general antes de trabajar con los rodamientos IMS: comparar las alarmas por σ con modelos de Machine
Learning.

**Datos.** *Condition monitoring of hydraulic systems* (UCI, ZeMA): 2205 ciclos de 60 s de un banco hidráulico,
17 sensores. La señal que se vigila es **EPS1, la potencia del motor (W)**, que es la variable sobre la que se
calculan las alarmas de 1.5σ y 3σ.

**Una advertencia sobre el dataset.** No es un registro de fallas que avancen con el tiempo: los investigadores
fijaron a mano el estado de cuatro componentes (enfriador, válvula, bomba, acumulador) en varios grados. Por eso la
pregunta equivalente a "anticiparse" es: **¿se detectan los grados leves de degradación antes de llegar a la falla
grave?**

**Dónde está el código.** Las características de cada ciclo se calculan con [features.py](https://github.com/mcarocantillo/Mantenimiento-predictivo/blob/main/scripts/features.py), la
comparación de alarmas y modelos con [analisis.py](https://github.com/mcarocantillo/Mantenimiento-predictivo/blob/main/scripts/analisis.py), y este notebook se genera con
[construir_notebook.py](https://github.com/mcarocantillo/Mantenimiento-predictivo/blob/main/scripts/construir_notebook.py). Los datos originales van en la carpeta `data/`.
""")

code(r"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

DATOS = Path("data")            # carpeta con los .txt del zip de UCI
CARACT = Path("features.csv")   # se genera si no existe

if not CARACT.exists():
    sys.path.insert(0, "scripts")
    from features import main as extraer
    extraer(DATOS, CARACT)

df = pd.read_csv(CARACT, index_col="ciclo")
ETIQUETAS = ["cooler", "valve", "pump", "acc", "stable"]
X_all = df.drop(columns=ETIQUETAS)
X_eps = X_all.filter(like="EPS1_")
estable = (df["stable"] == 0).to_numpy()
print(df.shape, "| características EPS1:", X_eps.shape[1], "| todas:", X_all.shape[1])
""")

code(r"""
# Estilo de gráficas: marcas delgadas, ejes y rejilla discretos, colores categóricos fijos
AZUL, NARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, TENUE, REJILLA, FONDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": TENUE, "axes.labelcolor": TINTA2, "axes.titlecolor": TINTA,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": REJILLA, "grid.linewidth": 0.6,
    "xtick.color": TENUE, "ytick.color": TENUE, "xtick.labelcolor": TINTA2, "ytick.labelcolor": TINTA2,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.dpi": 110,
})

# Componentes: valor sano y grados de degradación de leve a casi falla total
COMP = {
    "valve":  ("Válvula",     100, [90, 80, 73]),
    "pump":   ("Bomba",       0,   [1, 2]),
    "acc":    ("Acumulador",  130, [115, 100, 90]),
    "cooler": ("Enfriador",   100, [20, 3]),
}
""")

md(r"""
### Términos clave en palabras simples

| Término | Qué es |
| --- | --- |
| Ciclo | Una repetición de 60 s del trabajo del banco hidráulico. Cada fila de los datos es un ciclo. |
| Sensor | Instrumento que mide algo (presión, caudal, temperatura, potencia) muchas veces por segundo. |
| Potencia del motor (EPS1) | Cuánta energía por segundo consume el motor para mover el banco, en vatios (W). Si un componente se daña, el motor tiene que esforzarse distinto y su potencia cambia. |
| Degradación y grado | Qué tan dañado está un componente. Por ejemplo, la bomba tiene tres grados: sin fuga, fuga leve y fuga severa. |
| μ y σ | μ es el valor promedio de la señal en ciclos sanos; σ (desviación estándar) dice cuánto se aleja normalmente de ese promedio. |
| Alarma 1.5σ y 3σ | Suena si la señal se aleja del promedio más de 1,5 o 3 veces lo normal. La de 3σ es más estricta. |
| Detección | Porcentaje de ciclos con daño en que la alarma o el modelo sí avisa. Más alto es mejor. |
| Falsa alarma | Porcentaje de ciclos sanos en que avisa por error. Más bajo es mejor. |
| Machine Learning supervisado | El modelo aprende de ejemplos ya etiquetados ("este ciclo era sano", "este estaba dañado") y luego clasifica ciclos nuevos. |
| RandomForest | Modelo supervisado que combina cientos de árboles de decisión, cada uno con reglas simples del tipo "si la potencia máxima es mayor que X…". |
| Característica | Un número que resume la señal de un ciclo, como su promedio, su máximo o cuánto varía. |
| Validación por bloques (GroupKFold) | El modelo se prueba con ciclos que no vio al entrenar, separados por bloques de tiempo, para que no "haga trampa" con ciclos casi idénticos. |
| No supervisado (IsolationForest) | El modelo solo ve ciclos sanos y aprende cómo es lo normal; marca como anomalía lo que se aleja de eso. No necesita ejemplos de fallas. |
| Percentil 99 | El valor por debajo del cual queda el 99 % de los datos; se usa como umbral para decidir qué es raro. |
""")

md(r"""
## 1. Cómo se diseñó el experimento

Cada línea es el estado de un componente a lo largo de los 2205 ciclos. Los grados cambian **en escalones fijados
a mano** (la válvula incluso va de peor a mejor). No hay una trayectoria natural hacia la falla, así que no se puede
estudiar "qué pasa justo antes del cruce" en el tiempo.
""")

code(r"""
fig, axs = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
for ax, (c, (nombre, sano, grados)) in zip(axs, COMP.items()):
    ax.step(df.index, df[c], where="post", color=AZUL, lw=1.5)
    ax.set_yticks([sano] + grados)
    ax.set_ylabel(nombre, rotation=0, ha="right", va="center")
    ax.grid(axis="y")
axs[0].set_title("Estado de cada componente por ciclo (diseño del experimento)", loc="left")
axs[-1].set_xlabel("Ciclo (60 s cada uno)")
fig.tight_layout()
plt.show()
""")

md(r"""
**Qué quiere decir.** En este banco los daños no aparecen poco a poco: los investigadores los pusieron a propósito y
en escalones. Por eso aquí no se puede medir con cuánta anticipación avisa una alarma; solo si detecta los daños
leves antes que los graves.
""")

md(r"""
## 2. La señal del motor dentro de un ciclo

Promedio de la potencia (EPS1) a lo largo del ciclo, con la bomba sana frente a la bomba con fuga severa
(solo ciclos estables y con la válvula y el acumulador sanos, para aislar el efecto de la bomba).
""")

code(r"""
eps = pd.read_csv(DATOS / "EPS1.txt", sep="\t", header=None).to_numpy()
t = np.arange(eps.shape[1]) / 100  # 100 Hz
base = estable & (df["valve"] == 100).to_numpy() & (df["acc"] == 130).to_numpy()

fig, ax = plt.subplots(figsize=(10, 3.6))
for g, color, etiqueta in [(0, AZUL, "Bomba sana"), (2, NARANJA, "Fuga severa")]:
    m = base & (df["pump"] == g).to_numpy()
    ax.plot(t, eps[m].mean(0), color=color, lw=1.2)
    ax.annotate(f"{etiqueta} (n={m.sum()})", (t[-1], eps[m].mean(0)[-300:].mean()),
                xytext=(6, 0), textcoords="offset points", color=TINTA2, va="center")
ax.set_xlim(0, 68)
ax.set_title("Potencia del motor dentro del ciclo de carga", loc="left")
ax.set_xlabel("Tiempo dentro del ciclo (s)"); ax.set_ylabel("Potencia (W)")
fig.tight_layout(); plt.show()
""")

md(r"""
**Qué quiere decir.** Con la bomba dañada, el motor gasta un poco más de potencia en ciertas partes del ciclo, no en
todo el ciclo. Una sola cifra por ciclo (el promedio) puede esconder ese cambio; por eso el modelo de ML usa varias
cifras de cada ciclo.
""")

md(r"""
## 3. Las alarmas por σ sobre la potencia media de cada ciclo

Aquí se reproduce el sistema de alarmas actual: el promedio (μ) y la desviación estándar (σ) se calculan con los ciclos donde el componente está sano, y cada ciclo dispara alarma si
su potencia media sale de μ ± 1.5σ (línea discontinua) o μ ± 3σ (punteada).

Las franjas horizontales son los tres estados del **enfriador**: cambian la potencia mucho más que la degradación que
buscamos. La degradación sí desplaza los puntos (en la bomba se ve claro), pero el desplazamiento queda dentro de la
dispersión que causan los otros componentes, así que un umbral fijo no puede separarlos.
""")

code(r"""
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
rng = np.random.default_rng(0)
x = df["EPS1_mean"]
for ax, c in zip(axs, ["valve", "pump", "acc"]):
    nombre, sano, grados = COMP[c]
    ref = x[(df[c] == sano) & estable]
    mu, sd = ref.mean(), ref.std()
    for k, s in [(1.5, "--"), (3, ":")]:
        for signo in (-1, 1):
            ax.axhline(mu + signo * k * sd, color=TINTA2, lw=0.8, ls=s)
        ax.text(len(grados) + 0.85, mu + k * sd, f"+{k}σ", color=TINTA2, va="bottom", ha="right",
                fontsize=8.5)
    for i, g in enumerate([sano] + grados):
        v = x[(df[c] == g) & estable]
        ax.scatter(i + rng.uniform(-0.25, 0.25, len(v)), v, s=8, alpha=0.35,
                   color=AZUL if g == sano else NARANJA, linewidths=0)
    ax.set_xticks(range(len(grados) + 1), ["sano"] + [str(g) for g in grados])
    ax.set_title(nombre, loc="left"); ax.set_xlim(-0.6, len(grados) + 0.9)
axs[0].set_ylabel("Potencia media del ciclo (W)")
fig.suptitle("Las alarmas σ no separan los grados de degradación", x=0.01, ha="left",
             fontweight="bold", color=TINTA)
fig.tight_layout(); plt.show()
""")

md(r"""
**Qué quiere decir.** La alarma de 3σ casi nunca suena aunque el componente esté muy dañado, y la de 1.5σ suena casi
igual de seguido con el componente sano que con el dañado. En la práctica, este sistema de alarmas solo detectaría la
falla del enfriador, que cambia mucho la potencia del motor.
""")

md(r"""
## 4. ML supervisado: RandomForest

Para cada componente se entrena un clasificador "sano / degradado" de dos formas: solo con las 13 características de
EPS1 (la señal del motor) y con las 221 de todos los sensores.

**Validación honesta.** Los ciclos vecinos son casi idénticos; una división al azar inflaría los resultados. Por eso
se agrupan bloques contiguos de condiciones y se usa `GroupKFold`: el modelo siempre se evalúa en bloques que no vio.
Todas las tasas se reportan sobre ciclos estables, igual que las de σ.
""")

code(r"""
seg = (df[ETIQUETAS[:4]] != df[ETIQUETAS[:4]].shift()).any(axis=1).cumsum()
grupos = seg // 4

def tasa_sigma(c, k=1.5):
    nombre, sano, grados = COMP[c]
    x = df["EPS1_mean"]
    ref = x[(df[c] == sano) & estable]
    z = (x - ref.mean()).abs() / ref.std()
    return {g: 100 * (z[(df[c] == g) & estable] > k).mean() for g in [sano] + grados}

def tasa_rf(c, X):
    nombre, sano, grados = COMP[c]
    y = (df[c] != sano).astype(int)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)
    pred = cross_val_predict(rf, X, y, groups=grupos, cv=GroupKFold(5))
    return {g: 100 * pred[(df[c] == g).to_numpy() & estable].mean() for g in [sano] + grados}

filas = []
for c in COMP:
    for metodo, r in [("Alarma 1.5σ", tasa_sigma(c, 1.5)), ("Alarma 3σ", tasa_sigma(c, 3)),
                      ("ML solo EPS1", tasa_rf(c, X_eps)), ("ML todos los sensores", tasa_rf(c, X_all))]:
        for g, v in r.items():
            filas.append({"componente": COMP[c][0], "grado": g,
                          "estado": "sano" if g == COMP[c][1] else "degradado",
                          "método": metodo, "% ciclos con alarma": round(v, 1)})
res = pd.DataFrame(filas)
orden = [(n, g) for n, s, gs in COMP.values() for g in [s] + gs]
tabla = (res.pivot(index=["componente", "grado"], columns="método", values="% ciclos con alarma")
            .reindex(orden)[[m for m in res["método"].unique()]])
tabla
""")

md(r"""
En la fila del grado **sano**, el porcentaje son **falsas alarmas**; en las demás, es la tasa de **detección**.
""")

code(r"""
METODOS = [("Alarma 1.5σ", AZUL), ("ML solo EPS1", NARANJA), ("ML todos los sensores", AGUA)]
fig, axs = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True, layout="constrained",
                        gridspec_kw={"width_ratios": [4, 3, 4]})
ancho = 0.28
for ax, c in zip(axs, ["valve", "pump", "acc"]):
    nombre, sano, grados = COMP[c]
    orden = [sano] + grados
    for j, (m, color) in enumerate(METODOS):
        sub = res[(res["componente"] == nombre) & (res["método"] == m)].set_index("grado")
        vals = sub.loc[orden, "% ciclos con alarma"].to_numpy()
        xs = np.arange(len(orden)) + (j - 1) * ancho
        ax.bar(xs, vals, ancho, color=color, edgecolor=FONDO, linewidth=1.5, label=m)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, color=TINTA2)
    ax.axvspan(-0.5, 0.5, color=REJILLA, alpha=0.45, zorder=0, lw=0)
    ax.set_xticks(range(len(orden)), ["sano"] + [str(g) for g in grados])
    ax.set_xlabel("grado (de leve a casi falla →)")
    ax.set_title(nombre, loc="left"); ax.set_ylim(0, 112)
axs[0].set_ylabel("% de ciclos con alarma")
fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
fig.suptitle("Detección por grado de degradación: σ frente a ML  "
             "(franja gris = componente sano, es decir, falsas alarmas)",
             x=0.01, ha="left", fontweight="bold", color=TINTA)
plt.show()
""")

md(r"""
**Qué quiere decir.** Con la misma señal del motor, el modelo de ML detecta daños que las alarmas σ no ven: por
ejemplo, el 75 % de los ciclos con fuga leve de la bomba, con 8 % de falsas alarmas. En el acumulador y la válvula, en
cambio, da muchas falsas alarmas si solo usa la señal del motor; con los demás sensores del banco casi desaparecen.
""")

md(r"""
### ¿Qué mira el modelo en la señal del motor?

Importancia de las características de EPS1 para detectar la fuga de la bomba. Las características son: media, σ,
mínimo, máximo, pico a pico, pendiente, asimetría, curtosis, variación entre muestras (`rms_diff`) y la media de cada
cuarto del ciclo (`q1`–`q4`).
""")

code(r"""
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)
rf.fit(X_eps, (df["pump"] != 0).astype(int))
imp = pd.Series(rf.feature_importances_, index=X_eps.columns.str.replace("EPS1_", "")).sort_values()

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp.index, imp.values, color=AZUL, height=0.6)
for y, v in enumerate(imp.values):
    ax.text(v + 0.003, y, f"{v:.2f}", va="center", fontsize=8, color=TINTA2)
ax.grid(axis="y", visible=False); ax.grid(axis="x")
ax.set_title("Importancia de cada característica de EPS1 (bomba)", loc="left")
ax.set_xlabel("Importancia (reducción media de impureza)")
fig.tight_layout(); plt.show()
""")

md(r"""
**Qué quiere decir.** El modelo no se fija en una sola cifra: combina el nivel de potencia, su máximo y el nivel en
distintas partes del ciclo. Esa combinación es lo que le permite separar el daño de la bomba de los cambios que causan
los otros componentes.
""")

md(r"""
## 5. Detección de anomalías no supervisada: IsolationForest

Se entrena solo con ciclos donde el componente está sano (primera mitad en el tiempo) y el umbral es el percentil 99
de su puntaje. La segunda mitad de los ciclos sanos mide las falsas alarmas.

**Por qué aquí sale mal.** Solo hay 21 ciclos con *todo* sano. Cuando un componente está sano, los otros tres sí
varían, y además cambian con el tiempo, así que la referencia "normal" no es limpia: el modelo confunde cambios de
los otros componentes con anomalías. En un motor real con meses de operación normal este enfoque tiene mucho más
sentido.
""")

code(r"""
filas = []
for nombre_x, X in [("solo EPS1", X_eps), ("todos los sensores", X_all)]:
    for c, (nombre, sano, grados) in COMP.items():
        idx = np.where(((df[c] == sano).to_numpy()) & estable)[0]
        tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
        sc = StandardScaler().fit(X.iloc[tr])
        iso = IsolationForest(n_estimators=300, random_state=0).fit(sc.transform(X.iloc[tr]))
        s = -iso.score_samples(sc.transform(X))
        umbral = np.percentile(s[tr], 99)
        fila = {"datos": nombre_x, "componente": nombre, "falsas alarmas %": 100 * (s[te] > umbral).mean()}
        for g in grados:
            m = (df[c] == g).to_numpy() & estable
            fila[f"detección {g}"] = 100 * (s[m] > umbral).mean()
        filas.append(fila)
# Una tabla por componente evita columnas vacías (cada componente tiene sus propios grados)
for nombre in [v[0] for v in COMP.values()]:
    t = pd.DataFrame([f for f in filas if f["componente"] == nombre])
    display(t.drop(columns="componente").set_index("datos").dropna(axis=1).round(1)
             .style.set_caption(f"{nombre}: % de ciclos marcados como anómalos"))
""")

md(r"""
**Qué quiere decir.** El enfoque no supervisado falla aquí porque no hay un periodo largo en que todo esté sano para
aprender qué es lo normal. No significa que el método no sirva: con meses de operación normal registrada es una opción
útil, porque no necesita esperar a que ocurran fallas para aprender.
""")

md(r"""
## 6. Conclusiones

- **Las alarmas σ casi no sirven para anticiparse.** La de 3σ dispara en menos del 2 % de los ciclos de válvula,
  bomba y acumulador, incluso cerca de la falla total. La de 1.5σ suena, pero casi igual de seguido con el componente
  sano. Solo detectan bien el enfriador, cuya falla cambia mucho el nivel de potencia.
- **El ML supervisado con solo la señal del motor ya detecta degradación leve**: 75 % de la fuga leve de la bomba
  (8 % de falsas alarmas) y 89 % del acumulador ligeramente bajo. La ventaja es que **combina varias
  características** (nivel, máximo, nivel en cada fase del ciclo, dispersión) y así separa el efecto de la falla del
  de las otras condiciones de operación; un umbral σ sobre una sola cifra no puede hacerlo.
- **Con solo EPS1 hay demasiadas falsas alarmas en el acumulador (48 %) y la válvula (25 %).** Al sumar presión y
  caudal bajan a 13 % y 1 %: conviene registrar más variables del motor (corriente, velocidad, temperatura,
  vibración).
- **El no supervisado necesita una referencia sana limpia**, que este dataset no tiene.

## 7. Qué se puede hacer con los datos disponibles

Sin agregar sensores ni modificar el equipo:

1. Calcular μ y σ solo con un periodo sano, para no usar información del futuro.
2. Describir cada ciclo con varias características (promedio, máximo, nivel en cada parte del ciclo), no con una sola
   cifra: es lo que le permite al ML ver lo que la alarma σ no ve.
3. Si los registros tienen fecha y hora, marcar las ventanas previas a cada alarma ("pre-falla") y entrenar un
   clasificador que avise antes de que la alarma suene, validando por eventos.
4. Entrenar un detector no supervisado con los periodos sanos registrados y comprobar si su puntaje sube antes de las
   alarmas σ.
""")

nb = nbf.v4.new_notebook(cells=celdas, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
nbf.write(nb, "mantenimiento_predictivo.ipynb")
print("notebook escrito con", len(celdas), "celdas")
