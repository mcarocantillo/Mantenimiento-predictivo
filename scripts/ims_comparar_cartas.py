"""Fase 6 (objetivo 3): ¿qué carta (I-MR, X̄-R, X̄-S) relaciona mejor alarma y falla?

Dos comparaciones pareadas sobre los mismos rodamientos y los mismos instantes:
  1. Sin reglas: AUC del estadístico continuo de cada carta
     - posición: |z| de la carta I o X̄;
     - dispersión: MR/MR̄, R/R̄ o S/S̄.
     Positivos = puntos de las últimas H h de los 4 rodamientos que fallan; negativos = todo el
     monitoreo de los 8 sanos. Cada rodamiento pesa lo mismo (pesos 1/n puntos).
  2. Con reglas: J de Youden (fase 4) de N1, N2, N5, N6 y disp en cada carta.
Incertidumbre: bootstrap por rodamiento (se remuestrean con reemplazo los 4 que fallan y los
8 sanos, B veces) y en cada réplica se comparan las tres cartas sobre la misma muestra.
No uso el test de DeLong (notas de Lihki, 9.9.1) porque supone puntos independientes, y aquí
puntos de un mismo rodamiento están muy autocorrelacionados: la unidad real es el rodamiento.

Uso: python scripts/ims_comparar_cartas.py  (escribe comparacion_cartas.csv)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

H, B, SEMILLA = 48, 2000, 0
CARTAS = ["I-MR", "Xbar-R", "Xbar-S"]
PARES = [("Xbar-S", "I-MR"), ("Xbar-R", "I-MR"), ("Xbar-S", "Xbar-R")]
REGLAS = ["N1", "N2", "N5", "N6", "disp"]

al = pd.read_csv("alarmas_ims.csv")
mon = al[al["fase"] == "monitoreo"].copy()
mon["abs_z"] = mon["z"].abs()
falla_ids = sorted({(e, r) for e, r in mon.loc[mon["falla"], ["ensayo", "rod"]].itertuples(index=False)})
sano_ids = sorted({(e, r) for e, r in mon.loc[~mon["falla"], ["ensayo", "rod"]].itertuples(index=False)})

# Puntos relevantes por rodamiento: últimas H h si falla, todo el monitoreo si está sano
datos = {}
for (e, r, v, c), g in mon.groupby(["ensayo", "rod", "var", "carta"]):
    if (e, r) in falla_ids:
        g = g[g["horas_al_fin"] <= H]
    datos[(e, r, v, c)] = g


def auc(v, c, score, fallan, sanos):
    xs, ys, ws = [], [], []
    for ids, y in [(fallan, 1), (sanos, 0)]:
        for e, r in ids:
            s = datos[(e, r, v, c)][score].to_numpy()
            s = s[~np.isnan(s)]
            xs.append(s); ys.append(np.full(len(s), y)); ws.append(np.full(len(s), 1 / len(s)))
    return roc_auc_score(np.concatenate(ys), np.concatenate(xs), sample_weight=np.concatenate(ws))


def j(v, c, regla, fallan, sanos):
    sens = np.mean([datos[(e, r, v, c)][regla].mean() for e, r in fallan])
    fpr = np.mean([datos[(e, r, v, c)][regla].mean() for e, r in sanos])
    return sens - fpr


metricas = [("AUC", v, s, None) for v in ["rms", "kurt"] for s in ["abs_z", "disp_rel"]] + \
           [("J", v, None, regla) for v in ["rms", "kurt"] for regla in REGLAS]


def evaluar(fallan, sanos):
    out = {}
    for tipo, v, score, regla in metricas:
        for c in CARTAS:
            out[(tipo, v, score or regla, c)] = (auc(v, c, score, fallan, sanos) if tipo == "AUC"
                                                 else j(v, c, regla, fallan, sanos))
    return out


real = evaluar(falla_ids, sano_ids)
rng = np.random.default_rng(SEMILLA)
boot = []
for b in range(B):
    f = [falla_ids[i] for i in rng.integers(0, len(falla_ids), len(falla_ids))]
    s = [sano_ids[i] for i in rng.integers(0, len(sano_ids), len(sano_ids))]
    boot.append(evaluar(f, s))
    if (b + 1) % 500 == 0:
        print(f"bootstrap {b + 1}/{B}", flush=True)
boot = pd.DataFrame(boot)

filas = []
for tipo, v, score, regla in metricas:
    nombre = score or regla
    fila = {"métrica": tipo, "var": v, "estadístico/regla": nombre}
    for c in CARTAS:
        k = (tipo, v, nombre, c)
        lo, hi = boot[k].quantile([0.025, 0.975])
        fila[c] = f"{real[k]:.2f} [{lo:.2f}, {hi:.2f}]"
    for a, b_ in PARES:
        d = boot[(tipo, v, nombre, a)] - boot[(tipo, v, nombre, b_)]
        dr = real[(tipo, v, nombre, a)] - real[(tipo, v, nombre, b_)]
        lo, hi = d.quantile([0.025, 0.975])
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        fila[f"{a} − {b_}"] = f"{dr:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.3f}"
    filas.append(fila)
res = pd.DataFrame(filas)
res.to_csv("comparacion_cartas.csv", index=False)

pd.set_option("display.width", 300, "display.max_colwidth", 40)
print(f"\nValor real [IC 95 % bootstrap por rodamiento, B={B}], ventana pre-falla {H} h")
for tipo in ["AUC", "J"]:
    t = res[res["métrica"] == tipo].drop(columns="métrica")
    print(f"\n=== {tipo} por carta ===")
    print(t[["var", "estadístico/regla", *CARTAS]].to_string(index=False))
    print(f"\n=== {tipo}: diferencias pareadas ===")
    print(t[["var", "estadístico/regla", *[f"{a} − {b_}" for a, b_ in PARES]]].to_string(index=False))
