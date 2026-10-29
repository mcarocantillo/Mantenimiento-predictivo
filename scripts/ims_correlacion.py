"""Fase 5 (objetivo 2): correlación temporal entre las alarmas y la llegada de la falla.

- Tasa de alarma por bloques de BLOQUE horas de reloj (solo monitoreo).
- ρ de Spearman entre la tasa y la cercanía al fin (−horas al fin) para cada rodamiento:
  ρ > 0 = la alarma se vuelve más frecuente a medida que se acerca la falla.
- Estadístico = ρ medio de los que fallan − ρ medio de los sanos. Si sube igual en todos los
  rodamientos (vibración de toda la máquina), la diferencia es ~0: no señala la falla.
- p por permutación exacta de qué rodamiento falla (96 asignaciones) + Holm y BH.
- Épocas superpuestas: tasa media por bloque en las últimas EPOCA horas, fallan vs sanos.

Uso: python scripts/ims_correlacion.py  (lee alarmas_ims.csv; escribe correlacion_temporal.csv
y fig_epocas.png)
"""
from itertools import combinations, product

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLOQUE, EPOCA = 6, 168
REGLAS = [f"N{i}" for i in range(1, 9)] + ["disp"]
FALLAN = {1: ("B3", "B4"), 2: ("B1",), 3: ("B3",)}
RODS = ["B1", "B2", "B3", "B4"]
AZUL, NARANJA = "#2a78d6", "#eb6834"

al = pd.read_csv("alarmas_ims.csv")
mon = al[al["fase"] == "monitoreo"].copy()
mon["bloque"] = (mon["horas_al_fin"] // BLOQUE) * BLOQUE + BLOQUE / 2
tasa = mon.groupby(["var", "carta", "ensayo", "rod", "bloque"])[REGLAS].mean()


def rho(serie):
    s = serie.dropna()
    return spearmanr(-s.index.get_level_values("bloque"), s.to_numpy()).statistic if s.nunique() > 1 else 0.0


asigs = [{1: a, 2: b, 3: c} for a, b, c in
         product(combinations(RODS, 2), combinations(RODS, 1), combinations(RODS, 1))]
filas = []
for (v, carta), t in tasa.groupby(level=["var", "carta"]):
    for r in REGLAS:
        rhos = t[r].groupby(level=["ensayo", "rod"]).apply(rho)
        def dif(asig):
            falla = np.array([rod in asig[e] for e, rod in rhos.index])
            return rhos.to_numpy()[falla].mean() - rhos.to_numpy()[~falla].mean()
        real = dif(FALLAN)
        nulo = np.array([dif(a) for a in asigs])
        falla = np.array([rod in FALLAN[e] for e, rod in rhos.index])
        filas.append({"var": v, "carta": carta, "regla": r,
                      "rho_fallan": rhos[falla].mean(), "rho_sanos": rhos[~falla].mean(),
                      "dif": real, "p_perm": (nulo >= real - 1e-12).mean(),
                      **{f"rho_{e}{rod}": rhos[(e, rod)] for e, rod in rhos.index if rod in FALLAN[e]}})
res = pd.DataFrame(filas)
m = len(res)
orden = res["p_perm"].sort_values().index
res.loc[orden, "p_holm"] = np.minimum(np.maximum.accumulate([(m - i) * res.loc[k, "p_perm"]
                                                            for i, k in enumerate(orden)]), 1)
bh = [res.loc[k, "p_perm"] * m / (i + 1) for i, k in enumerate(orden)]
res.loc[orden, "p_bh"] = np.minimum(np.minimum.accumulate(bh[::-1])[::-1], 1)
res = res.sort_values("dif", ascending=False)
res.to_csv("correlacion_temporal.csv", index=False)

pd.set_option("display.width", 250)
print(f"ρ de Spearman (tasa de alarma por bloques de {BLOQUE} h vs cercanía a la falla)")
print(res.head(15).round(3).to_string(index=False))
print("\nCarta del profe (RMS · I-MR):")
print(res[(res["var"] == "rms") & (res["carta"] == "I-MR")].round(3).to_string(index=False))

# Épocas superpuestas para las alarmas que interesan
CASOS = [("rms", "Xbar-S", "disp", "RMS · carta S (dispersión)"),
         ("kurt", "I-MR", "N1", "Curtosis · I-MR · N1"),
         ("rms", "I-MR", "N1", "RMS · I-MR · N1 (método actual)")]
fig, ejes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, layout="constrained")
for ax, (v, carta, r, titulo) in zip(ejes, CASOS):
    t = tasa.loc[(v, carta), r].reset_index()
    t = t[t["bloque"] <= EPOCA]
    t["falla"] = [rod in FALLAN[e] for e, rod in zip(t["ensayo"], t["rod"])]
    for falla, color, nombre in [(True, NARANJA, "Rodamientos que fallan (4)"),
                                 (False, AZUL, "Rodamientos sanos (8)")]:
        g = t[t["falla"] == falla].groupby("bloque")[r]
        media, q1, q3 = 100 * g.mean(), 100 * g.quantile(0.25), 100 * g.quantile(0.75)
        ax.fill_between(-media.index, q1, q3, color=color, alpha=0.15, lw=0)
        ax.plot(-media.index, media, color=color, lw=2, label=nombre)
    ax.set_title(titulo, fontsize=11, loc="left")
    ax.set_xlabel("Horas antes del fin del ensayo")
    ax.set_xlim(-EPOCA, 0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
ejes[0].set_ylabel("% del tiempo con alarma")
ejes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Épocas superpuestas: tasa de alarma en la última semana (media y rango intercuartil)",
             fontsize=12, x=0.01, ha="left")
fig.savefig("fig_epocas.png", dpi=150)
print("\nguardado fig_epocas.png")
