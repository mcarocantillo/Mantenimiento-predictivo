"""Fase 4 (objetivo 1): ¿qué regla, en qué carta, corresponde a falla?

Unidad: cada rodamiento de cada ensayo (4 que fallan, 8 sanos), para que el ensayo 3
(6324 archivos) no pese más que los otros. Solo puntos de monitoreo (después del baseline).
  - sensibilidad: % de puntos con la alarma en las últimas H horas, en los que fallan;
  - FPR: % de puntos con la alarma en todo el monitoreo de los sanos;
  - J de Youden = sensibilidad − FPR (promedios por rodamiento);
  - evento = 3 puntos seguidos con alarma; falsos eventos por día en sanos;
  - anticipación: horas antes del fin del primer evento en la última semana (168 h).
Prueba de permutación exacta: se reasigna cuál rodamiento "falla" dentro de cada ensayo
(6 × 4 × 4 = 96 asignaciones). Si la regla de verdad señala al rodamiento que falla, la
asignación real debe tener una J mayor que casi todas. p mínimo posible = 1/96 ≈ 0.010.
Corrección de Holm y Benjamini-Hochberg sobre las 9 alarmas × 3 cartas × 2 variables.

Uso: python scripts/ims_reglas_falla.py   (lee alarmas_ims.csv, escribe reglas_falla.csv)
"""
from itertools import combinations, product

import numpy as np
import pandas as pd

VENTANAS = [24, 48, 96]
H_PRINCIPAL = 48
SEMANA, SEGUIDOS = 168, 3
REGLAS = [f"N{i}" for i in range(1, 9)] + ["disp"]
FALLAN = {1: ("B3", "B4"), 2: ("B1",), 3: ("B3",)}
RODS = ["B1", "B2", "B3", "B4"]

al = pd.read_csv("alarmas_ims.csv")
mon = al[al["fase"] == "monitoreo"]

# Por (ensayo, rod, var, carta, regla): tasas por ventana, FPR, eventos/día y anticipación
filas = []
for (ens, rod, v, carta), g in mon.groupby(["ensayo", "rod", "var", "carta"]):
    dias = (g["horas_al_fin"].iloc[0] - g["horas_al_fin"].iloc[-1]) / 24
    for r in REGLAS:
        a = g[r].to_numpy(bool)
        evento = pd.Series(a).rolling(SEGUIDOS).sum().eq(SEGUIDOS).to_numpy()
        inicio = evento & ~np.r_[False, evento[:-1]]
        h = g["horas_al_fin"].to_numpy()
        ult_semana = inicio & (h <= SEMANA)
        fila = {"ensayo": ens, "rod": rod, "var": v, "carta": carta, "regla": r,
                "fpr": a.mean(), "eventos_dia": inicio.sum() / dias,
                "anticip_h": h[ult_semana].max() if ult_semana.any() else 0.0}
        for H in VENTANAS:
            fila[f"sens_{H}"] = a[h <= H].mean()
        filas.append(fila)
tasas = pd.DataFrame(filas).set_index(["var", "carta", "regla", "ensayo", "rod"]).sort_index()


def asignaciones():
    for c1, c2, c3 in product(combinations(RODS, 2), combinations(RODS, 1), combinations(RODS, 1)):
        yield {1: c1, 2: c2, 3: c3}


def j_youden(t, asig, H):
    falla = np.array([rod in asig[ens] for ens, rod in t.index])
    return t[f"sens_{H}"].to_numpy()[falla].mean() - t["fpr"].to_numpy()[~falla].mean()


todas = list(asignaciones())
res = []
for (v, carta, regla), t in tasas.groupby(level=["var", "carta", "regla"]):
    t = t.droplevel(["var", "carta", "regla"])
    real = np.array([rod in FALLAN[ens] for ens, rod in t.index])
    fila = {"var": v, "carta": carta, "regla": regla,
            "FPR_sanos_%": 100 * t["fpr"][~real].mean(),
            "falsos_eventos_dia": t["eventos_dia"][~real].mean(),
            "anticip_h_mediana": t["anticip_h"][real].median(),
            "fallas_con_evento_semana": int((t["anticip_h"][real] > 0).sum())}
    for H in VENTANAS:
        fila[f"sens_{H}h_%"] = 100 * t[f"sens_{H}"][real].mean()
        fila[f"J_{H}h"] = j_youden(t, FALLAN, H)
    nulo = np.array([j_youden(t, a, H_PRINCIPAL) for a in todas])
    fila["p_perm"] = (nulo >= fila[f"J_{H_PRINCIPAL}h"] - 1e-12).mean()
    res.append(fila)
res = pd.DataFrame(res)

# Holm y BH
m = len(res)
orden = res["p_perm"].sort_values().index
holm = np.maximum.accumulate([(m - i) * res.loc[k, "p_perm"] for i, k in enumerate(orden)])
res.loc[orden, "p_holm"] = np.minimum(holm, 1)
bh = [res.loc[k, "p_perm"] * m / (i + 1) for i, k in enumerate(orden)]
res.loc[orden, "p_bh"] = np.minimum(np.minimum.accumulate(bh[::-1])[::-1], 1)

res = res.sort_values(f"J_{H_PRINCIPAL}h", ascending=False)
res.to_csv("reglas_falla.csv", index=False)

pd.set_option("display.width", 250)
cols = ["var", "carta", "regla", f"sens_{H_PRINCIPAL}h_%", "FPR_sanos_%", f"J_{H_PRINCIPAL}h",
        "J_24h", "J_96h", "falsos_eventos_dia", "fallas_con_evento_semana", "anticip_h_mediana",
        "p_perm", "p_holm", "p_bh"]
print(f"Ranking por J de Youden a {H_PRINCIPAL} h (4 fallas vs 8 sanos, promedio por rodamiento)")
print(res[cols].head(20).round(3).to_string(index=False))
print("\nReglas que nunca superan J = 0.05 en ninguna carta/variable:",
      sorted(set(REGLAS) - set(res.loc[res[f"J_{H_PRINCIPAL}h"] > 0.05, "regla"])))
print(f"\nMejor carta por variable (máx. J_{H_PRINCIPAL}h entre reglas):")
print(res.groupby(["var", "carta"])[f"J_{H_PRINCIPAL}h"].max().unstack().round(3).to_string())

for v, carta, regla in res[["var", "carta", "regla"]].head(3).itertuples(index=False):
    t = tasas.loc[(v, carta, regla)]
    print(f"\nDetalle {v} · {carta} · {regla}: sens {H_PRINCIPAL} h (fallan) / FPR (sanos) / anticipación h")
    print((t[[f"sens_{H_PRINCIPAL}", "fpr", "anticip_h"]] * [100, 100, 1]).round(1).T.to_string())
