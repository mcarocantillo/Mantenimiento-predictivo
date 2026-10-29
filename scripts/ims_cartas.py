"""Fase 3: cartas I-MR, X̄-R y X̄-S con las 8 reglas de Nelson, para RMS y curtosis.

- I-MR: un punto por archivo (estadístico del segundo completo), σ = MR̄/1.128.
- X̄-R: subgrupo de n=5 subbloques espaciados del mismo segundo, σ_x̄ = R̄/(d2·√n).
- X̄-S: subgrupo de n=20 subbloques (el segundo completo), σ_x̄ = S̄/(c4·√n).
Las reglas se aplican a la carta de posición (I o X̄); la de dispersión (MR, R o S) aporta
la alarma "disp" cuando sale de sus límites 3σ. Límites solo con el baseline de la fase 2.
En el ensayo 1 cada tramo posterior a un reinicio se recentra con la mediana de su primera
hora (lo que haría el operario al recalibrar).

Uso: python scripts/ims_cartas.py   (escribe alarmas_ims.csv y limites_cartas.csv)
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from nelson import reglas_nelson  # noqa: E402

ENSAYOS = {1: ("ims_features.csv", "sub1.csv", ["B3", "B4"], True),
           2: ("ims2_features.csv", "sub2.csv", ["B1"], False),
           3: ("ims3_features.csv", "sub3.csv", ["B3"], False)}
VARS = ["rms", "kurt"]
HUECO_MIN, RECENTRO = pd.Timedelta("30min"), 6
# Constantes de cartas de control (Montgomery)
D2_5, D3_5, D4_5 = 2.326, 0.0, 2.114
C4_20, B3_20, B4_20 = 0.9869, 0.510, 1.490
D4_MR = 3.267
SUB_R = [f"s{k:02d}" for k in range(0, 20, 4)]   # 5 subbloques espaciados
SUB_S = [f"s{k:02d}" for k in range(20)]

base_tab = pd.read_csv("baseline_ims.csv", parse_dates=["base_ini", "base_fin"])
salidas, limites = [], []

for ens, (f_feat, f_sub, fallan, recentrar) in ENSAYOS.items():
    feat = pd.read_csv(f_feat, parse_dates=["Timestamp"])
    sub = pd.read_csv(f_sub, parse_dates=["Timestamp"])
    assert (feat["Timestamp"] == sub["Timestamp"]).all()
    # Archivos con todos los canales casi en cero: la máquina ya estaba detenida (final de los ensayos 2 y 3)
    rms = feat.filter(regex=r"_rms$")
    vivo = ~(rms < 0.2 * rms.median()).all(axis=1)
    feat, sub = feat[vivo].reset_index(drop=True), sub[vivo].reset_index(drop=True)
    t = feat["Timestamp"]
    tramo = (t.diff() > HUECO_MIN).cumsum()
    b_ini, b_fin = base_tab.loc[base_tab["ensayo"] == ens, ["base_ini", "base_fin"]].iloc[0]
    base = ((t >= b_ini) & (t <= b_fin)).to_numpy()
    fase = np.where(base, "base", np.where(t > b_fin, "monitoreo", "previo"))
    horas_fin = (t.iloc[-1] - t).dt.total_seconds() / 3600
    tramo_base = tramo[base].iloc[0]

    for b in range(1, 5):
        for v in VARS:
            sg_r = sub[[f"B{b}_{v}_{s}" for s in SUB_R]]
            sg_s = sub[[f"B{b}_{v}_{s}" for s in SUB_S]]
            cartas = {
                "I-MR": (feat.filter(regex=rf"^B{b}_Ch\d_{v}$").mean(axis=1), None),
                "Xbar-R": (sg_r.mean(axis=1), sg_r.max(axis=1) - sg_r.min(axis=1)),
                "Xbar-S": (sg_s.mean(axis=1), sg_s.std(axis=1)),
            }
            for carta, (pos, disp) in cartas.items():
                if recentrar:
                    nivel = pos.where(pos.groupby(tramo).cumcount() < RECENTRO).groupby(tramo).transform("median")
                    pos = pos - (nivel - pos[base].mean()).where(tramo > tramo_base, 0)
                mu = pos[base].mean()
                if carta == "I-MR":
                    disp = pos.diff().abs()
                    centro = disp[base][1:].mean()
                    sigma, lic, lsc = centro / 1.128, 0.0, D4_MR * centro
                elif carta == "Xbar-R":
                    centro = disp[base].mean()
                    sigma, lic, lsc = centro / (D2_5 * np.sqrt(5)), D3_5 * centro, D4_5 * centro
                else:
                    centro = disp[base].mean()
                    sigma, lic, lsc = centro / (C4_20 * np.sqrt(20)), B3_20 * centro, B4_20 * centro
                r = reglas_nelson(pos, mu, sigma)
                r["disp"] = ((disp > lsc) | (disp < lic)).to_numpy()
                r.insert(0, "disp_rel", (disp / centro).to_numpy())  # MR/MR̄, R/R̄ o S/S̄
                r.insert(0, "z", ((pos - mu) / sigma).to_numpy())
                r.insert(0, "horas_al_fin", horas_fin.to_numpy())
                r.insert(0, "fase", fase)
                r.insert(0, "Timestamp", t.to_numpy())
                for k, val in [("carta", carta), ("var", v), ("falla", f"B{b}" in fallan),
                               ("rod", f"B{b}"), ("ensayo", ens)]:
                    r.insert(0, k, val)
                salidas.append(r)
                sd_base = pos[base].std()
                limites.append({"ensayo": ens, "rod": f"B{b}", "var": v, "carta": carta,
                                "media": mu, "sigma_carta": sigma, "sd_entre_puntos": sd_base,
                                "sd/sigma": sd_base / sigma})

al = pd.concat(salidas, ignore_index=True)
al.to_csv("alarmas_ims.csv", index=False)
lim = pd.DataFrame(limites)
lim.to_csv("limites_cartas.csv", index=False)

pd.set_option("display.width", 220)
print("Límites: sd real de los puntos del baseline / σ que usa la carta (≈1 si los límites son realistas)")
print(lim.pivot_table(index=["ensayo", "rod"], columns=["var", "carta"], values="sd/sigma").round(1).to_string())

REGLAS = [f"N{i}" for i in range(1, 9)] + ["disp"]
mon = al[al["fase"] == "monitoreo"]
for v in VARS:
    print(f"\n=== {v}: % de puntos de monitoreo con cada regla activa ===")
    m = mon[mon["var"] == v]
    grupos = {"sanos": ~m["falla"],
              "fallan >48h": m["falla"] & (m["horas_al_fin"] > 48),
              "fallan ≤48h": m["falla"] & (m["horas_al_fin"] <= 48)}
    tab = pd.concat({g: 100 * m[mask].groupby("carta")[REGLAS].mean() for g, mask in grupos.items()},
                    names=["grupo"]).T
    tab = tab.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0, sort_remaining=False)
    print(tab.round(1).to_string())
