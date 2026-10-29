"""¿Qué cambia ANTES de que el RMS cruce 1.5σ / 3σ? (IMS 1st_test)

Reproduce los límites del profesor (baseline 2003-11-08 a 2003-11-14, sigma = MR_bar/1.128)
y compara contra cuándo se salen otros indicadores, con el mismo método.

Uso: python scripts/ims_precursores.py ims_features.csv
"""
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], parse_dates=["Timestamp"])
INI, FIN = pd.Timestamp("2003-11-08"), pd.Timestamp("2003-11-14")
FIN_ENSAYO = df["Timestamp"].iloc[-1]
base = (df["Timestamp"] >= INI) & (df["Timestamp"] < FIN)
monitoreo = df["Timestamp"] >= FIN
SOSTENIDO = 3  # alarma = 3 archivos seguidos fuera del límite (evita picos aislados)

FEATS = sorted({c.split("_", 2)[2] for c in df.columns if c.startswith("B1_Ch1_")})
for b in range(1, 5):
    for f in FEATS:
        df[f"B{b}_{f}"] = (df[f"B{b}_Ch1_{f}"] + df[f"B{b}_Ch2_{f}"]) / 2


def limites_imr(x):
    """Media y sigma como en el código del profesor (carta I-MR)."""
    mr = x.diff().abs().iloc[1:].mean()
    return x.mean(), mr / 1.128


def primera_alarma(serie, mu, sigma, k, n=SOSTENIDO):
    """Primer instante (en monitoreo) con n puntos seguidos a más de k sigma."""
    fuera = ((serie - mu).abs() > k * sigma) & monitoreo
    seguidos = fuera.rolling(n).sum() == n
    if not seguidos.any():
        return None
    return df.loc[seguidos.idxmax(), "Timestamp"]


def horas_antes_fin(t):
    return None if t is None else (FIN_ENSAYO - t).total_seconds() / 3600


print(f"Fin del ensayo: {FIN_ENSAYO}   (B3: falla de pista interna, B4: falla de elemento rodante)")
print(f"Baseline: {INI.date()} a {FIN.date()} ({base.sum()} archivos). "
      f"Alarma = {SOSTENIDO} archivos seguidos fuera del límite.\n")

resumen = []
for b in range(1, 5):
    for f in FEATS:
        s = df[f"B{b}_{f}"]
        mu, sg = limites_imr(s[base])
        sd = s[base].std()
        t15 = primera_alarma(s, mu, sg, 1.5)
        t3 = primera_alarma(s, mu, sg, 3)
        # Misma regla con la sigma clásica (desviación estándar del baseline)
        t3_sd = primera_alarma(s, mu, sd, 3)
        resumen.append({"rod": f"B{b}", "indicador": f,
                        "h_antes_fin_1.5σ": horas_antes_fin(t15),
                        "h_antes_fin_3σ": horas_antes_fin(t3),
                        "h_antes_fin_3σ(std)": horas_antes_fin(t3_sd),
                        "% fuera 3σ en monitoreo": 100 * (((s - mu).abs() > 3 * sg) & monitoreo).sum()
                        / monitoreo.sum()})
res = pd.DataFrame(resumen)

pd.set_option("display.width", 200)
for b in ["B1", "B2", "B3", "B4"]:
    t = res[res["rod"] == b].drop(columns="rod").set_index("indicador")
    t = t.sort_values("h_antes_fin_3σ(std)", ascending=False)
    print(f"===== {b} =====  (horas ANTES del fin del ensayo en que salta cada alarma; vacío = nunca)")
    print(t.round(1).to_string(na_rep="-"))
    print()
