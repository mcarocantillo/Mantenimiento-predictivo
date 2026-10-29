"""Huecos de registro (posibles reinicios del ensayo) en cada ensayo IMS.

Un hueco es un salto entre archivos consecutivos mayor que HUECO_MIN (el muestreo nominal
es cada 10 min). Para cada tramo entre huecos reporta duración y el salto de nivel del RMS
(mediana de los 6 archivos después vs los 6 antes, en % del nivel anterior), porque los
reinicios cambian el nivel y disparan alarmas falsas si se usa un solo baseline.

Uso: python scripts/ims_huecos.py ims_features.csv ims2_features.csv ims3_features.csv
"""
import sys

import pandas as pd

HUECO_MIN = pd.Timedelta("30min")
VECINOS = 6

for ruta in sys.argv[1:]:
    df = pd.read_csv(ruta, parse_dates=["Timestamp"])
    rms = {b: df.filter(regex=rf"^B{b}_Ch\d_rms$").mean(axis=1) for b in range(1, 5)}
    dt = df["Timestamp"].diff()
    cortes = df.index[dt > HUECO_MIN]

    print(f"===== {ruta}: {len(df)} archivos, {df['Timestamp'].iloc[0]} a {df['Timestamp'].iloc[-1]} "
          f"({(df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds() / 3600:.0f} h)")
    print(f"Muestreo típico: {dt.median()}; huecos > {HUECO_MIN}: {len(cortes)}")
    for i in cortes:
        antes, despues = slice(max(i - VECINOS, 0), i), slice(i, i + VECINOS)
        saltos = "  ".join(
            f"B{b} {100 * (rms[b].iloc[despues].median() / rms[b].iloc[antes].median() - 1):+5.1f}%"
            for b in range(1, 5))
        print(f"  {df['Timestamp'].iloc[i - 1]} -> {df['Timestamp'].iloc[i]}  "
              f"hueco {dt.iloc[i].total_seconds() / 3600:6.1f} h | salto RMS: {saltos}")

    bordes = [0, *cortes, len(df)]
    print("  Tramos:")
    for a, b in zip(bordes[:-1], bordes[1:]):
        t0, t1 = df["Timestamp"].iloc[a], df["Timestamp"].iloc[b - 1]
        print(f"    {t0} a {t1}: {b - a} archivos, {(t1 - t0).total_seconds() / 3600:.0f} h")
    print()
