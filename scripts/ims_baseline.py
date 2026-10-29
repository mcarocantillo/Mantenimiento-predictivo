"""Fase 2: baseline (estado estable) por ensayo con el CV del RMS, y límites I-MR solo con él.

Criterio (recalibrado para IMS, donde el CV estable es ~1 % y no 10-30 %):
  - CV móvil del RMS en ventanas de VENT archivos, sin cruzar huecos de registro.
  - Baseline = bloque de LARGO archivos seguidos, dentro de un mismo tramo y en la primera
    mitad del ensayo, cuyo peor CV (entre los 4 rodamientos) es el más bajo.
  - Media y sigma (MR̄/1.128 como el código del profesor, y desviación estándar) se calculan
    solo con ese bloque; todo lo posterior es monitoreo.
Para el ensayo 1 también prueba "recentrar" el nivel tras cada reinicio con la mediana de los
primeros RECENTRO archivos del tramo (lo que haría un operario al recalibrar tras una parada).

Chequeo de cordura: % de archivos de monitoreo fuera de 3σ en la primera mitad del monitoreo,
donde todavía no debería haber falla.

Uso: python scripts/ims_baseline.py   (lee ims*_features.csv y escribe baseline_ims.csv)
"""
import numpy as np
import pandas as pd

ENSAYOS = {1: ("ims_features.csv", ["B3", "B4"]),
           2: ("ims2_features.csv", ["B1"]),
           3: ("ims3_features.csv", ["B3"])}
VARS = ["rms", "kurt"]
VENT, LARGO, RECENTRO = 36, 144, 6  # 6 h, 24 h de registro, 1 h
HUECO_MIN = pd.Timedelta("30min")


def cargar(ruta):
    df = pd.read_csv(ruta, parse_dates=["Timestamp"])
    # Archivos con todos los canales casi en cero: la máquina ya estaba detenida (final de los ensayos 2 y 3)
    rms = df.filter(regex=r"_rms$")
    df = df[~(rms < 0.2 * rms.median()).all(axis=1)].reset_index(drop=True)
    out = pd.DataFrame({"Timestamp": df["Timestamp"]})
    out["tramo"] = (df["Timestamp"].diff() > HUECO_MIN).cumsum()
    for b in range(1, 5):
        for v in VARS:
            out[f"B{b}_{v}"] = df.filter(regex=rf"^B{b}_Ch\d_{v}$").mean(axis=1)
    return out


def cv_movil(df):
    """CV (%) móvil del RMS por rodamiento, calculado dentro de cada tramo."""
    cols = [f"B{b}_rms" for b in range(1, 5)]
    g = df.groupby("tramo")[cols]
    return 100 * g.transform(lambda s: s.rolling(VENT).std()) / g.transform(lambda s: s.rolling(VENT).mean())


def elegir_baseline(df, cv):
    """Índice final del bloque de LARGO archivos con el menor peor-CV."""
    peor = cv.max(axis=1)
    mismo_tramo = df["tramo"].eq(df["tramo"].shift(LARGO - 1))
    candidato = peor.rolling(LARGO - VENT + 1).max().where(mismo_tramo)
    candidato[len(df) // 2:] = np.nan
    fin = candidato.idxmin()
    return fin - LARGO + 1, fin


def limites(x):
    return x.mean(), x.diff().abs().iloc[1:].mean() / 1.128, x.std()


filas = []
for ens, (ruta, fallan) in ENSAYOS.items():
    df = cargar(ruta)
    cv = cv_movil(df)
    ini, fin = elegir_baseline(df, cv)
    base = df.index.isin(range(ini, fin + 1))
    mon = df.index > fin
    primera_mitad = mon & (df.index <= fin + (len(df) - fin) // 2)
    t0, t1 = df["Timestamp"].iloc[ini], df["Timestamp"].iloc[fin]
    print(f"===== Ensayo {ens} (fallan {', '.join(fallan)}): baseline {t0} a {t1} "
          f"(tramo {df['tramo'].iloc[ini]}, {LARGO} archivos)")
    print(f"  CV móvil del RMS (%):  {'':8}" + "".join(f"{f'B{b}':>8}" for b in range(1, 5)))
    for nombre, mask in [("baseline", base), ("monit. 1a mitad", primera_mitad),
                         ("últimas 24 h", df.index >= len(df) - 144)]:
        print(f"    {nombre:<24}" + "".join(f"{cv[f'B{b}_rms'][mask].median():8.2f}" for b in range(1, 5)))
    print(f"    {'máximo en todo el ensayo':<24}" + "".join(f"{cv[f'B{b}_rms'].max():8.2f}" for b in range(1, 5)))

    # Recentrado tras reinicio: desplaza cada tramo de monitoreo para que sus primeros
    # RECENTRO archivos queden en la media del baseline (solo usa datos ya vistos).
    for b in range(1, 5):
        for v in VARS:
            s = df[f"B{b}_{v}"]
            mu, s_mr, sd = limites(s[base])
            inicio_tramo = df.groupby("tramo").cumcount() < RECENTRO
            nivel = s.where(inicio_tramo).groupby(df["tramo"]).transform("median")
            tramo_base = df["tramo"].iloc[ini]
            ajuste = (nivel - mu).where(df["tramo"] != tramo_base, 0)
            fuera = lambda serie: 100 * ((serie - mu).abs() > 3 * s_mr)[primera_mitad].mean()
            filas.append({"ensayo": ens, "rod": f"B{b}", "var": v, "falla": f"B{b}" in fallan,
                          "base_ini": t0, "base_fin": t1, "media": mu, "sigma_mr": s_mr, "sd": sd,
                          "cv_base_%": 100 * sd / mu if v == "rms" else np.nan,
                          "sd/sigma_mr": sd / s_mr,
                          "%fuera3σ_1a_mitad": fuera(s),
                          "%fuera3σ_1a_mitad_recentrado": fuera(s - ajuste)})
    print()

res = pd.DataFrame(filas)
res.to_csv("baseline_ims.csv", index=False)
pd.set_option("display.width", 220)
cols = ["ensayo", "rod", "var", "falla", "media", "sigma_mr", "sd/sigma_mr",
        "%fuera3σ_1a_mitad", "%fuera3σ_1a_mitad_recentrado"]
print(res[cols].round(3).to_string(index=False))
