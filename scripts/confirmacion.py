"""Aplica protocolo_confirmacion.md a los indicadores de XJTU-SY o FEMTO.

Hipótesis principal: la alarma de dispersión de la carta X̄-S sobre el RMS (carta S) avisa en el
10 % final de la vida de cada rodamiento más seguido que por casualidad. Se compara con la carta I-MR
del código original (RMS, regla 1) usando las mismas medidas.

Uso: python scripts/confirmacion.py <indicadores.csv> [<indicadores2.csv> ...] [--canal h|v] [--final 0.10]
Escribe confirmacion_<canal>_<final>.csv con una fila por rodamiento y alarma.
"""
import argparse
import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest

MIN_ARCHIVOS, B3, B4, D2_MR, SEGUIDOS = 100, 0.510, 1.490, 1.128, 3


def linea_base(rms):
    """Bloque de L archivos en la primera mitad con el menor CV máximo (ventanas de L/4)."""
    n = len(rms)
    largo = max(30, math.ceil(0.10 * n))
    ventana = max(2, largo // 4)
    s = pd.Series(rms)
    cv = s.rolling(ventana).std() / s.rolling(ventana).mean()
    peor = cv.rolling(largo - ventana + 1).max()  # CV máximo de las ventanas dentro del bloque que termina en i
    candidatos = peor[(peor.index >= largo - 1) & (peor.index < n // 2)]
    fin = int(candidatos.idxmin())
    return fin - largo + 1, fin


def sostenida(alarma):
    """True en cada archivo que cierra una racha de SEGUIDOS archivos fuera de los límites."""
    return pd.Series(alarma.astype(float)).rolling(SEGUIDOS).sum().eq(SEGUIDOS).to_numpy()


def eventos(alarma):
    """Inicio de cada evento (primer archivo en que la racha se completa)."""
    s = sostenida(alarma)
    return s & ~np.r_[False, s[:-1]]


def evaluar(g, canal, final):
    rms, s = g[f"rms_{canal}"].to_numpy(), g[f"s_{canal}"].to_numpy()
    n = len(g)
    ini, fin = linea_base(rms)
    base = slice(ini, fin + 1)
    s_bar = s[base].mean()
    mu = rms[base].mean()
    sigma = np.abs(np.diff(rms[base])).mean() / D2_MR
    alarmas = {"Carta S (dispersión)": (s < B3 * s_bar) | (s > B4 * s_bar),
               "I-MR, regla 1 (código original)": np.abs(rms - mu) > 3 * sigma}
    t = g["t_seg"].to_numpy()
    vida = t[-1]
    idx = np.arange(n)
    previo = t >= (1 - final) * vida            # tramo previo a la falla
    sano = (idx > fin) & (t <= 0.5 * vida)      # tramo sano de referencia
    monitoreo = idx > fin
    intervalo = np.median(np.diff(t))
    filas = []
    for nombre, a in alarmas.items():
        ev = eventos(a) & monitoreo
        racha = sostenida(a) & monitoreo
        ultimo30 = ev & (t >= 0.7 * vida)
        primero = t[ultimo30][0] if ultimo30.any() else np.nan
        filas.append({
            "conjunto": g["conjunto"].iloc[0], "rodamiento": g["rodamiento"].iloc[0], "alarma": nombre,
            "archivos": n, "vida_min": vida / 60, "base_ini": ini, "base_fin": fin,
            # Acierto: hay al menos un evento (racha de 3 archivos fuera de límites) dentro del tramo previo
            "acierto": bool((racha & previo).any()),
            "archivos_previo": int(previo.sum()),
            "eventos_sano": int((ev & sano).sum()), "archivos_sano": int(sano.sum()),
            "en_racha_sano": int((racha & sano).sum()),
            "sensibilidad_%": 100 * a[previo].mean(),
            "fpr_%": 100 * a[sano].mean() if sano.any() else np.nan,
            "eventos_por_hora_sano": (ev & sano).sum() / (sano.sum() * intervalo / 3600) if sano.any() else np.nan,
            "anticipacion_min": (vida - primero) / 60,
            "anticipacion_%_vida": 100 * (vida - primero) / vida,
            "eventos_total": int(ev.sum()), "eventos_previo": int((ev & previo).sum()),
        })
    return filas


def prueba(tabla, estricta=False):
    """Prueba binomial de una cola: aciertos frente a la probabilidad de acertar por azar.

    p0 del protocolo: probabilidad de que empiece al menos un evento en la ventana, 1 − e^(−λ·n).
    p0 estricta: además suma la probabilidad de que la ventana empiece con una alarma ya en curso
    (fracción del tramo sano en racha), porque esas rachas también cuentan como acierto.
    """
    lam = tabla["eventos_sano"].sum() / tabla["archivos_sano"].sum()
    p_ini = 1 - np.exp(-lam * tabla["archivos_previo"])
    if estricta:
        en_curso = tabla["en_racha_sano"].sum() / tabla["archivos_sano"].sum()
        p_ini = np.minimum(1, p_ini + en_curso)
    p0 = float(np.mean(p_ini))
    k, n = int(tabla["acierto"].sum()), len(tabla)
    return k, n, p0, binomtest(k, n, p0, alternative="greater").pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archivos", nargs="+")
    ap.add_argument("--canal", default="h")
    ap.add_argument("--final", type=float, default=0.10)
    args = ap.parse_args()
    datos = pd.concat([pd.read_csv(a) for a in args.archivos], ignore_index=True)
    filas, excluidos = [], []
    for (c, r), g in datos.groupby(["conjunto", "rodamiento"], sort=False):
        if len(g) < MIN_ARCHIVOS:
            excluidos.append(f"{c} {r} ({len(g)} archivos)")
            continue
        filas += evaluar(g.reset_index(drop=True), args.canal, args.final)
    res = pd.DataFrame(filas)
    res.to_csv(f"confirmacion_{args.canal}_{int(100 * args.final)}.csv", index=False)
    print(f"Canal {args.canal}, tramo previo = {100 * args.final:.0f} % final de la vida")
    print("Excluidos (< 100 archivos):", excluidos or "ninguno")
    pd.set_option("display.width", 220)
    for alarma, t in res.groupby("alarma", sort=False):
        print(f"\n=== {alarma}")
        grupos = [(c, tt) for c, tt in t.groupby("conjunto")] + ([("conjunto", t)] if t.conjunto.nunique() > 1 else [])
        for nombre, tt in grupos:
            k, n, p0, p = prueba(tt)
            _, _, p0e, pe = prueba(tt, estricta=True)
            prec = 100 * tt.eventos_previo.sum() / max(tt.eventos_total.sum(), 1)
            print(f"  {nombre:<9} aciertos {k}/{n} | p0 {p0:.3f} -> p {p:.4f} | p0 estricta {p0e:.3f} -> p {pe:.4f} | "
                  f"sensibilidad {tt['sensibilidad_%'].mean():.0f} % | falsas alarmas {tt['fpr_%'].mean():.1f} % "
                  f"({tt.eventos_por_hora_sano.mean():.2f} eventos/h) | precisión {prec:.0f} % | "
                  f"anticipación mediana {tt['anticipacion_%_vida'].median():.1f} % de la vida")


if __name__ == "__main__":
    main()
