"""Características por archivo (1 s de vibración, 20 kHz, 8 canales) del IMS 1st_test.

Además del RMS que usa el código del profesor, calcula indicadores que en rodamientos
suelen cambiar ANTES que el RMS: curtosis, factor de cresta, energía por bandas de
frecuencia y amplitud del espectro de envolvente en las frecuencias de defecto.

Uso: python scripts/ims_features.py <carpeta 1st_test> ims_features.csv
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal, stats

FS = 20_000
# Frecuencias de defecto del IMS a 2000 rpm (Qiu et al., 2006)
DEFECTOS = {"BPFO": 236.4, "BPFI": 296.9, "BSF": 139.9, "FTF": 14.8}
# La frecuencia real se corre de la teórica (deslizamiento, geometría): en el ensayo 2 el BPFO
# observado es ~231 Hz. Por eso el pico se busca en ±5 % y no en ±3 Hz.
TOL = 0.05
BANDAS = [(0, 1000), (1000, 2500), (2500, 5000), (5000, 10000)]
SOS_ENV = signal.butter(4, [2000, 8000], btype="bandpass", fs=FS, output="sos")


def feats_canal(x):
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2))
    pico = np.abs(x).max()
    d = {"rms": rms, "pico": pico, "cresta": pico / rms,
         "kurt": stats.kurtosis(x), "skew": stats.skew(x)}

    espectro = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / FS)
    total = espectro.sum()
    for lo, hi in BANDAS:
        d[f"banda_{lo//1000}_{hi//1000}k"] = espectro[(f >= lo) & (f < hi)].sum() / total

    # Espectro de envolvente: filtrado pasa banda + Hilbert, típico para defectos de rodamiento
    env = np.abs(signal.hilbert(signal.sosfiltfilt(SOS_ENV, x)))
    env_esp = np.abs(np.fft.rfft(env - env.mean()))
    fondo = np.median(env_esp[(f > 5) & (f < 1000)])
    for nombre, fd in DEFECTOS.items():
        d[f"env_{nombre}"] = env_esp[(f > fd * (1 - TOL)) & (f < fd * (1 + TOL))].max() / fondo
    return d


def main(carpeta, salida):
    archivos = sorted(a for a in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, a)))
    filas = []
    for i, a in enumerate(archivos):
        datos = pd.read_csv(os.path.join(carpeta, a), sep="\t", header=None).to_numpy()
        fila = {"Timestamp": datetime.strptime(a, "%Y.%m.%d.%H.%M.%S")}
        # 1st_test: 8 canales (2 por rodamiento); 2nd y 3rd: 4 canales (1 por rodamiento)
        por_rod = datos.shape[1] // 4
        for c in range(datos.shape[1]):
            b, ch = c // por_rod + 1, c % por_rod + 1
            for k, v in feats_canal(datos[:, c]).items():
                fila[f"B{b}_Ch{ch}_{k}"] = v
        filas.append(fila)
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(archivos)}", flush=True)
    pd.DataFrame(filas).to_csv(salida, index=False)
    print("guardado", salida, len(filas))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
