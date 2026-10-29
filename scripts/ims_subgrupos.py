"""Subgrupos racionales para las cartas X̄-R y X̄-S: cada archivo (1 s, 20 480 muestras)
se parte en N_SUB subbloques de 1024 muestras (~51 ms) y en cada uno se calcula RMS y curtosis.
En el 1st_test (2 canales por rodamiento) se promedian los dos canales, como hace el código
del profesor con el RMS.

Uso: python scripts/ims_subgrupos.py <carpeta del ensayo> salida.csv
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

N_SUB, LARGO = 20, 1024


def main(carpeta, salida):
    archivos = sorted(a for a in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, a)))
    filas = []
    for i, a in enumerate(archivos):
        datos = pd.read_csv(os.path.join(carpeta, a), sep="\t", header=None).to_numpy()
        por_rod = datos.shape[1] // 4
        bloques = datos[:N_SUB * LARGO].reshape(N_SUB, LARGO, -1)
        bloques = bloques - bloques.mean(axis=1, keepdims=True)
        rms = np.sqrt((bloques ** 2).mean(axis=1))          # (N_SUB, canales)
        kurt = stats.kurtosis(bloques, axis=1)
        fila = {"Timestamp": datetime.strptime(a, "%Y.%m.%d.%H.%M.%S")}
        for b in range(4):
            canales = slice(b * por_rod, (b + 1) * por_rod)
            for nombre, m in [("rms", rms), ("kurt", kurt)]:
                for k, v in enumerate(m[:, canales].mean(axis=1)):
                    fila[f"B{b + 1}_{nombre}_s{k:02d}"] = v
        filas.append(fila)
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(archivos)}", flush=True)
    pd.DataFrame(filas).to_csv(salida, index=False)
    print("guardado", salida, len(filas))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
