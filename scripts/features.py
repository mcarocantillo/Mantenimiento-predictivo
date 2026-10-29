"""Extrae características por ciclo (60 s) de cada sensor del dataset hidráulico (UCI).

Uso: python -I scripts/features.py data features.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SENSORES = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2",
            "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]


def feats(x, nombre):
    """x: matriz (ciclos, muestras). Devuelve DataFrame de características por ciclo."""
    t = np.arange(x.shape[1])
    pend = np.polyfit(t, x.T, 1)[0]
    n = x.shape[1]
    q = n // 4  # el ciclo tiene fases de carga distintas: resumimos por cuartos
    d = {
        "mean": x.mean(1), "std": x.std(1), "min": x.min(1), "max": x.max(1),
        "p2p": np.ptp(x, 1), "slope": pend,
        "skew": stats.skew(x, 1), "kurt": stats.kurtosis(x, 1),
        "rms_diff": np.sqrt((np.diff(x, axis=1) ** 2).mean(1)),
    }
    for i in range(4):
        d[f"q{i+1}_mean"] = x[:, i * q:(i + 1) * q].mean(1)
    return pd.DataFrame({f"{nombre}_{k}": v for k, v in d.items()})


def main(carpeta, salida):
    carpeta = Path(carpeta)
    partes = []
    for s in SENSORES:
        x = pd.read_csv(carpeta / f"{s}.txt", sep="\t", header=None).to_numpy()
        partes.append(feats(x, s))
        print(s, x.shape, flush=True)
    perfil = pd.read_csv(carpeta / "profile.txt", sep="\t", header=None,
                         names=["cooler", "valve", "pump", "acc", "stable"])
    df = pd.concat(partes + [perfil], axis=1)
    df.index.name = "ciclo"
    df.to_csv(salida)
    print("guardado", salida, df.shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
