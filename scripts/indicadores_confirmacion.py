"""Indicadores por archivo para los conjuntos de confirmación (XJTU-SY y FEMTO/PRONOSTIA).

Sigue protocolo_confirmacion.md: para cada canal (horizontal y vertical) calcula el RMS y la curtosis
del archivo completo, y divide el archivo en 20 trozos iguales para las cartas X̄-S (X̄ = promedio
de los 20 RMS, S = su desviación estándar).

Uso:
  python scripts/indicadores_confirmacion.py femto <carpeta datos FEMTO> femto_indicadores.csv
  python scripts/indicadores_confirmacion.py xjtu  <carpeta datos XJTU-SY> xjtu_indicadores.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

N_SUB = 20
# Carpeta de cada rodamiento, intervalo nominal entre archivos (s) y columnas (horizontal, vertical)
FORMATOS = {
    "femto": {"intervalo": 10, "columnas": (4, 5), "patron": r"^acc_\d+\.csv$",
              "carpetas": ["Training_set/Learning_set", "Validation_Set/Full_Test_Set"]},
    "xjtu": {"intervalo": 60, "columnas": (0, 1), "patron": r"^\d+\.csv$", "carpetas": None},
}


def leer(ruta, columnas):
    with open(ruta, encoding="utf-8", errors="replace") as f:
        primera = f.readline()
    sep = ";" if primera.count(";") > primera.count(",") else ","
    encabezado = 0 if re.search(r"[A-Za-z]", primera) else None
    d = pd.read_csv(ruta, sep=sep, header=encabezado)
    return d.iloc[:, list(columnas)].to_numpy(dtype=float)


def indicadores(x):
    x = x - x.mean()
    trozos = np.array_split(x, N_SUB)
    rms_trozos = np.array([np.sqrt(np.mean(t ** 2)) for t in trozos])
    return {"rms": np.sqrt(np.mean(x ** 2)), "kurt": stats.kurtosis(x),
            "xbar": rms_trozos.mean(), "s": rms_trozos.std(ddof=1)}


def rodamientos(conjunto, raiz):
    fmt = FORMATOS[conjunto]
    if fmt["carpetas"]:
        bases = [os.path.join(raiz, c) for c in fmt["carpetas"]]
    else:
        bases = [os.path.join(dp, d) for dp, dn, _ in os.walk(raiz) for d in dn if re.match(r"^\d+(\.\d+)?[A-Za-z]", d)]
    for base in bases:
        for nombre in sorted(os.listdir(base)):
            carpeta = os.path.join(base, nombre)
            if os.path.isdir(carpeta) and re.match(r"^Bearing\d_\d+$", nombre):
                yield nombre, carpeta


def main(conjunto, raiz, salida):
    fmt = FORMATOS[conjunto]
    filas = []
    for nombre, carpeta in rodamientos(conjunto, raiz):
        archivos = [a for a in os.listdir(carpeta) if re.match(fmt["patron"], a)]
        archivos.sort(key=lambda a: int(re.sub(r"\D", "", a)))
        for i, a in enumerate(archivos):
            datos = leer(os.path.join(carpeta, a), fmt["columnas"])
            fila = {"conjunto": conjunto, "rodamiento": nombre, "archivo": i + 1, "t_seg": i * fmt["intervalo"]}
            for canal, col in (("h", 0), ("v", 1)):
                for k, v in indicadores(datos[:, col]).items():
                    fila[f"{k}_{canal}"] = v
            filas.append(fila)
        print(nombre, len(archivos), "archivos", flush=True)
    pd.DataFrame(filas).to_csv(salida, index=False)
    print("guardado", salida, len(filas))


if __name__ == "__main__":
    main(*sys.argv[1:4])
