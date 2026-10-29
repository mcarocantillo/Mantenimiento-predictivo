"""Compara alarmas por sigma (EPS1 = potencia del motor, proxy del torque) contra ML
supervisado y no supervisado para detectar degradación temprana.

Uso: python -I scripts/analisis.py features.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(sys.argv[1], index_col="ciclo")
etiquetas = ["cooler", "valve", "pump", "acc", "stable"]
X_all = df.drop(columns=etiquetas)
X_eps = X_all.filter(like="EPS1_")

# Componentes: valor sano y orden de severidad (de leve a casi falla total)
COMP = {
    "valve": (100, [90, 80, 73]),
    "pump": (0, [1, 2]),
    "acc": (130, [115, 100, 90]),
    "cooler": (100, [20, 3]),
}

# Segmentos contiguos con las mismas condiciones -> grupos para validación cruzada.
# Ciclos vecinos son casi idénticos; un split aleatorio inflaría los resultados.
seg = (df[etiquetas[:4]] != df[etiquetas[:4]].shift()).any(axis=1).cumsum()
grupos = seg // 4  # bloques de ~4 segmentos para tener grupos con varias condiciones
estable = df["stable"] == 0

print("=" * 70)
print("1) ALARMAS POR SIGMA sobre la potencia media del motor (EPS1) por ciclo")
print("=" * 70)
x = df["EPS1_mean"]
for comp, (sano, grados) in COMP.items():
    # Referencia: ciclos donde ESTE componente está sano (los demás varían)
    ref = x[(df[comp] == sano) & estable]
    mu, sd = ref.mean(), ref.std()
    print(f"\n{comp}: referencia mu={mu:.1f} W, sigma={sd:.1f} W (n={len(ref)})")
    for g in [sano] + grados:
        v = x[(df[comp] == g) & estable]
        z = (v - mu).abs() / sd
        print(f"  {comp}={g:>4}: alarma 1.5σ {100*(z>1.5).mean():5.1f}%   "
              f"alarma 3σ {100*(z>3).mean():5.1f}%   (n={len(v)})")

print("\n" + "=" * 70)
print("2) SUPERVISADO: RandomForest, validación cruzada por bloques de tiempo")
print("   ¿Qué % de ciclos de cada grado se detecta como 'degradado'?")
print("=" * 70)
for nombre_x, X in [("solo EPS1 (motor)", X_eps), ("todos los sensores", X_all)]:
    print(f"\n--- {nombre_x} ({X.shape[1]} características) ---")
    for comp, (sano, grados) in COMP.items():
        y = (df[comp] != sano).astype(int)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    n_jobs=-1, random_state=0)
        pred = cross_val_predict(rf, X, y, groups=grupos, cv=GroupKFold(5))
        fila = [f"FA en sano {100*pred[(df[comp]==sano).values].mean():5.1f}%"]
        for g in grados:
            fila.append(f"{g}:{100*pred[(df[comp]==g).values].mean():5.1f}%")
        print(f"  {comp:6s} " + "  ".join(fila))

print("\n" + "=" * 70)
print("3) NO SUPERVISADO: IsolationForest entrenado solo con ciclos donde el")
print("   componente está sano. Umbral = percentil 99 del puntaje en entrenamiento.")
print("=" * 70)
for nombre_x, X in [("solo EPS1 (motor)", X_eps), ("todos los sensores", X_all)]:
    print(f"\n--- {nombre_x} ---")
    for comp, (sano, grados) in COMP.items():
        mask_sano = ((df[comp] == sano) & estable).values
        # Entrenamos con la primera mitad (en tiempo) de los ciclos sanos y evaluamos
        # la falsa alarma en la segunda mitad, para no medir sobre lo ya visto.
        idx_sano = np.where(mask_sano)[0]
        tr, te = idx_sano[: len(idx_sano) // 2], idx_sano[len(idx_sano) // 2:]
        sc = StandardScaler().fit(X.iloc[tr])
        iso = IsolationForest(n_estimators=300, random_state=0).fit(sc.transform(X.iloc[tr]))
        s = -iso.score_samples(sc.transform(X))
        umbral = np.percentile(s[tr], 99)
        fila = [f"FA en sano(test) {100*(s[te]>umbral).mean():5.1f}%"]
        for g in grados:
            m = ((df[comp] == g) & estable).values
            fila.append(f"{g}:{100*(s[m]>umbral).mean():5.1f}%")
        print(f"  {comp:6s} " + "  ".join(fila))
