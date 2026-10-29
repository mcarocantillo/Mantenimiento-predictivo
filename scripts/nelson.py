"""Las 8 reglas de Nelson tal como las implementa `reglas_nelson` del código del profesor
(Code01), vectorizadas con ventanas móviles. Ojo: su N8 exige además que los 8 puntos
alternen de lado, que es más estricto que la regla de Nelson estándar.
"""
import numpy as np
import pandas as pd


def _todos(cond, n):
    """True donde las últimas n posiciones cumplen cond."""
    return pd.Series(cond.astype(float)).rolling(n).sum().eq(n).to_numpy()


def _al_menos(cond, k, n):
    return pd.Series(cond.astype(float)).rolling(n).sum().ge(k).to_numpy()


def reglas_nelson(serie, media, sigma):
    x = np.asarray(serie, dtype=float)
    z = (x - media) / sigma
    d = np.diff(x, prepend=np.nan)
    signo_d = np.sign(d)
    signo_z = np.sign(z)
    return pd.DataFrame({
        "N1": np.abs(z) > 3,
        "N2": _todos(z > 0, 9) | _todos(z < 0, 9),
        "N3": _todos(d > 0, 5) | _todos(d < 0, 5),
        "N4": _todos(signo_d * np.roll(signo_d, 1) < 0, 12) & (np.arange(len(x)) >= 13),
        "N5": _al_menos(z > 2, 2, 3) | _al_menos(z < -2, 2, 3),
        "N6": _al_menos(z > 1, 4, 5) | _al_menos(z < -1, 4, 5),
        "N7": _todos(np.abs(z) < 1, 15),
        "N8": _todos(np.abs(z) > 1, 8) & _todos(signo_z * np.roll(signo_z, 1) < 0, 7)
              & (np.arange(len(x)) >= 7),
    })
