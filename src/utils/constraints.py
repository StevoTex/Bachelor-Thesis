from __future__ import annotations
from typing import List, Dict
import numpy as np

def _name_index(space: List[Dict], name: str) -> int:
    names = [str(s.get("name", f"x{i}")).lower() for i, s in enumerate(space)]
    try:
        return names.index(name.lower())
    except ValueError:
        return -1

def enforce_gear_descending(x: np.ndarray, space: List[Dict]) -> np.ndarray:
    """
    Vereinheitlichte Constraint-/Reparam-Schicht.
    Gibt IMMER einen Vektor in der Reihenfolge [final_drive, roll_radius, gear3, gear4, gear5] zurück,
    unabhängig davon, wie die Suchraum-Parameter benannt/angeordnet sind.

    Fälle:
    - Δ-Reparametrisierung vorhanden: names enthalten 'gear5','delta54','delta43'
      -> gear4 = gear5 + delta54; gear3 = gear4 + delta43
    - Klassische Gänge: names enthalten 'gear3','gear4','gear5'
      -> sortiere die drei Werte strikt absteigend (erzwingt gear3 > gear4 > gear5)
    """
    x = np.asarray(x, dtype=float).copy()

    # Indizes der Basisgrößen (robust; Standard-Fallback auf Position 0/1)
    i_fd  = _name_index(space, "final_drive_ratio")
    i_rr  = _name_index(space, "roll_radius")
    if i_fd < 0: i_fd = 0
    if i_rr < 0: i_rr = 1

    fd = float(x[i_fd])
    rr = float(x[i_rr])

    # Δ-Parametrisierung?
    i_g5 = _name_index(space, "gear5")
    i_d54 = _name_index(space, "delta54")
    i_d43 = _name_index(space, "delta43")

    if i_g5 >= 0 and i_d54 >= 0 and i_d43 >= 0:
        g5 = float(x[i_g5])
        d54 = float(x[i_d54])
        d43 = float(x[i_d43])
        g4 = g5 + d54
        g3 = g4 + d43
        return np.asarray([fd, rr, g3, g4, g5], dtype=float)

    # Klassische Gang-Parameter?
    i_g3 = _name_index(space, "gear3")
    i_g4 = _name_index(space, "gear4")
    i_g5b = _name_index(space, "gear5")
    if i_g3 >= 0 and i_g4 >= 0 and i_g5b >= 0:
        gvals = [float(x[i_g3]), float(x[i_g4]), float(x[i_g5b])]
        gvals_sorted = sorted(gvals, reverse=True)  # erzwingt g3 > g4 > g5
        g3, g4, g5 = gvals_sorted
        return np.asarray([fd, rr, g3, g4, g5], dtype=float)

    # Fallback: nehme an, dass die letzten drei Komponenten Gänge sind
    gvals = [float(x[2]), float(x[3]), float(x[4])]
    g3, g4, g5 = sorted(gvals, reverse=True)
    return np.asarray([fd, rr, g3, g4, g5], dtype=float)
