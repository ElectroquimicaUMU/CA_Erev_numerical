import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from main import solve_diffusion_implicit_planar, solve_diffusion_implicit_spherical


def _fmt_sci(x: float) -> str:
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10 ** exp)
    return f"{mant:.3g}e{exp}"


def _default_L(D: float, tmax: float) -> float:
    return 6.0 * np.sqrt(D * tmax)


def _parse_float(text: str) -> float:
    return float(text.strip().replace(",", "."))


def _build_txt_j(selected_runs: list[dict]) -> str:
    lines = []
    lines.append("# Export: |j(t)| por corrida (A/m^2)")
    lines.append("# Columnas: t[s]\t|j|[A/m^2]")
    for r in selected_runs:
        p = r["params"]
        lines.append("")
        lines.append(f"# --- RUN {r['id']} ---")
        lines.append(f"# label: {r['label']}")
        lines.append(f"# geometry: {r['geometry']}")
        lines.append(
            "# params: "
            + ", ".join([f"{k}={p[k]}" for k in p.keys()])
        )
        t = r["times"]
