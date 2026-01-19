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
        jabs = np.abs(r["j"])
        for ti, ji in zip(t, jabs):
            lines.append(f"{ti:.12g}\t{ji:.12g}")
    return "\n".join(lines) + "\n"


def _build_txt_profile(selected_runs: list[dict], t_profile: float, species: str) -> str:
    lines = []
    lines.append("# Export: perfiles c(distancia,t) al tiempo elegido")
    lines.append(f"# t_objetivo[s]={t_profile}")
    if species == "Oxidada (c_ox)":
        lines.append("# Columnas: dist[um]\tc_ox[mol/m^3]")
    elif species == "Reducida (c_red)":
        lines.append("# Columnas: dist[um]\tc_red[mol/m^3]  (c_total=c_bulk constante)")
    else:
        lines.append("# Columnas: dist[um]\tc_ox[mol/m^3]\tc_red[mol/m^3]  (c_total=c_bulk)")

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

        times = r["times"]
        idx = int(np.argmin(np.abs(times - t_profile)))
        t_used = float(times[idx])
        dist = r["coord_um"]
        c_ox = r["profiles"][idx]
        c_tot = float(p["c_bulk"])
        c_red = c_tot - c_ox

        lines.append(f"# t_usado[s]={t_used:.12g}")

        if species == "Oxidada (c_ox)":
            for x, co in zip(dist, c_ox):
                lines.append(f"{x:.12g}\t{co:.12g}")
        elif species == "Reducida (c_red)":
            for x, cr in zip(dist, c_red):
                lines.append(f"{x:.12g}\t{cr:.12g}")
        else:
            for x, co, cr in zip(dist, c_ox, c_red):
                lines.append(f"{x:.12g}\t{co:.12g}\t{cr:.12g}")

    return "\n".join(lines) + "\n"


st.set_page_config(page_title="Cronoamperometría Numérica", layout="wide")
st.title("Cronoamperometría: difusión planar semi-infinita y electrodo esférico (sin animaciones)")

if "runs" not in st.session_state:
    st.session_state.runs = []
if "run_id" not in st.session_state:
    st.session_state.run_id = 1

st.sidebar.header("Geometría")
geometry = st.sidebar.selectbox(
    "Selecciona el modelo",
    ["Planar (semi-infinita)", "Esférica (electrodo de radio a)"],
)

st.sidebar.header("Parámetros fisicoquímicos")
D = st.sidebar.number_input("D [m²/s]", value=1e-9, format="%.2e")
c_bulk = st.sidebar.number_input("c_total (constante) [mol/m³]", value=1.0, min_value=0.0)
E0 = st.sidebar.number_input("E⁰' [V]", value=0.0)

# E como texto
E_text = st.sidebar.text_input("Potencial aplicado E [V] (texto)", value="0.1")

st.sidebar.header("Discretización y dominio (definidos por el usuario)")
max_t = st.sidebar.slider("Duración [s]", 0.5, 30.0, 6.0, step=0.5)
delta_t = st.sidebar.number_input("Δt [s]", value=0.01, min_value=1e-6, format="%.3g")

# Validación de E
try:
    E = _parse_float(E_text)
    sim_enabled = True
except Exception:
    st.sidebar.error("E no es un número válido. Ej.: 0.1 o -0.25")
    E = np.nan
    sim_enabled = False

if geometry.startswith("Planar"):
    delta_x = st.sidebar.number_input("Δx [m]", value=2e-6, min_value=1e-9, format="%.2e")
    max_x_default = float(_default_L(D, max_t))
    max_x = st.sidebar.number_input(
        "max_x [m] (dominio)",
        value=max_x_default,
        min_value=float(5 * delta_x),
        format="%.2e",
        help="Por defecto: 6*sqrt(D*tmax)."
    )
    a = None
    delta_r = None
    r_max = None
else:
    a = st.sidebar.number_input("Radio del electrodo a [m]", value=25e-6, min_value=1e-9, format="%.2e")
    delta_r = st.sidebar.number_input("Δr [m]", value=2e-6, min_value=1e-9, format="%.2e")
    r_max_default = float(a + _default_L(D, max_t))
    r_max = st.sidebar.number_input(
        "r_max [m] (límite externo)",
        value=r_max_default,
        min_value=float(a + 5 * delta_r),
        format="%.2e",
        help="Por defecto: a + 6*sqrt(D*tmax)."
    )
    delta_x = None
    max_x = None

st.sidebar.header("Gestión de curvas")
def_label = f"{geometry.split()[0]} | D={_fmt_sci(D)} | E={E if sim_enabled else '??'} V | Δt={delta_t:g} s"
if geometry.startswith("Planar"):
    def_label += f" | Δx={_fmt_sci(delta_x)} | max_x={_fmt_sci(max_x)}"
else:
    def_label += f" | a={_fmt_sci(a)} | Δr={_fmt_sci(delta_r)} | r_max={_fmt_sci(r_max)}"

label = st.sidebar.text_input("Etiqueta (para la leyenda)", value=def_label)

col_btn1, col_btn2 = st.sidebar.columns(2)
run_and_add = col_btn1.button("Simular + añadir", disabled=not sim_enabled)
clear_all = col_btn2.button("Limpiar todo")

if clear_all:
    st.session_state.runs = []
    st.session_state.run_id = 1

if run_and_add and sim_enabled:
    with st.spinner("Resolviendo difusión (implícito)..."):
        if geometry.startswith("Planar"):
            times, j, coord, profiles = solve_diffusion_implicit_planar(
                D=D,
                delta_x=delta_x,
                delta_t=delta_t,
                max_t=max_t,
                max_x=max_x,
                c_bulk=c_bulk,
                E=E,
                E0=E0,
            )
            coord_um = coord * 1e6
        else:
            times, j, r, profiles = solve_diffusion_implicit_spherical(
                D=D,
                delta_r=delta_r,
                delta_t=delta_t,
                max_t=max_t,
                a=a,
                r_max=r_max,
                c_bulk=c_bulk,
                E=E,
                E0=E0,
            )
            coord_um = (r - a) * 1e6

    st.session_state.runs.append(
        {
            "id": st.session_state.run_id,
            "label": label,
            "geometry": geometry,
            "params": {
                "D": D,
                "c_bulk": c_bulk,
                "E0": E0,
                "E": E,
                "max_t": max_t,
                "delta_t": delta_t,
                "delta_x": delta_x,
                "max_x": max_x,
                "a": a,
                "delta_r": delta_r,
                "r_max": r_max,
            },
            "times": times,
            "j": j,
            "coord_um": coord_um,
            "profiles": profiles,
        }
    )
    st.session_state.run_id += 1

if len(st.session_state.runs) == 0:
    st.info("Simula y añade una curva para empezar a comparar.")
    st.stop()

st.subheader("Curvas almacenadas")
rows = []
for r in st.session_state.runs:
    p = r["params"]
    rows.append(
        {
            "ID": r["id"],
            "Etiqueta": r["label"],
            "Geometría": r["geometry"],
            "D [m²/s]": p["D"],
            "E [V]": p["E"],
            "tmax [s]": p["max_t"],
        }
    )
st.dataframe(rows, use_container_width=True, hide_index=True)

ids = [r["id"] for r in st.session_state.runs]
selected_ids = st.multiselect("Selecciona curvas a mostrar", options=ids, default=ids)
selected = [r for r in st.session_state.runs if r["id"] in selected_ids]
if len(selected) == 0:
    st.warning("No hay curvas seleccionadas.")
    st.stop()

# --- Controles de perfiles ---
st.markdown("### Visualización")
species = st.radio(
    "Perfil a representar",
    ["Oxidada (c_ox)", "Reducida (c_red)", "Ambas (c_ox y c_red)"],
    horizontal=True,
)

t_max_sel = float(max(r["times"][-1] for r in selected))
t_profile = st.slider(
    "Tiempo para los perfiles [s]",
    0.0,
    t_max_sel,
    min(1.0, t_max_sel),
    step=max(t_max_sel / 200.0, 1e-6),
)

# --- Descargas ---
txt_j = _build_txt_j(selected)
txt_prof = _build_txt_profile(selected, t_profile, species)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        label="Descargar |j(t)| (seleccionadas) .txt",
        data=txt_j,
        file_name="j_abs_vs_t.txt",
        mime="text/plain",
        use_container_width=True,
    )
with dl2:
    st.download_button(
        label="Descargar perfiles a t (seleccionadas) .txt",
        data=txt_prof,
        file_name="profiles_at_t.txt",
        mime="text/plain",
        use_container_width=True,
    )

# --- (1) Lado a lado: |j(t)| y perfiles ---
col_left, col_right = st.columns(2)

with col_left:
    fig, ax = plt.subplots()
    for r in selected:
        ax.plot(r["times"], np.abs(r["j"]), label=f"{r['id']}: {r['label']}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("|j| [A/m²]")  # (2) valor absoluto
    ax.set_title("Densidad de corriente (valor absoluto) vs tiempo")
    ax.grid(True)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)

with col_right:
    fig, ax = plt.subplots()
    for r in selected:
        times = r["times"]
        idx = int(np.argmin(np.abs(times - t_profile)))
        dist = r["coord_um"]
        c_ox = r["profiles"][idx]
        c_tot = float(r["params"]["c_bulk"])
        c_red = c_tot - c_ox

        if species == "Oxidada (c_ox)":
            ax.plot(dist, c_ox, label=f"{r['id']}: {r['label']} (t={times[idx]:.3g}s)")
        elif species == "Reducida (c_red)":
            ax.plot(dist, c_red, label=f"{r['id']}: {r['label']} (t={times[idx]:.3g}s)")
        else:
            ax.plot(dist, c_ox, label=f"{r['id']}: {r['label']} c_ox (t={times[idx]:.3g}s)")
            ax.plot(dist, c_red, linestyle="--", label=f"{r['id']}: {r['label']} c_red (t={times[idx]:.3g}s)")

    ax.set_xlabel("Distancia a la superficie (µm)")
    ax.set_ylabel("c [mol/m³]")
    ax.set_title("Perfiles de concentración")
    ax.grid(True)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)

st.caption(
    "Notas: (i) La condición en la superficie es tipo Nernst. "
    "(ii) Se asumen coeficientes de difusión iguales para las especies oxidada y reducida."
)

