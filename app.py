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


st.set_page_config(page_title="Cronoamperometría Numérica", layout="wide")
st.title("Cronoamperometría: difusión lineal (semi-infinita) y electrodo esférico")

# -----------------------------
# Estado de la app
# -----------------------------
if "runs" not in st.session_state:
    st.session_state.runs = []
if "run_id" not in st.session_state:
    st.session_state.run_id = 1

# -----------------------------
# Sidebar: parámetros
# -----------------------------
st.sidebar.header("Geometría")
geometry = st.sidebar.selectbox(
    "Selecciona el modelo",
    ["Planar (semi-infinita)", "Esférica (electrodo de radio a)"],
)

st.sidebar.header("Parámetros fisicoquímicos")
D = st.sidebar.number_input("D [m²/s]", value=1e-9, format="%.2e")
c_bulk = st.sidebar.number_input("c* (bulk) [mol/m³]", value=1.0, min_value=0.0)
E0 = st.sidebar.number_input("E⁰' [V]", value=0.0)
E = st.sidebar.slider("Potencial aplicado E [V]", -1.0, 1.0, 0.1)
n_e = st.sidebar.number_input("n (electrones)", value=1, min_value=1, step=1)

st.sidebar.header("Discretización y dominio")
max_t = st.sidebar.slider("Duración [s]", 0.5, 30.0, 6.0, step=0.5)
delta_t = st.sidebar.number_input("Δt [s]", value=0.01, min_value=1e-6, format="%.3g")
domain_factor = st.sidebar.number_input("Factor de dominio (≈ L/√(D·tmax))", value=6.0, min_value=2.0)

if geometry.startswith("Planar"):
    delta_x = st.sidebar.number_input("Δx [m]", value=2e-6, min_value=1e-9, format="%.2e")
    use_manual_max = st.sidebar.checkbox("Especificar max_x manualmente", value=False)
    if use_manual_max:
        max_x = st.sidebar.number_input("max_x [m]", value=3e-4, min_value=delta_x * 5, format="%.2e")
    else:
        max_x = None
    a = None
    r_max = None
    delta_r = None
else:
    a = st.sidebar.number_input("Radio del electrodo a [m]", value=25e-6, min_value=1e-9, format="%.2e")
    delta_r = st.sidebar.number_input("Δr [m]", value=2e-6, min_value=1e-9, format="%.2e")
    use_manual_rmax = st.sidebar.checkbox("Especificar r_max manualmente", value=False)
    if use_manual_rmax:
        r_max = st.sidebar.number_input(
            "r_max [m]",
            value=float(a + 3e-4),
            min_value=float(a + delta_r * 5),
            format="%.2e",
        )
    else:
        r_max = None
    max_x = None
    delta_x = None

# Etiqueta por defecto del experimento
st.sidebar.header("Gestión de curvas")
def_label = f"{geometry.split()[0]} | D={_fmt_sci(D)} | E={E:.3g} V | Δt={delta_t:g} s"
if geometry.startswith("Planar"):
    def_label += f" | Δx={_fmt_sci(delta_x)}"
else:
    def_label += f" | a={_fmt_sci(a)} | Δr={_fmt_sci(delta_r)}"

label = st.sidebar.text_input("Etiqueta (para la leyenda)", value=def_label)

col_btn1, col_btn2 = st.sidebar.columns(2)
run_and_add = col_btn1.button("Simular + añadir")
clear_all = col_btn2.button("Limpiar todo")

if clear_all:
    st.session_state.runs = []
    st.session_state.run_id = 1

# -----------------------------
# Simulación
# -----------------------------
if run_and_add:
    with st.spinner("Resolviendo difusión (implícito)..."):
        if geometry.startswith("Planar"):
            times, j, coord, profiles = solve_diffusion_implicit_planar(
                D=D,
                delta_x=delta_x,
                delta_t=delta_t,
                max_t=max_t,
                max_x=max_x,
                domain_factor=domain_factor,
                c_bulk=c_bulk,
                E=E,
                E0=E0,
                n_e=int(n_e),
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
                domain_factor=domain_factor,
                c_bulk=c_bulk,
                E=E,
                E0=E0,
                n_e=int(n_e),
            )
            # Para comparar geometrías, usamos distancia a la superficie: (r-a)
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
                "n_e": int(n_e),
                "max_t": max_t,
                "delta_t": delta_t,
                "domain_factor": domain_factor,
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

# -----------------------------
# UI principal
# -----------------------------
if len(st.session_state.runs) == 0:
    st.info("Simula y añade una curva para empezar a comparar.")
    st.stop()

st.subheader("Curvas almacenadas")
summary_rows = []
for r in st.session_state.runs:
    p = r["params"]
    summary_rows.append(
        {"ID": r["id"], "Etiqueta": r["label"], "Geometría": r["geometry"], "D [m²/s]": p["D"], "E [V]": p["E"], "max_t [s]": p["max_t"]}
    )
st.dataframe(summary_rows, use_container_width=True, hide_index=True)

ids = [r["id"] for r in st.session_state.runs]
selected_ids = st.multiselect("Selecciona curvas a mostrar", options=ids, default=ids)
selected = [r for r in st.session_state.runs if r["id"] in selected_ids]
if len(selected) == 0:
    st.warning("No hay curvas seleccionadas.")
    st.stop()

Tab_j, Tab_c = st.tabs(["j(t)", "Perfiles de concentración"])

with Tab_j:
    fig, ax = plt.subplots()
    for r in selected:
        ax.plot(r["times"], r["j"], label=f"{r['id']}: {r['label']}")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("j [A/m²]")
    ax.set_title("Densidad de corriente vs tiempo")
    ax.grid(True)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)

with Tab_c:
    t_max_sel = float(max(r["times"][-1] for r in selected))
    t_profile = st.slider(
        "Tiempo para los perfiles [s]",
        0.0,
        t_max_sel,
        min(1.0, t_max_sel),
        step=max(t_max_sel / 200.0, 1e-6),
    )

    fig, ax = plt.subplots()
    for r in selected:
        times = r["times"]
        idx = int(np.argmin(np.abs(times - t_profile)))
        prof = r["profiles"][idx]
        ax.plot(r["coord_um"], prof, label=f"{r['id']}: {r['label']} (t={times[idx]:.3g}s)")

    ax.set_xlabel("Distancia a la superficie (µm)")
    ax.set_ylabel("c [mol/m³]")
    ax.set_title("Perfiles de concentración")
    ax.grid(True)
    ax.legend(fontsize=8)
    st.pyplot(fig, use_container_width=True)

st.caption(
    "Notas: (i) La condición en la superficie usa una relación tipo Nernst; "
    "(ii) el dominio semi-infinito se aproxima con un límite externo ~ factor·√(D·tmax). "
    "Para electrodo esférico, la j mostrada es local en la superficie; el total sería I=4πa²·j."
)
