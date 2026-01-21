import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from main import solve_diffusion_implicit_1d

st.set_page_config(page_title="Cronoamperometría Numérica (1D)", layout="wide")
st.title("Cronoamperometría")

# --- Parámetros del sistema ---
st.sidebar.header("Parámetros de simulación")

# (CAMBIO) Se eliminan entradas de usuario para Δx y Δt y se definen internamente
def _choose_grid(max_t: float, max_x: float) -> tuple[float, float, int, int]:
    """
    Selección interna de malla para mejorar fiabilidad:
      - Objetivo espacial: dx ~ 1e-5 m (10 µm), con límites en nº de nodos
      - Objetivo temporal: dt ~ 2e-3 s (2 ms), con límites en nº de pasos
    Devuelve: (delta_x, delta_t, Nx, Nt)
    """
    DX_TARGET = 1e-5   # m
    DT_TARGET = 2e-3   # s

    NX_MIN, NX_MAX = 80, 600
    NT_MIN, NT_MAX = 500, 8000

    Nx = int(np.clip(np.ceil(max_x / DX_TARGET), NX_MIN, NX_MAX))
    Nt = int(np.clip(np.ceil(max_t / DT_TARGET), NT_MIN, NT_MAX))

    delta_x = float(max_x / Nx)
    delta_t = float(max_t / Nt)
    return delta_x, delta_t, Nx, Nt


max_t = st.sidebar.slider("Duración del experimento [s]", 1.0, 20.0, 6.0, step=1.0)
max_x = st.sidebar.number_input("Dominio de difusión maxX [m]", value=0.003, format="%.1e")

# (CAMBIO) calcular grid interno una vez que max_t y max_x están definidos
delta_x, delta_t, Nx, Nt = _choose_grid(float(max_t), float(max_x))

# (CAMBIO) mostrar valores internos (solo informativo; no editable)
st.sidebar.caption(
    f"Grid interno (no editable): Δx = {delta_x:.3e} m (Nx={Nx}), "
    f"Δt = {delta_t:.3e} s (Nt={Nt})"
)

c_bulk = st.sidebar.number_input("c*Ox [mol/m³]", value=1.0)
E0 = st.sidebar.number_input("E⁰' [V]", value=0.0)
E = st.sidebar.slider("Potencial aplicado E [V]", -1.0, 1.0, 0.1)

n_frames = st.sidebar.slider("Frames de la animación", 1, 100, 10)

# --- Layout para gráficos lado a lado ---
col1, col2 = st.columns(2)
placeholder1 = col1.empty()
placeholder2 = col2.empty()

# --- Sesión de estado para revisar después ---
if "done_anim" not in st.session_state:
    st.session_state.done_anim = False
if "times" not in st.session_state:
    st.session_state.times = None
if "j_vals" not in st.session_state:
    st.session_state.j_vals = None
if "x_vals" not in st.session_state:
    st.session_state.x_vals = None
if "profiles" not in st.session_state:
    st.session_state.profiles = None

# --- Ejecutar simulación y animación ---
if st.button("▶ Reproducir animación"):
    with st.spinner("Resolviendo el sistema por método numérico..."):
        times, j_vals, x_vals, profiles = solve_diffusion_implicit_1d(
            delta_x=delta_x,
            delta_t=delta_t,
            max_t=max_t,
            max_x=max_x,
            c_bulk=c_bulk,
            E=E,
            E0=E0
        )

    st.session_state.done_anim = True
    st.session_state.times = times
    st.session_state.j_vals = j_vals
    st.session_state.x_vals = x_vals
    st.session_state.profiles = profiles

    idx_frames = np.linspace(0, len(times) - 1, n_frames, dtype=int)

    for i in idx_frames:
        t = times[i]
        c = profiles[i]

        # Perfil de concentración
        fig1, ax1 = plt.subplots()
        ax1.plot(x_vals * 1e6, c)
        ax1.set_xlabel("x (μm)")
        ax1.set_ylabel("c (mol/m³)")
        ax1.set_title(f"Perfil de concentración (t = {t:.2f} s)")
        ax1.grid()
        placeholder1.pyplot(fig1)

        # Densidad de corriente
        fig2, ax2 = plt.subplots()
        ax2.plot(times, j_vals, label="j(t)")
        ax2.axvline(t, color="red", linestyle="--", label=f"t = {t:.2f} s")
        ax2.set_xlabel("Tiempo (s)")
        ax2.set_ylabel("Densidad de corriente (A/m²)")
        ax2.set_title("Densidad de corriente vs tiempo")
        ax2.legend()
        ax2.grid()
        placeholder2.pyplot(fig2)

        time.sleep(0.05)

# --- Revisión manual post-animación ---
if st.session_state.done_anim:
    st.subheader("🔍 Revisión manual del perfil de concentración")
    idx = st.slider("Selecciona un tiempo simulado", 0, len(st.session_state.times) - 1,
                    len(st.session_state.times) // 2)
    t_sel = st.session_state.times[idx]
    c_sel = st.session_state.profiles[idx]

    fig1, ax1 = plt.subplots()
    ax1.plot(st.session_state.x_vals * 1e6, c_sel)
    ax1.set_xlabel("x (μm)")
    ax1.set_ylabel("c (mol/m³)")
    ax1.set_title(f"Perfil de concentración (t = {t_sel:.2f} s)")
    ax1.grid()
    placeholder1.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(st.session_state.times, st.session_state.j_vals, label="j(t)")
    ax2.axvline(t_sel, color="red", linestyle="--", label=f"t = {t_sel:.2f} s")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Densidad de corriente (A/m²)")
    ax2.set_title("Densidad de corriente vs tiempo")
    ax2.legend()
    ax2.grid()
    placeholder2.pyplot(fig2)
