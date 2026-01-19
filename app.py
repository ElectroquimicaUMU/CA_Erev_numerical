import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from main import solve_diffusion_numeric, current_density_numeric

st.set_page_config(page_title="Simulación Numérica - Cronoamperometría Esférica", layout="wide")
st.title("Cronoamperometría en Electrodo Esférico (Resolución Numérica)")

# --- Parámetros de entrada ---
st.sidebar.header("Parámetros del sistema")
c_ox_star = st.sidebar.number_input("c*Ox [mol/m³]", value=1.0)
D = st.sidebar.number_input("D_Ox [m²/s]", value=1e-9, format="%.1e")
r0 = st.sidebar.number_input("Radio del electrodo r₀ [m]", value=1e-4, format="%.1e")
max_r = st.sidebar.number_input("Dominio de difusión (radio máx) [m]", value=5e-4, format="%.1e")
E0 = st.sidebar.number_input("E⁰' [V]", value=0.0)
E = st.sidebar.slider("Potencial aplicado E [V]", min_value=-1.0, max_value=1.0, value=0.1)
t_max = st.sidebar.slider("Duración del experimento [s]", min_value=1.0, max_value=20.0, value=10.0, step=1.0)
n_frames = st.sidebar.slider("Número de frames (perfil concentración)", min_value=10, max_value=200, value=100)

# --- Parámetros numéricos internos ---
dt = 0.01
dr = (max_r - r0) / 200

# --- Mallas temporales y espaciales ---
t_highres = np.arange(0.01, t_max + dt, dt)
t_frames = np.linspace(0.01, t_max, n_frames)

# --- Estado de sesión para guardar resultados ---
if "done_anim" not in st.session_state:
    st.session_state.done_anim = False
if "c_profiles" not in st.session_state:
    st.session_state.c_profiles = None
if "j_vals" not in st.session_state:
    st.session_state.j_vals = None
if "r_vals" not in st.session_state:
    st.session_state.r_vals = None
if "t_vals" not in st.session_state:
    st.session_state.t_vals = None

# --- Layout gráfico lado a lado ---
col1, col2 = st.columns(2)
placeholder1 = col1.empty()
placeholder2 = col2.empty()

# --- Ejecutar simulación y animación ---
if st.button("▶ Reproducir animación"):
    with st.spinner("Resolviendo ecuaciones de difusión..."):
        t_vals, r_vals, c_matrix = solve_diffusion_numeric(
            c_ox_star, D, r0, max_r, E, E0, t_max, dt, dr
        )
        j_vals = current_density_numeric(c_matrix, r_vals, D, t_vals)

    # Guardamos resultados
    st.session_state.done_anim = True
    st.session_state.t_vals = t_vals
    st.session_state.r_vals = r_vals
    st.session_state.j_vals = j_vals
    st.session_state.c_profiles = []

    # Índices de los frames a mostrar
    frame_indices = [np.abs(t_vals - tf).argmin() for tf in t_frames]

    for i in frame_indices:
        t = t_vals[i]
        c = c_matrix[i, :]
        st.session_state.c_profiles.append(c)

        # --- Perfil de concentración ---
        fig1, ax1 = plt.subplots()
        ax1.plot(r_vals * 1e6, c)
        ax1.set_xlabel("r (μm)")
        ax1.set_ylabel("c_Ox (mol/m³)")
        ax1.set_title(f"Perfil de concentración (t = {t:.2f} s)")
        ax1.grid()
        placeholder1.pyplot(fig1)

        # --- Densidad de corriente ---
        fig2, ax2 = plt.subplots()
        ax2.plot(t_vals, j_vals, label="j(t)")
        ax2.axvline(t, color="red", linestyle="--", label=f"t = {t:.2f} s")
        ax2.set_xlabel("Tiempo (s)")
        ax2.set_ylabel("Densidad de corriente (A/m²)")
        ax2.set_title("Densidad de corriente vs tiempo")
        ax2.legend()
        ax2.grid()
        placeholder2.pyplot(fig2)

        time.sleep(0.05)

# --- Exploración manual post-animación ---
if st.session_state.done_anim:
    st.subheader("🔍 Revisión manual del perfil de concentración")
    t_idx = st.slider("Selecciona un tiempo simulado", 0, len(st.session_state.c_profiles)-1, len(st.session_state.c_profiles)//2)
    t_sel = t_frames[t_idx]
    c_sel = st.session_state.c_profiles[t_idx]

    # --- Gráfico del perfil de concentración ---
    fig1, ax1 = plt.subplots()
    ax1.plot(st.session_state.r_vals * 1e6, c_sel)
    ax1.set_xlabel("r (μm)")
    ax1.set_ylabel("c_Ox (mol/m³)")
    ax1.set_title(f"Perfil de concentración (t = {t_sel:.2f} s)")
    ax1.grid()
    placeholder1.pyplot(fig1)

    # --- Gráfico de j(t) completo ---
    fig2, ax2 = plt.subplots()
    ax2.plot(st.session_state.t_vals, st.session_state.j_vals, label="j(t)")
    ax2.axvline(t_sel, color="red", linestyle="--", label=f"t = {t_sel:.2f} s")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Densidad de corriente (A/m²)")
    ax2.set_title("Densidad de corriente vs tiempo")
    ax2.legend()
    ax2.grid()
    placeholder2.pyplot(fig2)
