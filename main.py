import numpy as np

def solve_diffusion_implicit_1d(delta_x=2e-4, delta_t=0.02, max_t=6.0):
    max_x = 6.0 * np.sqrt(max_t)
    n = int(max_x / delta_x)
    m = int(max_t / delta_t)

    # Parámetros del sistema lineal
    lam = delta_t / (delta_x ** 2)
    a = -lam
    b = 1 + 2 * lam
    c = -lam

    # Tridiagonal constante
    A_diag = np.full(n, b)
    A_lower = np.full(n - 1, a)
    A_upper = np.full(n - 1, c)

    # Condiciones iniciales
    C = np.ones(n)
    C[0] = 0.0
    delta = C.copy()

    times = []
    fluxes = []
    profiles = []

    # Preparar coeficientes modificados para el método de Thomas
    c_star = np.zeros(n)
    d_star = np.zeros(n)

    for k in range(m):
        # Construir RHS (delta viene del paso anterior)
        d = delta.copy()
        d[0] = 0.0     # condición Dirichlet en el electrodo
        d[-1] = 1.0    # concentración fija en el bulk

        # FORWARD sweep
        c_star[0] = A_upper[0] / A_diag[0]
        d_star[0] = d[0] / A_diag[0]

        for i in range(1, n - 1):
            denom = A_diag[i] - A_lower[i - 1] * c_star[i - 1]
            c_star[i] = A_upper[i] / denom
            d_star[i] = (d[i] - A_lower[i - 1] * d_star[i - 1]) / denom

        # Último paso
        denom = A_diag[-1] - A_lower[-2] * c_star[-2]
        d_star[-1] = (d[-1] - A_lower[-2] * d_star[-2]) / denom

        # BACKWARD substitution
        C[-1] = d_star[-1]
        for i in range(n - 2, -1, -1):
            C[i] = d_star[i] - c_star[i] * C[i + 1]

        # Actualizar para el próximo paso
        delta = C.copy()

        # Guardar resultados
        time = (k + 1) * delta_t
        flux = -(-C[2] + 4*C[1] - 3*C[0]) / (2 * delta_x)  # derivada de orden 2 en el borde

        times.append(time)
        fluxes.append(flux)
        profiles.append(C.copy())

    x = np.arange(n) * delta_x
    return np.array(times), np.array(fluxes), x, np.array(profiles)
