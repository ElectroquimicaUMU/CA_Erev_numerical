import numpy as np

F = 96485      # C/mol
R = 8.314      # J/mol·K
T = 298        # K

def eta(E, E0):
    return (F / (R * T)) * (E - E0)

def c_surf(c_star, E, E0):
    η = eta(E, E0)
    return c_star * np.exp(η) / (1 + np.exp(η))

def solve_diffusion_implicit_1d(
    delta_x=2e-4,
    delta_t=0.02,
    max_t=6.0,
    max_x=None,
    c_bulk=1.0,
    E=0.1,
    E0=0.0
):
    if max_x is None:
        max_x = 6.0 * np.sqrt(max_t)

    n = int(max_x / delta_x)
    m = int(max_t / delta_t)

    lam = delta_t / (delta_x ** 2)
    a = -lam
    b = 1 + 2 * lam
    c = -lam

    A_diag = np.full(n, b)
    A_lower = np.full(n - 1, a)
    A_upper = np.full(n - 1, c)

    c_electrode = c_surf(c_bulk, E, E0)

    C = np.ones(n) * c_bulk
    C[0] = c_electrode
    delta = C.copy()

    times = []
    fluxes = []
    profiles = []

    c_star = np.zeros(n)
    d_star = np.zeros(n)

    for k in range(m):
        d = delta.copy()
        d[0] = c_electrode
        d[-1] = c_bulk

        # Forward sweep
        c_star[0] = A_upper[0] / A_diag[0]
        d_star[0] = d[0] / A_diag[0]

        for i in range(1, n - 1):
            denom = A_diag[i] - A_lower[i - 1] * c_star[i - 1]
            c_star[i] = A_upper[i] / denom
            d_star[i] = (d[i] - A_lower[i - 1] * d_star[i - 1]) / denom

        denom = A_diag[-1] - A_lower[-2] * c_star[-2]
        d_star[-1] = (d[-1] - A_lower[-2] * d_star[-2]) / denom

        # Back substitution
        C[-1] = d_star[-1]
        for i in range(n - 2, -1, -1):
            C[i] = d_star[i] - c_star[i] * C[i + 1]

        delta = C.copy()

        t = (k + 1) * delta_t
        flux = -(-C[2] + 4 * C[1] - 3 * C[0]) / (2 * delta_x)

        times.append(t)
        fluxes.append(flux)
        profiles.append(C.copy())

    x = np.arange(n) * delta_x
    return np.array(times), np.array(fluxes), x, np.array(profiles)
