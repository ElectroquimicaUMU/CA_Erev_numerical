import numpy as np

F = 96485
R = 8.314
T = 298

def eta(E, E0):
    return (F / (R * T)) * (E - E0)

def c_ox_surface(c_ox_star, eta_val):
    return c_ox_star * np.exp(eta_val) / (1 + np.exp(eta_val))

def solve_diffusion_numeric(c_ox_star, D, r0, max_r, E, E0, t_max, dt, dr):
    eta_val = eta(E, E0)
    c_surf = c_ox_surface(c_ox_star, eta_val)

    r = np.arange(r0, max_r + dr, dr)
    t = np.arange(0, t_max + dt, dt)
    N = len(r)
    M = len(t)

    c = np.zeros((M, N))
    c[0, :] = c_ox_star

    for n in range(0, M - 1):
        for i in range(1, N - 1):
            term1 = D * dt / dr**2
            term2 = D * dt / (2 * dr * r[i])
            c[n+1, i] = c[n, i] + term1 * (c[n, i+1] - 2 * c[n, i] + c[n, i-1]) \
                        + term2 * (c[n, i+1] - c[n, i-1])
        # Condiciones de frontera
        c[n+1, 0] = c_surf
        c[n+1, -1] = c[n+1, -2]  # Neumann: sin flujo

    return t, r, c

def current_density_numeric(c, r, D, t):
    dc_dr = (c[:, 1] - c[:, 0]) / (r[1] - r[0])
    return -D * dc_dr
