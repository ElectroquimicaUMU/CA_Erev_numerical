import numpy as np

def solve_diffusion_CN(
    delta_x=2e-4,
    delta_t=0.02,
    max_t=6.0
):
    max_x = 6.0 * np.sqrt(max_t)
    n = int(max_x / delta_x)
    m = int(max_t / delta_t)

    g_mod = np.zeros(n)
    delta = np.ones(n)
    delta[0] = 0.0

    d_mod = np.zeros(n)
    C = np.ones(n)

    lam = delta_t / (delta_x ** 2)
    alpha = -lam
    beta = 2.0 * lam + 1.0
    gamma = -lam

    g_mod[0] = 0.0
    for i in range(1, n - 1):
        g_mod[i] = gamma / (beta - g_mod[i - 1] * alpha)

    times = []
    fluxes = []
    profiles = []

    for k in range(m):
        d_mod[0] = 0.0
        for i in range(1, n - 1):
            d_mod[i] = (delta[i] - d_mod[i - 1] * alpha) / (
                beta - g_mod[i - 1] * alpha
            )

        C[n - 1] = 1.0
        for i in range(n - 2, -1, -1):
            C[i] = d_mod[i] - g_mod[i] * C[i + 1]
            delta[i] = C[i]

        time = (k + 1) * delta_t
        flux = -(-C[2] + 4*C[1] - 3*C[0]) / (2 * delta_x)

        times.append(time)
        fluxes.append(flux)
        profiles.append(C.copy())

    x = np.arange(n) * delta_x

    return np.array(times), np.array(fluxes), x, np.array(profiles)
