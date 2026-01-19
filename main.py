import numpy as np

# --- Constantes ---
F = 96485.0  # C/mol
R = 8.314    # J/mol/K
T = 298.0    # K


def eta(E: float, E0: float) -> float:
    """Sobretensión adimensional: (F/RT)(E - E0)."""
    return (F / (R * T)) * (E - E0)


def c_surf_nernst(c_bulk: float, E: float, E0: float) -> float:
    """Concentración superficial bajo condición de Nernst (forma logística)."""
    x = np.exp(eta(E, E0))
    return c_bulk * x / (1.0 + x)


def _thomas_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Resuelve un sistema tridiagonal Ax=rhs con el algoritmo de Thomas."""
    n = diag.size
    c_star = np.empty(n - 1, dtype=float)
    d_star = np.empty(n, dtype=float)

    c_star[0] = upper[0] / diag[0]
    d_star[0] = rhs[0] / diag[0]

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_star[i - 1]
        c_star[i] = upper[i] / denom
        d_star[i] = (rhs[i] - lower[i - 1] * d_star[i - 1]) / denom

    denom = diag[-1] - lower[-1] * c_star[-1]
    d_star[-1] = (rhs[-1] - lower[-1] * d_star[-2]) / denom

    x = np.empty(n, dtype=float)
    x[-1] = d_star[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]
    return x


def solve_diffusion_implicit_planar(
    *,
    D: float,
    delta_x: float,
    delta_t: float,
    max_t: float,
    max_x: float,
    c_bulk: float,
    E: float,
    E0: float,
):
    """Difusión lineal (semi-infinita aproximada) hacia un electrodo plano.

    PDE:  ∂c/∂t = D ∂²c/∂x²,  x>=0

    BCs (Dirichlet):
      c(0,t) = c_surf(E) (Nernst)
      c(max_x,t) = c_bulk

    Devuelve: times, j (A/m²), x, profiles
    """
    if D <= 0:
        raise ValueError("D debe ser > 0")
    if delta_x <= 0 or delta_t <= 0:
        raise ValueError("delta_x y delta_t deben ser > 0")
    if max_t <= 0:
        raise ValueError("max_t debe ser > 0")
    if max_x <= 0:
        raise ValueError("max_x debe ser > 0")
    if max_x < 5 * delta_x:
        raise ValueError("max_x debe ser al menos ~5*Δx")

    n = int(np.ceil(max_x / delta_x)) + 1
    m = int(np.ceil(max_t / delta_t))

    lam = D * delta_t / (delta_x ** 2)

    n_int = n - 2
    diag = np.full(n_int, 1.0 + 2.0 * lam)
    lower = np.full(n_int - 1, -lam)
    upper = np.full(n_int - 1, -lam)

    c0 = c_surf_nernst(c_bulk, E, E0)

    C = np.full(n, c_bulk)
    C[0] = c0

    times = np.empty(m)
    j = np.empty(m)
    profiles = np.empty((m, n))

    for k in range(m):
        rhs = C[1:-1].copy()
        rhs[0] += lam * c0
        rhs[-1] += lam * c_bulk

        C_int = _thomas_tridiagonal(lower, diag, upper, rhs)
        C[0] = c0
        C[1:-1] = C_int
        C[-1] = c_bulk

        dc_dx_0 = (-3.0 * C[0] + 4.0 * C[1] - C[2]) / (2.0 * delta_x)
        N_mol = -D * dc_dx_0

        times[k] = (k + 1) * delta_t
        j[k] = F * N_mol  # n=1 fijo
        profiles[k] = C

    x = np.arange(n) * delta_x
    return times, j, x, profiles


def solve_diffusion_implicit_spherical(
    *,
    D: float,
    delta_r: float,
    delta_t: float,
    max_t: float,
    a: float,
    r_max: float,
    c_bulk: float,
    E: float,
    E0: float,
):
    """Difusión esférica (electrodo esférico de radio a).

    PDE: ∂c/∂t = D( ∂²c/∂r² + (2/r)∂c/∂r ), r>=a
    Con u=r c => ∂u/∂t = D ∂²u/∂r²

    BCs:
      c(a,t)=c_surf(E)  => u(a,t)=a*c_surf
      c(r_max,t)=c_bulk => u(r_max,t)=r_max*c_bulk

    Devuelve: times, j (A/m²), r, profiles
    """
    if D <= 0:
        raise ValueError("D debe ser > 0")
    if delta_r <= 0 or delta_t <= 0:
        raise ValueError("delta_r y delta_t deben ser > 0")
    if max_t <= 0:
        raise ValueError("max_t debe ser > 0")
    if a <= 0:
        raise ValueError("a debe ser > 0")
    if r_max <= a:
        raise ValueError("r_max debe ser > a")
    if r_max < a + 5 * delta_r:
        raise ValueError("r_max debe ser al menos ~a+5*Δr")

    n = int(np.ceil((r_max - a) / delta_r)) + 1
    m = int(np.ceil(max_t / delta_t))
    r = a + np.arange(n) * delta_r

    lam = D * delta_t / (delta_r ** 2)

    n_int = n - 2
    diag = np.full(n_int, 1.0 + 2.0 * lam)
    lower = np.full(n_int - 1, -lam)
    upper = np.full(n_int - 1, -lam)

    c0 = c_surf_nernst(c_bulk, E, E0)
    u0 = a * c0
    uR = r[-1] * c_bulk

    U = r * c_bulk
    U[0] = u0
    U[-1] = uR

    times = np.empty(m)
    j = np.empty(m)
    profiles = np.empty((m, n))

    for k in range(m):
        rhs = U[1:-1].copy()
        rhs[0] += lam * u0
        rhs[-1] += lam * uR

        U_int = _thomas_tridiagonal(lower, diag, upper, rhs)
        U[0] = u0
        U[1:-1] = U_int
        U[-1] = uR

        C = U / r

        dc_dr_a = (-3.0 * C[0] + 4.0 * C[1] - C[2]) / (2.0 * delta_r)
        N_mol = -D * dc_dr_a

        times[k] = (k + 1) * delta_t
        j[k] = F * N_mol  # n=1 fijo
        profiles[k] = C

    return times, j, r, profiles
