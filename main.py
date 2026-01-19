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
    """Difusión lineal hacia un electrodo plano con:
       - c(0,t) fijada por Nernst (Dirichlet)
       - flujo nulo en x=max_x: dc/dx = 0 (Neumann)

    PDE:  ∂c/∂t = D ∂²c/∂x², x∈[0,max_x]
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
        raise ValueError("max_x debe ser al menos ~5*Δx (mejor bastante mayor)")

    n = int(np.ceil(max_x / delta_x)) + 1
    m = int(np.ceil(max_t / delta_t))

    lam = D * delta_t / (delta_x ** 2)

    c0 = c_surf_nernst(c_bulk, E, E0)

    # Estado inicial
    C = np.full(n, c_bulk)
    C[0] = c0

    # Desconocidas: C[1]..C[n-1] (tamaño n-1)
    Nunk = n - 1
    diag = np.empty(Nunk, dtype=float)
    lower = np.empty(Nunk - 1, dtype=float)
    upper = np.empty(Nunk - 1, dtype=float)

    # Filas interiores (corresponden a i=1..n-2)
    diag[:-1] = 1.0 + 2.0 * lam
    lower[:-1] = -lam
    upper[:] = -lam

    # Última fila: condición Neumann (flujo nulo) => C[n-1] - C[n-2] = 0
    diag[-1] = 1.0
    lower[-1] = -1.0  # coeficiente de C[n-2] en la última ecuación
    # upper no se usa en la última fila (no existe superdiagonal)

    times = np.empty(m)
    j = np.empty(m)
    profiles = np.empty((m, n))

    for k in range(m):
        rhs = np.empty(Nunk, dtype=float)

        # Ecuaciones interiores: -lam*C[i-1] + (1+2lam)C[i] - lam*C[i+1] = C_old[i]
        rhs[:-1] = C[1:-1].copy()

        # Incorporar C[0]=c0 conocida en la primera ecuación interior (i=1)
        rhs[0] += lam * c0

        # Última ecuación: C[n-1] - C[n-2] = 0
        rhs[-1] = 0.0

        C_unk = _thomas_tridiagonal(lower, diag, upper, rhs)
        C[0] = c0
        C[1:] = C_unk

        # Flujo molar hacia el electrodo: N = -D (dc/dx)|_{x=0}
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
    """Difusión esférica (electrodo de radio a) con:
       - c(a,t) fijada por Nernst (Dirichlet)
       - flujo nulo en r=r_max: dc/dr = 0 (Neumann)

    PDE: ∂c/∂t = D( ∂²c/∂r² + (2/r)∂c/∂r ), r∈[a,r_max]
    Transformación u=r c => ∂u/∂t = D ∂²u/∂r²

    Neumann en c: dc/dr=0 en r_max  =>  d(u/r)/dr=0  =>  u'(r_max)=u(r_max)/r_max
    Discretización (1º orden hacia atrás):
      (U_N - U_{N-1})/dr = U_N / r_max
      => -U_{N-1} + (1 - dr/r_max) U_N = 0
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
        raise ValueError("r_max debe ser al menos ~a+5*Δr (mejor bastante mayor)")

    n = int(np.ceil((r_max - a) / delta_r)) + 1
    m = int(np.ceil(max_t / delta_t))
    r = a + np.arange(n) * delta_r

    lam = D * delta_t / (delta_r ** 2)

    c0 = c_surf_nernst(c_bulk, E, E0)
    u0 = a * c0

    # Estado inicial: c=c_bulk uniforme => u=r*c_bulk
    U = r * c_bulk
    U[0] = u0

    # Desconocidas: U[1]..U[n-1] (tamaño n-1)
    Nunk = n - 1
    diag = np.empty(Nunk, dtype=float)
    lower = np.empty(Nunk - 1, dtype=float)
    upper = np.empty(Nunk - 1, dtype=float)

    # Filas interiores (i=1..n-2)
    diag[:-1] = 1.0 + 2.0 * lam
    lower[:-1] = -lam
    upper[:] = -lam

    # Última fila: condición Robin equivalente a flujo nulo en c
    # -U[n-2] + (1 - dr/r_max) U[n-1] = 0
    diag[-1] = 1.0 - (delta_r / r_max)
    lower[-1] = -1.0

    if abs(diag[-1]) < 1e-14:
        raise ValueError("La condición en r_max da una ecuación casi singular; reduce Δr o aumenta r_max.")

    times = np.empty(m)
    j = np.empty(m)
    profiles = np.empty((m, n))

    for k in range(m):
        rhs = np.empty(Nunk, dtype=float)

        rhs[:-1] = U[1:-1].copy()
        rhs[0] += lam * u0       # incorpora U[0]=u0 conocida
        rhs[-1] = 0.0            # ecuación de contorno

        U_unk = _thomas_tridiagonal(lower, diag, upper, rhs)
        U[0] = u0
        U[1:] = U_unk

        C = U / r

        # Flujo molar en r=a: N = -D (dc/dr)|_a
        dc_dr_a = (-3.0 * C[0] + 4.0 * C[1] - C[2]) / (2.0 * delta_r)
        N_mol = -D * dc_dr_a

        times[k] = (k + 1) * delta_t
        j[k] = F * N_mol  # n=1 fijo
        profiles[k] = C

    return times, j, r, profiles
