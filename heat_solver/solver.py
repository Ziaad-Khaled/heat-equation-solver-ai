import numpy as np
from .config import HeatConfig

def explicit_fd(config: HeatConfig) -> np.ndarray:
    """Return temperature field u(x, t) of shape (Nt+1, Nx+1)."""

    dx = config.L / config.Nx
    dt = config.T / config.Nt
    r = config.alpha * dt / dx**2

    if r > 0.5:
        raise ValueError(f"Stability condition violated (r={r:.3f}>0.5). Reduce dt or increase dx.")

    # grid
    u = np.full((config.Nt + 1, config.Nx + 1), config.u0, dtype=float)

    # enforce Dirichlet boundaries for all times
    u[:, 0] = config.bc_left
    u[:, -1] = config.bc_right

    # time stepping
    for n in range(config.Nt):
        # interior points
        u[n + 1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])
    return u
