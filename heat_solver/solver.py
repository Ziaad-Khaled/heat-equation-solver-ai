import numpy as np
from .config import HeatConfig

def explicit_fd(config: HeatConfig) -> np.ndarray:
    """
    Solve the 1D heat equation using the explicit finite difference method.

    Parameters
    ----------
    config : HeatConfig
        Configuration object containing the problem parameters:
        - L: Length of the spatial domain
        - Nx: Number of spatial grid points
        - T: Total simulation time
        - Nt: Number of time steps
        - alpha: Thermal diffusivity
        - u0: Initial temperature
        - bc_left: Dirichlet boundary condition at the left end
        - bc_right: Dirichlet boundary condition at the right end

    Raises
    ------
    ValueError
        If the stability condition (r = alpha * dt / dx^2 <= 0.5) is violated.

    Returns
    -------
    u : np.ndarray
        Array of shape (Nt+1, Nx+1) containing the temperature at each spatial
        point (axis 1) for each time step (axis 0), i.e., u[n, i] is the
        temperature at time step n and spatial index i.
    """
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
