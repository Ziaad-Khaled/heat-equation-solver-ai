import numpy as np
import pytest
from heat_solver.config import HeatConfig
from heat_solver.solver import explicit_fd

def make_config(ratio: float = 0.4) -> HeatConfig:
    """Helper that returns a config with a controllable stability ratio r."""
    L, T, Nx, Nt, alpha = 1.0, 0.1, 20, 2000, 1e-4
    dx = L / Nx
    dt = ratio * dx**2 / alpha      # enforce r = ratio
    return HeatConfig(
        L=L,
        T=T,
        Nx=Nx,
        Nt=int(T / dt),
        alpha=alpha,
        u0=20.0,
        bc_left=100.0,
        bc_right=0.0,
        output="dummy.csv",
    )

def test_stability_condition_pass():
    cfg = make_config(ratio=0.45)               # r < 0.5 → should not raise
    u = explicit_fd(cfg)
    assert u.shape == (cfg.Nt + 1, cfg.Nx + 1)

def test_stability_condition_fail():
    cfg = make_config(ratio=0.6)                # r > 0.5 → ValueError
    with pytest.raises(ValueError):
        explicit_fd(cfg)

def test_boundary_values():
    cfg = make_config()
    u = explicit_fd(cfg)
    # boundaries stay constant for all times
    assert np.allclose(u[:, 0], cfg.bc_left)
    assert np.allclose(u[:, -1], cfg.bc_right)
