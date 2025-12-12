import numpy as np
import pytest
import tempfile
import os
from heat_solver.config import HeatConfig, load_config
from heat_solver.solver import explicit_fd

def make_cfg(ratio: float = 0.4) -> HeatConfig:
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
    cfg = make_cfg(ratio=0.45)               # r < 0.5 → should not raise
    u = explicit_fd(cfg)
    assert u.shape == (cfg.Nt + 1, cfg.Nx + 1)

def test_stability_condition_fail():
    cfg = make_cfg(ratio=0.6)                # r > 0.5 → ValueError
    with pytest.raises(ValueError):
        explicit_fd(cfg)

def test_boundary_values():
    cfg = make_cfg()
    u = explicit_fd(cfg)
    # boundaries stay constant for all times
    assert np.allclose(u[:, 0], cfg.bc_left)
    assert np.allclose(u[:, -1], cfg.bc_right)

# Tests for config loading functionality
def test_load_config_valid():
    """Test that load_config successfully parses a valid config file."""
    config_content = """
L: 1.0
T: 0.1
Nx: 50
Nt: 5000
alpha: 1e-4
u0: 20.0
bc_left: 100.0
bc_right: 0.0
output: results.csv
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    try:
        cfg = load_config(temp_path)
        assert cfg.L == 1.0
        assert cfg.T == 0.1
        assert cfg.Nx == 50
        assert cfg.Nt == 5000
        assert cfg.alpha == 1e-4
        assert cfg.u0 == 20.0
        assert cfg.bc_left == 100.0
        assert cfg.bc_right == 0.0
        assert cfg.output == "results.csv"
    finally:
        os.unlink(temp_path)

def test_load_config_missing_field():
    """Test that load_config raises KeyError when required field is missing."""
    config_content = """
L: 1.0
T: 0.1
Nx: 50
Nt: 5000
alpha: 1e-4
u0: 20.0
bc_left: 100.0
# bc_right is missing
output: results.csv
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    try:
        with pytest.raises(KeyError):
            load_config(temp_path)
    finally:
        os.unlink(temp_path)

def test_load_config_invalid_type():
    """Test that load_config handles invalid parameter types."""
    config_content = """
L: 1.0
T: 0.1
Nx: "not_an_integer"
Nt: 5000
alpha: 1e-4
u0: 20.0
bc_left: 100.0
bc_right: 0.0
output: results.csv
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError):
            load_config(temp_path)
    finally:
        os.unlink(temp_path)

def test_load_config_missing_file():
    """Test that load_config raises FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")

def test_load_config_invalid_yaml():
    """Test that load_config handles invalid YAML syntax."""
    config_content = """
L: 1.0
T: 0.1
Nx: [50
    missing closing bracket
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    try:
        # yaml.safe_load will raise a yaml.YAMLError for invalid syntax
        import yaml
        with pytest.raises(yaml.YAMLError):
            load_config(temp_path)
    finally:
        os.unlink(temp_path)
