import numpy as np
import pytest
import csv
import os
import tempfile
from heat_solver.config import HeatConfig
from heat_solver.solver import explicit_fd
from heat_solver.io import write_csv

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

def test_write_csv_creates_file():
    """Test that write_csv creates a CSV file at the specified path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a simple config and solution
        cfg = HeatConfig(
            L=1.0, T=0.05, Nx=10, Nt=5,
            alpha=1e-4, u0=20.0, bc_left=100.0, bc_right=0.0,
            output=tmp_path
        )
        u = explicit_fd(cfg)
        
        # Write CSV
        write_csv(u, cfg)
        
        # Verify file exists
        assert os.path.exists(tmp_path)
        assert os.path.getsize(tmp_path) > 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_write_csv_header_format():
    """Test that CSV headers are formatted correctly."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        cfg = HeatConfig(
            L=1.0, T=0.05, Nx=10, Nt=5,
            alpha=1e-4, u0=20.0, bc_left=100.0, bc_right=0.0,
            output=tmp_path
        )
        u = explicit_fd(cfg)
        write_csv(u, cfg)
        
        # Read and check header
        with open(tmp_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # First column should be "time"
            assert header[0] == "time"
            
            # Remaining columns should be x positions
            assert len(header) == cfg.Nx + 2  # time + (Nx+1) spatial points
            
            # Check that x values are formatted correctly
            for col in header[1:]:
                assert col.startswith("x=")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_write_csv_output_dimensions():
    """Test that output dimensions match expectations."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        cfg = HeatConfig(
            L=1.0, T=0.05, Nx=10, Nt=5,
            alpha=1e-4, u0=20.0, bc_left=100.0, bc_right=0.0,
            output=tmp_path
        )
        u = explicit_fd(cfg)
        write_csv(u, cfg)
        
        # Read CSV and check dimensions
        with open(tmp_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Should have header + (Nt+1) time steps
            assert len(rows) == cfg.Nt + 2  # header + (Nt+1) rows
            
            # Each row should have time + (Nx+1) spatial points
            for row in rows:
                assert len(row) == cfg.Nx + 2  # time + (Nx+1) values
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_write_csv_content_correctness():
    """Test that CSV content matches the solution array."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        cfg = HeatConfig(
            L=1.0, T=0.05, Nx=10, Nt=5,
            alpha=1e-4, u0=20.0, bc_left=100.0, bc_right=0.0,
            output=tmp_path
        )
        u = explicit_fd(cfg)
        write_csv(u, cfg)
        
        dt = cfg.T / cfg.Nt
        
        # Read CSV and verify content
        with open(tmp_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            
            for n, row in enumerate(reader):
                # Check time value
                expected_time = n * dt
                assert np.isclose(float(row[0]), expected_time)
                
                # Check temperature values
                temp_values = [float(val) for val in row[1:]]
                assert np.allclose(temp_values, u[n])
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
