import numpy as np
import csv
from .config import HeatConfig

def write_csv(u: np.ndarray, config: HeatConfig) -> None:
    """Write solution to CSV: first column = time, then temperatures at each x."""
    dx = config.L / config.Nx
    dt = config.T / config.Nt
    x = np.linspace(0, config.L, config.Nx + 1)

    with open(config.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["time"] + [f"x={xi:.5g}" for xi in x]
        writer.writerow(header)

        for n, row in enumerate(u):
            time = n * dt
            writer.writerow([time] + list(row))
