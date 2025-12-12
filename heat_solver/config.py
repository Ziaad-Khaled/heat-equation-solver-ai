import yaml
from dataclasses import dataclass

@dataclass
class HeatConfig:
    L: float          # length of rod
    T: float          # total simulation time
    Nx: int           # spatial grid points
    Nt: int           # time steps
    alpha: float      # thermal diffusivity
    u0: float         # initial temperature (constant)
    bc_left: float    # Dirichlet left boundary
    bc_right: float   # Dirichlet right boundary
    output: str       # CSV file name

def load_config(path: str) -> HeatConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return HeatConfig(
        L=float(data["L"]),
        T=float(data["T"]),
        Nx=int(data["Nx"]),
        Nt=int(data["Nt"]),
        alpha=float(data["alpha"]),
        u0=float(data["u0"]),
        bc_left=float(data["bc_left"]),
        bc_right=float(data["bc_right"]),
        output=str(data["output"]),
    )
