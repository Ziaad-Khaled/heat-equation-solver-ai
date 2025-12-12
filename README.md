# 1D Heat Equation – Explicit Finite Differences

A tiny Python project that solves the one‑dimensional heat conduction equation  

\[
\frac{\partial u}{\partial t}= \alpha \frac{\partial^2 u}{\partial x^2}
\]

using the explicit forward‑time central‑space (FTCS) scheme.

## Features
* Parameters are read from a simple YAML file (`config.yaml`).
* stability check (`r = α dt/dx² ≤ 0.5`) is performed automatically.
* Solution is saved as a CSV file (`time, x₀, x₁, …`).
* Small test suite (`pytest`) verifies the stability condition and output dimensions.

## Quick start


```bash
# install dependencies
pip install -r requirements.txt

# run the simulation (uses default config.yaml)
python run.py

# run the tests
pytest -q
```

Edit config.yaml to change domain size, material parameters, grid resolution, etc.

## License

MIT – feel free to adapt for teaching or research.
