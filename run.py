#!/usr/bin/env python3
import argparse
from heat_solver.config import load_config
from heat_solver.solver import explicit_fd
from heat_solver.io import write_csv

def main():
    parser = argparse.ArgumentParser(description="1D heat equation explicit FD solver")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    u = explicit_fd(cfg)
    write_csv(u, cfg)
    print(f"Simulation finished. Results written to {cfg.output}")

if __name__ == "__main__":
    main()
