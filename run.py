#!/usr/bin/env python3
import argparse
from heat_solver.config import load_config
from heat_solver.solver import explicit_fd
from heat_solver.io import write_csv
import sys
import yaml
def main():
    parser = argparse.ArgumentParser(description="1D heat equation explicit FD solver")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        u = explicit_fd(cfg)
        write_csv(u, cfg)
        print(f"Simulation finished. Results written to {cfg.output}")
    except FileNotFoundError as e:
        print(f"Error: Config file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax in config file: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing or invalid parameter in config file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
if __name__ == "__main__":
    main()
