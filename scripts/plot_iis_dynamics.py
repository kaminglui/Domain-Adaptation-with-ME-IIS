import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_dynamics(npz_path: str, output: str = None) -> None:
    data = np.load(npz_path)
    steps = range(len(data["delta_norm"]))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, data["delta_norm"], label="||delta||")
    plt.plot(steps, data["kl"], label="KL(target||source)")
    plt.xlabel("IIS iteration")
    plt.legend()
    plt.title("ME-IIS Dynamics")
    plt.tight_layout()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot IIS weight dynamics.")
    parser.add_argument("--npz", type=str, required=True, help="Path to npz saved by adapt_me_iis.")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for the plot.")
    args = parser.parse_args()
    plot_dynamics(args.npz, args.output)


if __name__ == "__main__":
    main()
