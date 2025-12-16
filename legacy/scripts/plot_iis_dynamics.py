import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_dynamics(npz_path: str, output: str = None) -> None:
    data = np.load(npz_path)
    steps = range(len(data["delta_norm"]))
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(steps, data["delta_norm"], label="||delta||")
    axes[0].plot(steps, data["kl"], label="KL(target||source)")
    axes[0].set_ylabel("Update / KL")
    axes[0].set_title("ME-IIS Dynamics")
    axes[0].legend()

    if "moment_max" in data:
        axes[1].plot(steps, data["moment_max"], label="Max |moment error|", color="C2")
        axes[1].set_ylabel("Max |moment error|")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "moment_max not found in npz", ha="center", va="center")
        axes[1].set_ylabel("Max |moment error|")
    axes[1].set_xlabel("IIS iteration")

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
