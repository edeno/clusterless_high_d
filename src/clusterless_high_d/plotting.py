"""Plotting functions for clusterless decoding results."""

import warnings
from pathlib import Path
from typing import Optional

# Import matplotlib with headless backend support
import matplotlib
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    # If matplotlib isn't available, we'll handle it below
    plt = None


def _setup_matplotlib_backend(config: dict):
    """Setup matplotlib backend based on config."""
    if plt is None:
        raise ImportError("matplotlib is required for plotting")

    if config.get("general", {}).get("headless", False):
        matplotlib.use("Agg")


def plot_posterior(
    time: np.ndarray,
    posterior: np.ndarray,
    actual_position: np.ndarray,
    config: dict,
    output_path: Path,
) -> None:
    """Plot posterior over time.

    Parameters
    ----------
    time : np.ndarray
        Time stamps (n_time,)
    posterior : np.ndarray
        Posterior probability (n_time, n_x_bins, n_y_bins)
    actual_position : np.ndarray
        Actual position (n_time, 2)
    config : dict
        Plotting configuration
    output_path : Path
        Where to save the plot
    """
    _setup_matplotlib_backend(config)

    # Extract config
    posterior_config = config.get("posterior", {})
    general_config = config.get("general", {})
    figsize_config = config.get("figsize", {})
    fonts_config = config.get("fonts", {})

    fig, axes = plt.subplots(
        2, 1, figsize=figsize_config.get("posterior", [12, 8]), sharex=True
    )

    # Plot 1: Posterior heatmap (marginal over y, showing x vs time)
    posterior_x = posterior.sum(axis=2)  # Sum over y dimension

    im = axes[0].imshow(
        posterior_x.T,
        aspect="auto",
        origin="lower",
        extent=[time[0], time[-1], 0, posterior_x.shape[1]],
        cmap=posterior_config.get("cmap", "viridis"),
        vmin=posterior_config.get("vmin"),
        vmax=posterior_config.get("vmax"),
    )

    # Overlay actual position if requested
    if posterior_config.get("show_actual_position", True):
        # Normalize actual position to grid coordinates
        pos_x_norm = (
            actual_position[:, 0] / actual_position[:, 0].max() * posterior_x.shape[1]
        )
        axes[0].plot(
            time,
            pos_x_norm,
            color=posterior_config.get("actual_position_color", "red"),
            alpha=posterior_config.get("actual_position_alpha", 0.7),
            linewidth=2,
            label="Actual position (X)",
        )
        axes[0].legend()

    axes[0].set_ylabel("X Position (bins)", fontsize=fonts_config.get("label_size", 12))
    axes[0].set_title(
        "Posterior Probability (marginalized over Y)",
        fontsize=fonts_config.get("title_size", 14),
    )
    plt.colorbar(im, ax=axes[0], label="Probability")

    # Plot 2: Posterior heatmap (marginal over x, showing y vs time)
    posterior_y = posterior.sum(axis=1)  # Sum over x dimension

    im2 = axes[1].imshow(
        posterior_y.T,
        aspect="auto",
        origin="lower",
        extent=[time[0], time[-1], 0, posterior_y.shape[1]],
        cmap=posterior_config.get("cmap", "viridis"),
        vmin=posterior_config.get("vmin"),
        vmax=posterior_config.get("vmax"),
    )

    if posterior_config.get("show_actual_position", True):
        pos_y_norm = (
            actual_position[:, 1] / actual_position[:, 1].max() * posterior_y.shape[1]
        )
        axes[1].plot(
            time,
            pos_y_norm,
            color=posterior_config.get("actual_position_color", "red"),
            alpha=posterior_config.get("actual_position_alpha", 0.7),
            linewidth=2,
            label="Actual position (Y)",
        )
        axes[1].legend()

    axes[1].set_xlabel("Time (s)", fontsize=fonts_config.get("label_size", 12))
    axes[1].set_ylabel("Y Position (bins)", fontsize=fonts_config.get("label_size", 12))
    axes[1].set_title(
        "Posterior Probability (marginalized over X)",
        fontsize=fonts_config.get("title_size", 14),
    )
    plt.colorbar(im2, ax=axes[1], label="Probability")

    plt.tight_layout()
    plt.savefig(output_path, dpi=general_config.get("dpi", 300), bbox_inches="tight")
    plt.close()


def plot_summary(
    time: np.ndarray,
    posterior: np.ndarray,
    actual_position: np.ndarray,
    config: dict,
    output_path: Path,
) -> None:
    """Plot summary statistics and examples.

    Parameters
    ----------
    time : np.ndarray
        Time stamps (n_time,)
    posterior : np.ndarray
        Posterior probability (n_time, n_x_bins, n_y_bins)
    actual_position : np.ndarray
        Actual position (n_time, 2)
    config : dict
        Plotting configuration
    output_path : Path
        Where to save the plot
    """
    _setup_matplotlib_backend(config)

    summary_config = config.get("summary", {})
    general_config = config.get("general", {})
    figsize_config = config.get("figsize", {})
    fonts_config = config.get("fonts", {})

    # Compute decoded position (argmax of posterior)
    n_time = len(time)
    decoded_position = np.zeros((n_time, 2))

    for t in range(n_time):
        flat_idx = posterior[t].argmax()
        decoded_position[t, 0] = flat_idx // posterior.shape[2]
        decoded_position[t, 1] = flat_idx % posterior.shape[2]

    # Normalize to same scale as actual position
    decoded_position[:, 0] = (
        decoded_position[:, 0] / posterior.shape[1] * actual_position[:, 0].max()
    )
    decoded_position[:, 1] = (
        decoded_position[:, 1] / posterior.shape[2] * actual_position[:, 1].max()
    )

    # Create figure
    fig = plt.figure(figsize=figsize_config.get("summary", [16, 10]))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Trajectory comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        actual_position[:, 0],
        actual_position[:, 1],
        "r-",
        alpha=0.5,
        linewidth=2,
        label="Actual",
    )
    ax1.plot(
        decoded_position[:, 0],
        decoded_position[:, 1],
        "b-",
        alpha=0.5,
        linewidth=2,
        label="Decoded",
    )
    ax1.set_xlabel("X Position", fontsize=fonts_config.get("label_size", 12))
    ax1.set_ylabel("Y Position", fontsize=fonts_config.get("label_size", 12))
    ax1.set_title("Trajectory Comparison", fontsize=fonts_config.get("title_size", 14))
    ax1.legend()
    ax1.set_aspect("equal")

    # Plot 2: Position error over time
    if summary_config.get("show_error", True):
        ax2 = fig.add_subplot(gs[0, 1])
        position_error = np.sqrt(
            (actual_position[:, 0] - decoded_position[:, 0]) ** 2
            + (actual_position[:, 1] - decoded_position[:, 1]) ** 2
        )
        ax2.plot(time, position_error, "k-", linewidth=1)
        ax2.set_xlabel("Time (s)", fontsize=fonts_config.get("label_size", 12))
        ax2.set_ylabel("Position Error", fontsize=fonts_config.get("label_size", 12))
        ax2.set_title(
            "Decoding Error Over Time", fontsize=fonts_config.get("title_size", 14)
        )
        ax2.axhline(
            position_error.mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {position_error.mean():.2f}",
        )
        ax2.legend()

    # Plot 3: Error histogram
    if summary_config.get("show_error", True):
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(position_error, bins=50, color="gray", alpha=0.7, edgecolor="black")
        ax3.set_xlabel("Position Error", fontsize=fonts_config.get("label_size", 12))
        ax3.set_ylabel("Count", fontsize=fonts_config.get("label_size", 12))
        ax3.set_title("Error Distribution", fontsize=fonts_config.get("title_size", 14))
        ax3.axvline(
            position_error.mean(),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {position_error.mean():.2f}",
        )
        ax3.axvline(
            np.median(position_error),
            color="b",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(position_error):.2f}",
        )
        ax3.legend()

    # Plot 4-6: Example time bins showing 2D posterior
    n_examples = min(summary_config.get("n_example_bins", 3), 3)
    example_indices = np.linspace(0, n_time - 1, n_examples, dtype=int)

    for i, idx in enumerate(example_indices):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(
            posterior[idx].T,
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )

        # Mark actual and decoded positions
        actual_x = (
            actual_position[idx, 0] / actual_position[:, 0].max() * posterior.shape[1]
        )
        actual_y = (
            actual_position[idx, 1] / actual_position[:, 1].max() * posterior.shape[2]
        )
        ax.plot(actual_x, actual_y, "r*", markersize=15, label="Actual")

        decoded_x = (
            decoded_position[idx, 0] / actual_position[:, 0].max() * posterior.shape[1]
        )
        decoded_y = (
            decoded_position[idx, 1] / actual_position[:, 1].max() * posterior.shape[2]
        )
        ax.plot(decoded_x, decoded_y, "bx", markersize=15, label="Decoded")

        ax.set_title(
            f"t = {time[idx]:.2f}s", fontsize=fonts_config.get("title_size", 14)
        )
        ax.set_xlabel("X Position", fontsize=fonts_config.get("label_size", 12))
        ax.set_ylabel("Y Position", fontsize=fonts_config.get("label_size", 12))
        ax.legend()
        plt.colorbar(im, ax=ax, label="Probability")

    plt.savefig(output_path, dpi=general_config.get("dpi", 300), bbox_inches="tight")
    plt.close()
