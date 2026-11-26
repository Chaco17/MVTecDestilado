"""Tables and figures of the training runs, shared by the notebooks of the models."""
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_COLOR, VAL_COLOR, GUIDE_COLOR = "#2a78d6", "#eb6834", "#898781"


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def runs_table(results, metrics=("loss", "acc")):
    """One row per run with its best epoch (lowest val/loss) and the test metrics.

    results maps a run name to the dict returned by src.training.run_experiment.
    """
    rows = []
    for name, r in results.items():
        row = {
            "corrida": name,
            "epocas": r["epochs"],
            "parada temprana": r["stopped_early"],
            "mejor epoca": r["best_epoch"],
        }
        for metric in metrics:
            for stage in ("train", "val"):
                row[f"{stage}/{metric}"] = r["best"].get(f"{stage}/{metric}")
            row[f"test/{metric}"] = r["test"].get(f"test/{metric}")
        row["minutos"] = r["minutes"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("corrida")


def plot_curves(results, metrics):
    """Train and validation curves per run (columns) and metric (rows).

    metrics is a list of (metric, label, log_scale), e.g. [("loss", "entropia cruzada", True)].
    The dotted vertical line marks the best epoch (the checkpoint used for test).
    """
    fig, axes = plt.subplots(len(metrics), len(results), figsize=(5 * len(results), 3.4 * len(metrics)),
                             squeeze=False)
    for j, (name, r) in enumerate(results.items()):
        hist = pd.DataFrame(r["history"])
        for i, (metric, label, log_scale) in enumerate(metrics):
            ax = axes[i, j]
            ax.plot(hist["epoch"], hist[f"train/{metric}"], color=TRAIN_COLOR, lw=2, label="entrenamiento")
            ax.plot(hist["epoch"], hist[f"val/{metric}"], color=VAL_COLOR, lw=2, ls="--", label="validacion")
            ax.axvline(r["best_epoch"], color=GUIDE_COLOR, lw=1, ls=":")
            if log_scale:
                ax.set_yscale("log")
            ax.set_title(f"{name}: {label}", fontsize=10)
            ax.set_xlabel("epoca")
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    return fig
