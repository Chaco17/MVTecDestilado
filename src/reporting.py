"""Tables of the training runs and charts, shared by the notebooks of the models.

The charts are plotly figures. Each one is shown in the notebook (it is stored inside the .ipynb, so it
can be seen without a W&B account) and the same figure is logged to a W&B "analysis" run
(job_type "analysis", group "analisis"). Every analysis run has a fixed id and is opened with
resume="allow", so executing the notebook again updates the same run instead of creating a duplicate.
"""
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots

from src.utils import resolve_path

# Categorical colors in a fixed order (validated palette: the first three are distinguishable with
# color vision deficiencies). Train is always slot 1 and validation slot 2.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TRAIN_COLOR, VAL_COLOR, GUIDE_COLOR = COLORS[0], COLORS[1], "#898781"
LAYOUT = dict(template="plotly_white", font=dict(size=12), hovermode="x unified",
              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


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


def train_val_figure(results, metrics, title):
    """Grid with one row per metric and one column per run, each panel with the train and validation
    curves. metrics is a list of (metric, label, log_y). The dotted line marks the best epoch."""
    names = list(results)
    fig = make_subplots(rows=len(metrics), cols=len(names), shared_xaxes=True,
                        subplot_titles=[f"{name}<br>{label}" for _, label, _ in metrics for name in names],
                        vertical_spacing=0.12, horizontal_spacing=0.06)
    for j, name in enumerate(names, start=1):
        history = results[name]["history"]
        epochs = [h["epoch"] for h in history]
        for i, (metric, label, log_y) in enumerate(metrics, start=1):
            for stage, color, dash, stage_name in (("train", TRAIN_COLOR, "solid", "entrenamiento"),
                                                   ("val", VAL_COLOR, "dash", "validacion")):
                fig.add_trace(go.Scatter(
                    x=epochs, y=[h.get(f"{stage}/{metric}") for h in history], mode="lines",
                    name=stage_name, legendgroup=stage, showlegend=(i == 1 and j == 1),
                    line=dict(color=color, width=2, dash=dash),
                    hovertemplate=f"{stage_name}: %{{y:.4g}}<extra></extra>"), row=i, col=j)
            fig.add_vline(x=results[name]["best_epoch"], line=dict(color=GUIDE_COLOR, width=1, dash="dot"), row=i, col=j)
            if log_y:
                fig.update_yaxes(type="log", row=i, col=j)
            fig.update_xaxes(title_text="epoca" if i == len(metrics) else None, row=i, col=j)
    fig.update_annotations(font_size=11)
    fig.update_layout(**LAYOUT, title=title, height=300 * len(metrics) + 120)
    return fig


def runs_figure(results, key, label, log_y=False, title=None):
    """One line per run with the same metric (e.g. val/loss) by epoch, colors in fixed order."""
    fig = go.Figure()
    for color, (name, r) in zip(COLORS, results.items()):
        fig.add_trace(go.Scatter(
            x=[h["epoch"] for h in r["history"]], y=[h.get(key) for h in r["history"]], mode="lines",
            name=name, line=dict(color=color, width=2), hovertemplate=f"{name}: %{{y:.4g}}<extra></extra>"))
    fig.update_layout(**LAYOUT, title=title or label, xaxis_title="epoca", yaxis_title=label, height=420)
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def bar_figure(df, label, value, title, value_format=".3g"):
    """Horizontal bars, one per row of the DataFrame, with the value written at the end of each bar."""
    fig = go.Figure(go.Bar(
        x=df[value], y=df[label], orientation="h", marker=dict(color=COLORS[0]),
        text=[format(v, value_format) for v in df[value]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:" + value_format + "}<extra></extra>"))
    fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, title=title, xaxis_title=value,
                      yaxis=dict(autorange="reversed"), height=110 + 45 * len(df), margin=dict(r=80))
    return fig


def log_figures(cfg, run_id, name, notes, figures):
    """Log plotly figures to the W&B analysis run of a notebook section and print its link.

    figures maps a W&B key (e.g. "curvas/modelo_a") to a plotly figure.
    """
    run = wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        id=run_id,
        name=name,
        group="analisis",
        job_type="analysis",
        tags=["analisis"],
        notes=notes,
        resume="allow",
        dir=str(resolve_path(cfg.logger.save_dir)),
    )
    run.log({key: wandb.Plotly(fig) for key, fig in figures.items()})
    url = run.url
    run.finish()
    print("Graficos registrados en W&B:", url)
