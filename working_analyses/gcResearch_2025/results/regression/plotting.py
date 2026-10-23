#%%
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import lasso as ls
import ols

from typing import Literal, TYPE_CHECKING
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def get_simple_regression_line(
        x: np.ndarray,
        y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    return slope, intercept

def plot_model_permutations(
        X: pd.DataFrame,
        y: np.ndarray | pd.DataFrame,
        n_perm: int = 10,
        lasso_alpha: float = 0.25,
        use_ols: bool = True,
        x_label=None,
        y_label=None,
        title=None,
        subtitle=None,
        point_alpha=0.7,
        line_alpha=1,
        lim_buffer:int=10,
        seed: int = 0):

    np.random.seed(seed)
    fig, ax = plt.subplots(figsize=(8, 6))
    ylim = (0,0)
    maes = list()
    rmses = list()
    r2s = list()
    reg_lines = list()
    xlim = (y.min() - lim_buffer, y.max() + lim_buffer)
    x_line = np.linspace(xlim[0], xlim[1], 200)
    tot_features_in_common = 0
    prev_selected_idxs = np.full(shape=1, fill_value=-1)

    for i in range(n_perm):
        y_perm = np.random.permutation(y)

        if lasso_alpha > 0:
            model, X_scaled, scaler = ls.run_lasso(X, y_perm, alpha=lasso_alpha)
            selected_idxs = np.flatnonzero(model.coef_)
            in_common = np.intersect1d(selected_idxs, prev_selected_idxs, assume_unique=True).size
            tot_features_in_common += in_common
            prev_selected_idxs = selected_idxs

            if use_ols and len(selected_idxs) > 0:
                X_selected = X.iloc[:,selected_idxs]
                model, X_scaled, scaler = ols.run_ols(X_selected, y_perm)

        else:
            model, X_scaled, scaler = ols.run_ols(X, y_perm)

        y_pred = model.predict(X_scaled)

        if y_pred.min() < ylim[0]:
            ylim = (y_pred.min(), ylim[1])

        if y_pred.max() > ylim[1]:
            ylim = (ylim[0], y_pred.max())

        # Plot the predictions.
        plot_scatter(
            y_perm,
            y_pred,
            point_alpha=point_alpha,
            show_line=False,
            ax=ax)

        slope, intercept = get_simple_regression_line(y_perm, y_pred)
        reg_lines.append((slope, intercept))
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            alpha=line_alpha,
            color="tab:red",
            linewidth=2)
        
        mae = mean_absolute_error(y_perm, y_pred)
        rmse = root_mean_squared_error(y_perm, y_pred)
        r2 = r2_score(y_perm, y_pred)
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
  
    ylim = (ylim[0] - lim_buffer, ylim[1] + lim_buffer)
    reg_lines_arr = np.array(reg_lines)
    avg_slope, avg_intercept = reg_lines_arr.mean(axis=0)
    y_line = avg_slope * x_line + avg_intercept

    ax.plot(
        x_line,
        y_line,
        alpha=1,
        color="tab:red",
        linewidth=2,
        label="Regression fit")

    add_xy_line(ax, "Expected relationship")
    ax.set_xlabel(x_label or "True value")
    ax.set_ylabel(y_label or "Predicted value")
    fig.suptitle(title or "True vs. Predicted Permutation Test Comparison")

    if subtitle:
        ax.set_title(subtitle, fontsize=10)

    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.tight_layout()
    

    avg_features_in_common = tot_features_in_common / (n_perm - 1) if lasso_alpha > 0 \
        else -1
    return fig, ax, maes, rmses, r2s, avg_features_in_common
   

def plot_model_scatter(
        model,
        X_values,
        y_true,
        x_label=None,
        y_label=None,
        title=None,
        subtitle=None,
        show_line=True,
        point_alpha=0.7,
        line_alpha=1,
        fig=None,
        ax=None
        ):

    X = np.asarray(X_values)
    y_pred = model.predict(X)
    y_true = np.asarray(y_true)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    plot_scatter(
        y_true,
        y_pred,
        point_alpha=point_alpha,
        show_line=show_line,
        line_label="Expected relationship",
        ax=ax)

    if show_line:
        xlim = ax.get_xlim()
        slope, intercept = np.polyfit(y_true, y_pred, deg=1)
        x_line = np.linspace(xlim[0], xlim[1], 200)
        y_line = slope * x_line + intercept

        ax.plot(
            x_line,
            y_line,
            alpha=line_alpha,
            color="tab:red",
            linewidth=2,
            label="Regression fit")

    ax.set_xlabel(x_label or "True value")
    ax.set_ylabel(y_label or "Predicted value")
    fig.suptitle(title or "True vs. Predicted Scatter Comparison")

    if subtitle:
        ax.set_title(subtitle, fontsize=10)

    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_model_alpha_sweep(
        expr_results_metrics_df: pd.DataFrame,
        pyspoc_results_metrics_df: pd.DataFrame,
        include_plots: list[Literal["MAE", "RMSE", "R2"]] | \
            Literal["MAE", "RMSE", "R2"] = ["MAE", "RMSE", "R2"],
        is_pca: bool = False):

    if isinstance(include_plots, str):
        include_plots = [include_plots]

    expr_results_metrics_df = expr_results_metrics_df.loc[include_plots]
    pyspoc_results_metrics_df = pyspoc_results_metrics_df.loc[include_plots]
    alphas = [round(float(a), 4) for a in expr_results_metrics_df.columns.to_list()]
    n_metrics = expr_results_metrics_df.shape[0]
    fig, axs = plt.subplots(ncols=n_metrics, figsize=(5*n_metrics,6))

    for i in range(n_metrics):
        expr_metrics = expr_results_metrics_df.iloc[i]
        metric_name = str(expr_metrics.name)
        pyspoc_metrics = pyspoc_results_metrics_df.iloc[i]
        plot_scatter(expr_metrics,
                     pyspoc_metrics,
                     show_line=True,
                     ax=axs[i])

        xlab = "Average Expression LASSO Error/Metric"

        if is_pca:
            xlab += " (PCA)"

        ylab = "pySPOC Expression Features LASSO Error/Metric"

        if is_pca:
            ylab += " (PCA)"

        title = f"{metric_name.upper()} Results (Average Expression vs. pySPoC)"

        if is_pca:
            title += " (PCA)"

        axs[i].set_xlabel(xlab)
        axs[i].set_ylabel(ylab)
        axs[i].set_title(title)
        axs[i].grid(alpha=0.3)
        ymax = axs[i].get_ylim()[1]

        for xi, yi, alpha in zip(expr_metrics, pyspoc_metrics, alphas):
            label = f"$\\alpha={alpha}$"
            offset = (2, -12) if yi <= 0.8 * ymax else (-16, -12)
            axs[i].annotate(
                label,
                (xi, yi),
                xytext=offset,           # Offset from the point
                textcoords="offset points",
                fontsize=9,
            )

    fig.tight_layout()
    return fig, axs


def plot_scatter(
        xs,
        ys,
        point_alpha: float = 0.7,
        show_line: bool = False,
        line_label: str | None = None,
        ax: Axes | None = None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(
        xs,
        ys,
        s=30,             # Point size
        alpha=point_alpha,        # Point transparency
        color="tab:blue",
        edgecolors="black",
        linewidths=0.5)

    if show_line:
        add_xy_line(ax, line_label)


def add_xy_line(ax: Axes,
                label: str | None = None):
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    limits = [
        min(xlim[0], ylim[0]),
        max(xlim[1], ylim[1])]
    ax.plot(
        limits,
        limits,
        linestyle="--",
        color="tab:green",
        linewidth=2,
        label=label)
