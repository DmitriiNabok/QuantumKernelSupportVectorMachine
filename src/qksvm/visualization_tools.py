# Global imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_dataset(
    X,
    y,
    ax,
    axes,
    marker="o",
    size=50,
    alpha=1.0,
    stepsize=0.5,
    grid=False,
    cmap=ListedColormap(["#FF0000", "#0000FF"]),
):
    """Simple routine to visualize a 2D dataset"""
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cmap,
        edgecolors="k",
        marker=marker,
        s=size,
        alpha=alpha,
    )
    ax.axis(axes)
    ax.grid(grid, which="both")
    ax.set_xlabel(r"$x_1$", fontsize=24)
    ax.set_ylabel(r"$x_2$", fontsize=24, rotation=0)
    ax.xaxis.set_ticks(np.arange(axes[0], axes[1] + 0.01, stepsize))
    ax.yaxis.set_ticks(np.arange(axes[2], axes[3] + 0.01, stepsize))


def plot_predictions(clf, ax, axes, N=100, cmap=plt.cm.RdBu):
    """Plot prediction 2D map for a binary classifier"""
    x0s = np.linspace(axes[0], axes[1], N)
    x1s = np.linspace(axes[2], axes[3], N)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    ax.contourf(x0, x1, y_pred, cmap=cmap, alpha=0.9)


def plot_decisions(clf, ax, axes, N=100, cmap=plt.cm.RdBu):
    """Plot the 2D decision function for a binary classifier"""
    x0s = np.linspace(axes[0], axes[1], N)
    x1s = np.linspace(axes[2], axes[3], N)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_decision = clf.decision_function(X).reshape(x0.shape)
    ax.contourf(x0, x1, y_decision, cmap=cmap, alpha=0.9)
