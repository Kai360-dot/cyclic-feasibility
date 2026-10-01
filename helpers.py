"""
Generic helpers: trellis plots of 4-D point clouds, Sobol designs and a small
MLP surrogate with k-fold cross-validation and adaptive width selection.

    X, Y = ...                                   # (n, d_in), (n, d_out) raw units
    width, table = select_width(X, Y)            # doubles hidden width until CV stops improving
    model = fit_mlp(X, Y, hidden=width)          # numpy MLP, raw units in and out
    model.export("surrogate.txt")
"""
from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc

__all__ = ["Trellis", "generate_sobol_points", "Surrogate", "fit_mlp", "cross_validate", "select_width", "hausdorff"]


# ---------------------------------------------------------------------------
# Plotting: trellis of (x, y) panels binned by two further variables
# ---------------------------------------------------------------------------
class Trellis:
    """Grid of (x, y) panels for 4-D point clouds: columns bin a third variable (low to high,
    left to right), rows bin a fourth (high to low, top to bottom). A strip above each column
    and right of each row marks its bin within the full range.

        grid = Trellis(np.linspace(300, 350, 4), np.linspace(2, 30, 4),        # bin edges
                       xlabel="$T_1$ [K]", ylabel=r"$\tau_1$ [min]",
                       col_label="$T_2$ [K]", row_label=r"$\tau_2$ [min]")
        grid.scatter(dead.T1, dead.tau1, dead.T2, dead.tau2, color="crimson", label="dead")
        pcs = grid.scatter(live.T1, live.tau1, live.T2, live.tau2, c=margin)   # as Axes.scatter
        grid.fig.colorbar(pcs[0], ax=grid.axes, label="margin")
        grid.fig.legend()
        grid.fig.savefig("trellis.png")

    `fig` and `axes` (2-D array, `axes[0]` the top row) are the plain matplotlib objects.
    Edges need not be uniform; `np.histogram_bin_edges(values, 3)` takes them from data. Bins
    are [lo, hi), the last one [lo, hi]; points outside the edges appear in no panel (warned).
    Further keywords go to `plt.subplots`.
    """

    def __init__(self, col_edges, row_edges, *, xlabel=None, ylabel=None,
                 col_label=None, row_label=None, **subplots_kw):
        self.col_edges = np.asarray(col_edges, dtype=float)
        self.row_edges = np.asarray(row_edges, dtype=float)
        for edges in (self.col_edges, self.row_edges):
            if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError("bin edges must be 1-D, strictly increasing, at least two")
        nrows, ncols = self.row_edges.size - 1, self.col_edges.size - 1
        subplots_kw = {"figsize": (3.0 * ncols, 3.0 * nrows), "sharex": True, "sharey": True,
                       "layout": "constrained", **subplots_kw}
        self.fig, self.axes = plt.subplots(nrows, ncols, squeeze=False, **subplots_kw)

        def title(label, lo, hi):
            return f"{label}: {lo:.4g} to {hi:.4g}" if label else f"{lo:.4g} to {hi:.4g}"

        for ax, lo, hi in zip(self.axes[0], self.col_edges[:-1], self.col_edges[1:]):
            strip = ax.inset_axes([0.0, 1.04, 1.0, 0.06])
            strip.axvspan(lo, hi, color="0.55")
            strip.set(xlim=self.col_edges[[0, -1]], xticks=[], yticks=[])
            strip.set_title(title(col_label, lo, hi), fontsize="medium")
        for ax, lo, hi in zip(self.axes[::-1, -1], self.row_edges[:-1], self.row_edges[1:]):
            strip = ax.inset_axes([1.04, 0.0, 0.06, 1.0])
            strip.axhspan(lo, hi, color="0.55")
            strip.set(ylim=self.row_edges[[0, -1]], xticks=[], yticks=[])
            strip.yaxis.set_label_position("right")
            strip.set_ylabel(title(row_label, lo, hi), rotation=270, va="bottom")
        if xlabel:
            self.fig.supxlabel(xlabel)
        if ylabel:
            self.fig.supylabel(ylabel)

    def split(self, col, row):
        """Yield `(ax, mask)` per panel, `mask` selecting the points whose `col` and `row` values
        fall in the panel's bins. For drawing anything other than a scatter."""
        nrows, ncols = self.axes.shape
        j, i = _bin_index(col, self.col_edges), _bin_index(row, self.row_edges)
        outside = (j < 0) | (j >= ncols) | (i < 0) | (i >= nrows)
        if outside.any():
            warnings.warn(f"{outside.sum()} of {outside.size} points lie outside the bin edges "
                          "and appear in no panel", stacklevel=3)
        for (r, c), ax in np.ndenumerate(self.axes):
            yield ax, (i == nrows - 1 - r) & (j == c)

    def scatter(self, x, y, col, row, **kwargs):
        """`Axes.scatter` of the points of every panel; returns the collections, one per panel
        (row-major). Per-point keywords (`c`, `s`, ...) are split along with the points, a
        numeric `c` shares one colour scale over all panels, `label` makes one legend entry."""
        x, y = np.asarray(x), np.asarray(y)
        per_point = {k: np.asarray(v) for k, v in kwargs.items()
                     if not isinstance(v, str) and np.ndim(v) >= 1 and len(v) == x.size}
        c = per_point.get("c")
        if c is not None and c.ndim == 1 and c.dtype.kind in "fiu" and c.size and "norm" not in kwargs:
            kwargs.setdefault("vmin", np.nanmin(c))
            kwargs.setdefault("vmax", np.nanmax(c))
        label = kwargs.pop("label", None)
        collections = []
        for ax, mask in self.split(col, row):
            point_kw = {k: v[mask] for k, v in per_point.items()}
            collections.append(ax.scatter(x[mask], y[mask], label=label, **{**kwargs, **point_kw}))
            label = None
        return collections


def _bin_index(values, edges):
    """Bin of each value as `np.histogram` assigns it; no valid index for values outside."""
    values = np.asarray(values, dtype=float)
    idx = np.searchsorted(edges, values, side="right") - 1
    idx[values == edges[-1]] = edges.size - 2
    return idx


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def generate_sobol_points(n: int, bounds, seed: int = 42):
    """`n` scrambled Sobol points in the box `bounds` (2 x d: lower row, upper row).
    Use a power of two for `n`; other counts truncate the sequence."""
    m = int(np.ceil(np.log2(n)))
    pts = qmc.Sobol(d=bounds.shape[1], seed=seed).random_base2(m)
    return qmc.scale(pts, bounds[0], bounds[1])[:n]


# ---------------------------------------------------------------------------
# Surrogate: tanh MLP as plain numpy, standardisation folded into the weights
# ---------------------------------------------------------------------------
class Surrogate:
    """Plain-numpy MLP: y = W_L(tanh(... tanh(W_0 x + b_0) ...)) + b_L, raw units."""

    def __init__(self, W, b):
        self.W, self.b = W, b

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        h = np.atleast_2d(x)
        for i, (Wl, bl) in enumerate(zip(self.W, self.b)):
            h = h @ Wl.T + bl
            if i + 1 < len(self.W):
                h = np.tanh(h)
        return h[0] if x.ndim == 1 else h

    def export(self, path):
        with open(path, "w") as f:
            f.write(f"{len(self.W)}\n")
            for Wl, bl in zip(self.W, self.b):
                f.write(f"{Wl.shape[1]} {Wl.shape[0]}\n")
                for row in Wl:
                    f.write(" ".join(f"{v:.17g}" for v in row) + "\n")
                f.write(" ".join(f"{v:.17g}" for v in bl) + "\n")

    @classmethod
    def load(cls, path):
        """Read a net written by `export`."""
        with open(path) as f:
            it = iter(f.read().split())
        W, b = [], []
        for _ in range(int(next(it))):
            n_in, n_out = int(next(it)), int(next(it))
            W.append(np.array([float(next(it)) for _ in range(n_in * n_out)]).reshape(n_out, n_in))
            b.append(np.array([float(next(it)) for _ in range(n_out)]))
        return cls(W, b)


def fit_mlp(X, Y, *, hidden: int = 16, layers: int = 2, max_iter: int = 500, seed: int = 0) -> Surrogate:
    """
    Fit a tanh MLP with L-BFGS on standardised data and return it with the
    standardisation folded into the first and last layer.
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    xmu, xsig = X.mean(0), X.std(0)
    ymu, ysig = Y.mean(0), Y.std(0)
    if np.any(xsig == 0) or np.any(ysig == 0):
        raise ValueError("constant column in X or Y; drop it before fitting")
    tx = torch.tensor((X - xmu) / xsig)
    ty = torch.tensor((Y - ymu) / ysig)

    torch.manual_seed(seed)
    dims = [X.shape[1]] + [hidden] * layers + [Y.shape[1]]
    mods = []
    for i in range(len(dims) - 1):
        mods += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
    net = nn.Sequential(*mods[:-1]).double()

    opt = torch.optim.LBFGS(net.parameters(), max_iter=max_iter, history_size=50,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

    def closure():
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(tx), ty)
        loss.backward()
        return loss

    opt.step(closure)

    lins = [m for m in net if isinstance(m, nn.Linear)]
    W = [m.weight.detach().numpy().copy() for m in lins]
    b = [m.bias.detach().numpy().copy() for m in lins]
    b[0] = b[0] - W[0] @ (xmu / xsig)
    W[0] = W[0] / xsig
    W[-1] = ysig[:, None] * W[-1]
    b[-1] = ysig * b[-1] + ymu
    return Surrogate(W, b)


def cross_validate(X, Y, *, k: int = 5, seed: int = 0, **fit_kw):
    """
    k-fold cross-validation of `fit_mlp(**fit_kw)`.  Returns the RMSE per
    output in raw units and the out-of-fold predictions (n, d_out).
    """
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    Y_oof = np.empty_like(Y)
    for test in np.array_split(np.random.default_rng(seed).permutation(len(X)), k):
        train = np.setdiff1d(np.arange(len(X)), test)
        Y_oof[test] = fit_mlp(X[train], Y[train], **fit_kw)(X[test])
    return np.sqrt(((Y_oof - Y) ** 2).mean(0)), Y_oof


def select_width(X, Y, *, start: int = 4, tol: float = 0.05, k: int = 5, verbose: bool = True, **fit_kw):
    """
    Double the hidden width from `start` until the cross-validated error,
    averaged over outputs in units of their standard deviation, improves by
    less than `tol` relative.  Returns the best width and a {width: score} table.
    """
    Y = np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    ysig = Y.std(0)
    table, width = {}, start
    while True:
        rmse, _ = cross_validate(X, Y, k=k, hidden=width, **fit_kw)
        table[width] = float((rmse / ysig).mean())
        if verbose:
            print(f"  hidden {width:4d}  CV rmse/std {table[width]:.4f}")
        prev = table.get(width // 2)
        if prev is not None and table[width] > (1 - tol) * prev:
            break
        width *= 2
    return min(table, key=table.get), table

def eucl_dist(x, y):
    """Compute the euclidian distance"""
    assert x.ndim == 1 and y.ndim == 1
    assert x.shape == y.shape
    return np.sqrt(((x - y)**2).sum())

def get_sup_d_x_Y(X, Y):
    """Get the largest distance among any point in x to their closest neighbor in Y"""
    largest = -1
    for x in X:
        cmin = np.inf # current closest
        for y in Y:
            dist = eucl_dist(x, y)
            if dist < cmin:
                xcand = dict(x=x, y=y) 
            cmin = min(cmin, dist)
        if cmin > largest: # update
            cand = xcand
        largest = max(largest, cmin)
    return largest, cand

def get_sup_d_y_X(X, Y):
    return get_sup_d_x_Y(Y, X)

def hausdorff(X, Y, verbose: bool = True):
    """Uses euclidian distance"""
    assert len(X) > 0 and len(Y) > 0
    a, a_cand = get_sup_d_x_Y(X, Y)
    b, b_cand = get_sup_d_y_X(X, Y)
    pair = a_cand if a > b else b_cand
    if verbose:
        print(f"Hausdorff distance between: {pair['x']} and {pair['y']}")
    return max(a, b)
