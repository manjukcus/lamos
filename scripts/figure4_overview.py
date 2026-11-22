"""
Figure 4 of the paper: overview of the batted-ball data (not part of the GUI).

(a) Spatial distribution of all batted balls, one small-multiple map per
    xwOBA class (Jenks natural breaks, k = 5), with a histogram of xwOBA that
    shows the class breaks.
(b) xwOBA of every batted ball by date on the full value range, with a
    seasonal LOESS trend (local quadratic, span 0.75, 95% confidence band)
    fitted to all values of each season; the lower panel zooms the y-axis
    to the trend lines only.

    python scripts/figure4_overview.py [--out figures]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEIGHT = "ewoba"
COLORS = ["#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"]
FOCUS = "#cb181d"
SEASON_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def jenks_breaks(y, k=5):
    import mapclassify
    return np.asarray(mapclassify.FisherJenks(y, k=k).bins)


def loess(x, y, xs, span=0.75, degree=2, level=0.95):
    """LOESS fit with pointwise confidence band (tricube weights, local polynomial),
    following the defaults of R's loess(), which geom_smooth() uses."""
    n = len(x)
    k = int(np.ceil(span * n))

    def smoother_rows(points):
        rows = []
        for x0 in points:
            d = np.abs(x - x0)
            h = np.partition(d, k - 1)[k - 1]
            w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
            X = np.vander(x - x0, degree + 1, increasing=True)
            XtW = X.T * w
            rows.append(np.linalg.solve(XtW @ X, XtW)[0])
        return np.array(rows)

    L_data = smoother_rows(x)
    resid = y - L_data @ y
    df = n - np.trace(L_data)
    sigma = np.sqrt((resid ** 2).sum() / df)
    L = smoother_rows(xs)
    fit = L @ y
    half = stats.t.ppf((1 + level) / 2, df) * sigma * np.sqrt((L ** 2).sum(axis=1))
    return fit, fit - half, fit + half


def panel_a(d0, bg, out):
    y = d0[WEIGHT].to_numpy()
    upper = jenks_breaks(y)
    lower = np.r_[y.min(), upper[:-1]]
    cls = np.searchsorted(upper, y, side="left")
    fig = Figure(figsize=(10, 6.0), layout="constrained")
    axes = fig.subplots(2, 3)
    xmin, ymin, xmax, ymax = bg.total_bounds
    for i, ax in enumerate(axes.ravel()[:5]):
        m = cls == i
        bg.boundary.plot(ax=ax, color="black", linewidth=0.8)
        ax.scatter(d0.geometry.x[~m], d0.geometry.y[~m], s=1, color="#e0e0e0", rasterized=True)
        ax.scatter(d0.geometry.x[m], d0.geometry.y[m], s=3, color=FOCUS, edgecolor="none", rasterized=True)
        lo = 0 if i == 0 else lower[i] + 0.01
        ax.set_title(f"xwOBA {lo:.2f}-{upper[i]:.2f}  (n = {m.sum():,})", fontsize=9)
        ax.set_xlim(xmin - 10, xmax + 10)
        ax.set_ylim(ymin - 10, ymax + 10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    ax = axes.ravel()[5]
    bins = np.linspace(0, y.max(), 60)
    for i in range(5):
        ax.hist(y[cls == i], bins=bins, color=COLORS[i], edgecolor="none")
    for b in upper[:-1]:
        ax.axvline(b, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("xwOBA")
    ax.set_ylabel("Batted balls")
    ax.set_title("Class breaks: Jenks natural breaks (k = 5)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.savefig(out / "fig04a_xwoba_classes.png", dpi=200)
    return upper


def panel_b(d0, out):
    t = pd.to_datetime(d0["game_date"])
    y = d0[WEIGHT].to_numpy()
    fig = Figure(figsize=(8, 8), layout="constrained")
    top, bottom = fig.subplots(2, 1, height_ratios=[2, 1], sharex=True)
    top.scatter(t, y, s=2, color="black", alpha=0.15, edgecolor="none", rasterized=True)
    for i, yr in enumerate(sorted(t.dt.year.unique())):
        m = (t.dt.year == yr).to_numpy()
        days = (t[m] - pd.Timestamp(f"{yr}-01-01")).dt.days.to_numpy(dtype=float)
        grid = np.linspace(days.min(), days.max(), 60)
        fit, lo, hi = loess(days, y[m], grid)
        dates = pd.Timestamp(f"{yr}-01-01") + pd.to_timedelta(grid, unit="D")
        for ax in (top, bottom):
            ax.fill_between(dates, lo, hi, color=SEASON_COLORS[i], alpha=0.3, linewidth=0)
            ax.plot(dates, fit, color=SEASON_COLORS[i], linewidth=2, label=str(yr))
    top.set_ylim(0, max(1.95, y.max()))
    top.set_ylabel("xwOBA (every batted ball)")
    top.legend(title="Season", ncol=5, fontsize=8, loc="upper center")
    bottom.set_ylabel("xwOBA trend\n(zoomed y-axis)")
    bottom.set_xlabel("Date")
    for ax in (top, bottom):
        ax.grid(True, alpha=0.3)
    fig.savefig(out / "fig04b_xwoba_trends.png", dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d0 = gpd.read_file(ROOT / "sample_data" / "dodger_of_20152019.shp")
    bg = gpd.read_file(ROOT / "sample_data" / "outfield_poly.shp")
    upper = panel_a(d0, bg, out)
    panel_b(d0, out)
    print("Jenks class upper bounds:", np.round(upper, 2))
    print(f"saved fig04a_xwoba_classes.png and fig04b_xwoba_trends.png in {out}")


if __name__ == "__main__":
    main()
