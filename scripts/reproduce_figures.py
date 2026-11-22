"""
Reproduce the ESDA figures (Figures 6-11) and the Monte Carlo result window
(Figure 12) of the paper from the sample data, without the GUI.

    python scripts/reproduce_figures.py                 # paper case study, all figures
    python scripts/reproduce_figures.py --only 6 11     # selected figures
    python scripts/reproduce_figures.py --quick         # small settings for a fast check

Case study (Section 5 of the paper): game_year 2016-2019, month 4-8, inning 1-6,
left-handed batters (stand = L) facing right-handed pitchers (p_thrws = R);
weight = ewoba (xwOBA); 3 outfielders; coverage distance 90 ft.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lamos_core as core  # noqa: E402
import lamos_esda as esda  # noqa: E402

DEMAND = ROOT / "sample_data" / "dodger_of_20152019.shp"
AREA = ROOT / "sample_data" / "outfield_poly.shp"
WEIGHT = "ewoba"


def case_study(d0):
    return d0[d0.game_year.between(2016, 2019) & d0.month.between(4, 8) & d0.inning.between(1, 6)
              & (d0.stand == "L") & (d0.p_thrws == "R")]


def save(fig, out, name):
    path = out / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "figures"), help="output folder (default: figures/)")
    ap.add_argument("--only", nargs="*", type=int, default=[6, 7, 8, 9, 10, 11, 12])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="reduced iterations for a fast smoke test")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d0 = gpd.read_file(DEMAND)
    bg = gpd.read_file(AREA)
    da = case_study(d0)
    ext = esda.study_extent(d0, bg)
    hull = core.convex_hull_area(d0)
    print(f"D_I = {len(d0):,} demands, D'_I = {len(da):,} demands (case study)")

    if 6 in a.only:   # Figure 6 (a) box plots, (b) resolution 7, (c) resolution 28
        t = time.time()
        summary = esda.quadrat_summary(d0, da, WEIGHT, ext, bg)
        fig = Figure(figsize=(11, 7.5), layout="constrained")
        esda.draw_quadrat_summary(fig.add_subplot(), summary, highlight=(7, 28))
        save(fig, out, "fig06a_quadrat_boxplots.png")
        for tag, n in (("b", 7), ("c", 28)):
            fig = Figure(figsize=(12, 4.6), layout="constrained")
            axes = esda.draw_quadrat_maps(fig, d0, da, WEIGHT, ext, n, bg)
            if n == 28:   # cluster on the right side of the outfield discussed in Section 6
                axes[1].add_patch(Circle((140, 200), 55, fill=False, color="black", linewidth=2.5))
            save(fig, out, f"fig06{tag}_quadrat_resolution{n}.png")
        print(f"  weighted quadrat analysis: {time.time() - t:.1f}s")
    if 7 in a.only:   # Figure 7 (a) top view, (b) side view
        t = time.time()
        gx, gy, diff = esda.surface_difference(d0, da, WEIGHT, ext, bg, max_points=300 if a.quick else 800,
                                               seed=a.seed)
        for tag, view, size in (("a", "top", (8, 5.6)), ("b", "side", (8, 6.5))):
            fig = Figure(figsize=size, layout="constrained")
            esda.draw_surface_difference(fig, gx, gy, diff, bg, views=(view,))
            save(fig, out, f"fig07{tag}_surface_{view}.png")
        print(f"  demand surface difference: {time.time() - t:.1f}s")
    if 8 in a.only:   # Figure 8 (a) LISA from D_I, (b) LISA from D'_I
        t = time.time()
        perms = 99 if a.quick else 999
        area = esda._area(bg, hull)
        for tag, g, name, label in (("a", d0, "DI", esda.D0_LABEL), ("b", da, "DpI", esda.DA_LABEL)):
            res = esda.lisa(g, WEIGHT, permutations=perms, seed=a.seed)
            fig = Figure(figsize=(8, 5.4), layout="constrained")
            ax = fig.add_subplot()
            esda.draw_lisa_panel(ax, g, res, label,
                                 ext, area, bg, title="")
            for (cx, cy), color in (((-185, 235), "tab:blue"), ((135, 150), "tab:red")):
                ax.add_patch(Circle((cx, cy), 58, fill=False, color=color, linewidth=2.5))   # regions in Section 6
            save(fig, out, f"fig08{tag}_lisa_{name}.png")
            print(f"  LISA {name}: Moran's I = {res[0]:.3f}")
        print(f"  LISA: {time.time() - t:.1f}s")
    if 9 in a.only:   # Figure 9: KDE by month (rows) and inning (columns), one figure per year
        t = time.time()
        for year in sorted(da.game_year.unique()):
            sub = da[da.game_year == year]
            fig = Figure(figsize=esda.kde_figsize(sub.month.nunique(), sub.inning.nunique(), ext),
                         layout="constrained")
            esda.draw_kde_grid(fig, da, ["game_year", "month", "inning"], year, ext, bg)
            save(fig, out, f"fig09_kde_{year}.png")
        print(f"  KDE: {time.time() - t:.1f}s")
    if 10 in a.only:  # Figure 10: K-S tests and Ripley's K
        t = time.time()
        fig = Figure(figsize=(16, 13), layout="constrained")
        esda.draw_ks_ripley(fig, d0, da, WEIGHT, bg, hull, num_points=60,
                            iterations=20 if a.quick else 100, seed=a.seed, voronoi=False)
        save(fig, out, "fig10_ks_ripley.png")
        print(f"  K-S and Ripley's K: {time.time() - t:.1f}s")
    if 11 in a.only:  # Figure 11 (a) three and (b) four outfielders
        t = time.time()
        for tag, k in (("a", 3), ("b", 4)):
            fig = Figure(figsize=(15, 6), layout="constrained")
            esda.draw_kmeans(fig, d0, da, WEIGHT, k, ext, bg, seed=a.seed)
            save(fig, out, f"fig11{tag}_kmeans_k{k}.png")
        print(f"  K-means: {time.time() - t:.1f}s")
    if 12 in a.only:
        t = time.time()
        # Figure 5 parameters (number of partitions, threshold, iterations, Voronoi iterations)
        p = core.Params(weight_col=WEIGHT, num_fac=3, coverage_dist=90, num_partitions=30,
                        threshold_dist=5, max_iter=10, iterations=5,
                        num_vor=5 if a.quick else 100, seed=a.seed)
        res = core.run_monte_carlo(da, hull, p)
        fig = Figure(figsize=(14, 13), layout="constrained")
        esda.draw_results(fig, res, da, bg, p)
        save(fig, out, "fig12_tradeoff.png")
        g = res.global_values
        print(f"  Analyze: {len(res.simulated_values)} local solutions, {len(g)} global non-dominated; "
              f"coverage {g.covered_per.min():.1f}-{g.covered_per.max():.1f}%, "
              f"mean distance {g.mean_dist.min():.1f}-{g.mean_dist.max():.1f} ft ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
