"""Smoke tests: every ESDA method and the Monte Carlo analysis run on the sample data."""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lamos_core as core  # noqa: E402
import lamos_esda as esda  # noqa: E402

warnings.filterwarnings("ignore")
W = "ewoba"


@pytest.fixture(scope="module")
def data():
    d0 = gpd.read_file(ROOT / "sample_data" / "dodger_of_20152019.shp")
    bg = gpd.read_file(ROOT / "sample_data" / "outfield_poly.shp")
    da = d0[d0.game_year.between(2016, 2019) & d0.month.between(4, 8) & d0.inning.between(1, 6)
            & (d0.stand == "L") & (d0.p_thrws == "R")]
    return d0, da, bg, esda.study_extent(d0, bg), core.convex_hull_area(d0)


def test_case_study_subset(data):
    d0, da, *_ = data
    assert len(d0) == 8788 and len(da) == 1228


def test_quadrat(data):
    d0, da, bg, ext, _ = data
    n, diffs = esda.quadrat_summary(d0, da, W, ext, bg)
    assert n[0] == 1 and n[-1] == 30
    assert np.median(diffs[-1]) > np.median(diffs[1])     # finer quadrats differ more
    ex, ey, keep = esda.quadrat_grid(ext, 7, bg)
    assert np.isclose(esda.quadrat_shares(da, W, ex, ey, keep).sum(), 100)
    esda.draw_quadrat(Figure(), d0, da, W, ext, (7, 28), bg)


def test_surface_difference(data):
    d0, da, bg, ext, _ = data
    gx, gy, diff = esda.surface_difference(d0, da, W, ext, bg, grid_n=30, max_points=200, seed=1)
    assert diff.shape == (30, 30) and np.nanmax(np.abs(diff)) <= 1
    esda.draw_surface_difference(Figure(), gx, gy, diff, bg)


def test_lisa(data):
    d0, da, bg, ext, hull = data
    i0, lab0 = esda.lisa(d0, W, permutations=49, seed=1)
    ia, laba = esda.lisa(da, W, permutations=49, seed=1)
    assert 0 < i0 < 0.5 and len(lab0) == len(d0)
    esda.draw_lisa(Figure(), d0, (i0, lab0), da, (ia, laba), ext, bg, hull)


def test_kde(data):
    d0, da, bg, ext, _ = data
    axes = esda.draw_kde_grid(Figure(), da, ["game_year", "month", "inning"], 2017, ext, bg)
    xl = {tuple(np.round(ax.get_xlim(), 6)) for ax in axes.ravel()}
    assert len(xl) == 1                                      # common extent in every panel


def test_ks_ripley(data):
    d0, da, bg, ext, hull = data
    esda.draw_ks_ripley(Figure(), d0, da, W, bg, hull, iterations=5, voronoi_iterations=2, seed=1)


def test_kmeans(data):
    d0, da, bg, ext, _ = data
    labels, centres = esda.kmeans_fit(d0, W, 3, seed=0)
    assert centres.shape == (3, 2)
    esda.draw_kmeans(Figure(), d0, da, W, 4, ext, bg, seed=0)


def test_monte_carlo_reproducible(data):
    d0, da, bg, ext, hull = data
    p = core.Params(weight_col=W, num_fac=3, coverage_dist=90, num_partitions=2, threshold_dist=5,
                    max_iter=5, iterations=2, num_vor=2, seed=7)
    r1 = core.run_monte_carlo(da, hull, p)
    r2 = core.run_monte_carlo(da, hull, p)
    assert len(r1.global_solutions) > 0
    assert r1.global_values[["covered_per", "mean_dist"]].equals(r2.global_values[["covered_per", "mean_dist"]])
    esda.draw_results(Figure(), r1, da, bg, p)
    out = core.solutions_for_export(r1.global_solutions)
    assert out.geometry.geom_type.eq("Point").all()
