"""lamos_core reproduces the original LAMOS optimization exactly (same seed, same output)."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import lamos_core as core  # noqa: E402
import legacy_reference as legacy  # noqa: E402

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def data():
    d0 = gpd.read_file(ROOT / "sample_data" / "dodger_of_20152019.shp")
    da = d0[d0.game_year.between(2016, 2019) & d0.month.between(4, 8) & d0.inning.between(1, 6)
            & (d0.stand == "L") & (d0.p_thrws == "R")]
    return d0, da


def configure(p):
    for k, v in dict(weight_col=p.weight_col, demand_buffer_distance=p.coverage_dist, num_fac=p.num_fac,
                     iteration=p.iterations, maximum_lambda=p.num_partitions, max_iter=p.max_iter,
                     threshold_dist=p.threshold_dist).items():
        setattr(legacy, k, v)


def test_optimize_matches_original(data):
    _, da = data
    p = core.Params(weight_col="ewoba", num_fac=3, coverage_dist=90, num_partitions=2,
                    threshold_dist=1.0, max_iter=4, iterations=1, seed=None)
    configure(p)
    np.random.seed(1)
    demands = da.sample(30)[["ewoba", "geometry"]].reset_index(drop=True)
    np.random.seed(5)
    old = legacy.analysis(demands)
    np.random.seed(5)
    new = core.optimize(demands, p)
    a = sorted(zip(np.round(old.geometry.x, 6), np.round(old.geometry.y, 6)))
    b = sorted(zip(np.round(new.geometry.x, 6), np.round(new.geometry.y, 6)))
    assert a == b


def test_objective_values_match_original(data):
    _, da = data
    configure(core.Params(weight_col="ewoba", coverage_dist=90))
    np.random.seed(3)
    sol = da.sample(6)[["geometry"]].reset_index(drop=True)
    sol["uniq_ind"] = [0, 0, 0, 1, 1, 1]
    old = legacy.calculate_values(da, sol)
    new = core.evaluate_solutions(da, sol, "ewoba", 90)
    assert np.allclose(old["covered_per"].astype(float), new["covered_per"])
    assert np.allclose(old["mean_dist"].astype(float), new["mean_dist"])
