"""
LAMOS core: multi-objective (coverage + access) location optimization.

This module contains the optimization and Monte Carlo routines used by the
LAMOS GUI (``lamos_app.py``). It has no GUI dependency, so it can be imported
and tested on its own.

Algorithm (unchanged from the original LAMOS implementation, Cho et al. 2023):

1. Demands are buffered by the coverage distance; the buffer boundaries are
   polygonized into "faces" (regions covered by the same set of demands).
2. Starting from ``num_fac`` randomly sampled demands, each iteration
   allocates demands to the nearest facility, moves each facility to its
   weighted (Weber) centre, snaps candidates to the nearest point of every
   face, evaluates coverage and weighted distance for each candidate, keeps
   the non-dominated candidates and picks the next solution with the
   lambda-interval rule (``generate_next_solution``).
3. The loop stops when facilities move less than ``threshold_dist``.
4. Monte Carlo: representative demands are generated inside Voronoi regions
   of a random sample of the scenario demands (``simulate_demands``) and the
   optimization is repeated ``num_vor`` times; the global non-dominated set
   is evaluated against the scenario demands.

Changes relative to the original notebook version:
- fixed: the coverage step received the unallocated demand table, which made
  "3. Analyze" fail with ``KeyError: 'nearest_id'``;
- fixed: solution ids are offset per Monte Carlo run instead of assuming every
  solution has exactly ``num_fac`` points (crashed when a facility received no
  demand);
- distance computations are vectorised with NumPy/Shapely (same results,
  much faster); parameters are passed explicitly instead of via globals;
- an optional random seed makes runs reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import nearest_points, polygonize, unary_union, voronoi_diagram
from scipy.spatial.distance import cdist


@dataclass
class Params:
    """Model and computation parameters (GUI: '2. Parameter Setting')."""
    weight_col: str
    num_fac: int = 3                 # number of facilities
    coverage_dist: float = 90.0      # coverage standard distance
    num_partitions: int = 2          # 'Number of partitions' (lambda intervals)
    threshold_dist: float = 1.0      # stop when facilities move less than this
    max_iter: int = 10               # maximum iterations per partition
    iterations: int = 5              # random restarts per optimization
    num_vor: int = 10                # Monte Carlo (Voronoi) simulations
    demands_per_facility: int = 20   # sampled demands per facility in each simulation
    seed: int | None = 42            # random seed (None = not reproducible)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def union_all(geoms) -> shapely.Geometry:
    """GeoSeries/GeoDataFrame union that works on geopandas < 1.0 and >= 1.0."""
    if hasattr(geoms, "union_all"):
        return geoms.union_all()
    return geoms.unary_union


def xy(gdf: gpd.GeoDataFrame | gpd.GeoSeries, column: str | None = None) -> np.ndarray:
    geom = gdf[column] if column is not None else gdf.geometry
    geom = gpd.GeoSeries(geom)
    return np.column_stack([geom.x.to_numpy(), geom.y.to_numpy()])


def empty_points(crs=None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def convex_hull_area(demand_gdf: gpd.GeoDataFrame, pad: float = 5.0) -> gpd.GeoDataFrame:
    """Padded convex hull of all demands (used to clip Voronoi regions)."""
    hull = union_all(demand_gdf.geometry).convex_hull.buffer(pad)
    return gpd.GeoDataFrame(geometry=[Polygon(hull.exterior)], crs=demand_gdf.crs)


# --------------------------------------------------------------------------
# optimization building blocks
# --------------------------------------------------------------------------
def create_intersected_areas(gdf: gpd.GeoDataFrame, buffer_distance: float) -> gpd.GeoDataFrame:
    """Polygonize the arrangement of demand coverage buffers into faces."""
    buffers = gdf.geometry.buffer(buffer_distance)
    unioned_polygon = unary_union(list(buffers))
    unioned_lines = unary_union(list(buffers.boundary))
    combined_lines = unary_union([unioned_polygon.boundary, unioned_lines])
    faces = list(polygonize(combined_lines))
    out = gpd.GeoDataFrame(geometry=faces, crs=gdf.crs)
    out["index_id"] = range(len(out))
    return out


def find_nearest(source_gdf: gpd.GeoDataFrame, target_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Allocate each source point to the nearest target (by 'index_id')."""
    src = xy(source_gdf)
    tgt = xy(target_gdf)
    dist = cdist(src, tgt)
    pos = dist.argmin(axis=1)                       # first minimum, as idxmin
    out = source_gdf.copy()
    out["nearest_id"] = target_gdf["index_id"].to_numpy()[pos]
    out["distance"] = dist[np.arange(len(src)), pos]
    return out


def weber(gdf: gpd.GeoDataFrame, uniq_id: str, weight_col: str) -> gpd.GeoDataFrame:
    """Weighted centre of the demands allocated to each facility."""
    rows = []
    gx, gy = gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()
    w = gdf[weight_col].to_numpy(dtype=float)
    ids = gdf[uniq_id].to_numpy()
    for nid in pd.unique(ids):
        m = ids == nid
        total = w[m].sum()
        cx = (gx[m] * w[m]).sum() / (total + 0.001)
        cy = (gy[m] * w[m]).sum() / (total + 0.001)
        rows.append((nid, cx, cy))
    df = pd.DataFrame(rows, columns=["index_id", "center_x", "center_y"])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.center_x, df.center_y), crs=gdf.crs)


def snap_to_faces(face_poly_gdf: gpd.GeoDataFrame, solution_weber: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """For every face, the point of the face nearest to its closest Weber point."""
    polys = face_poly_gdf.geometry.to_numpy()
    pts = solution_weber.geometry.to_numpy()
    lines = shapely.shortest_line(polys[:, None], pts[None, :])   # face -> point
    dist = shapely.length(lines)
    pos = dist.argmin(axis=1)
    chosen = lines[np.arange(len(polys)), pos]
    nearest = shapely.get_point(chosen, 0)
    out = gpd.GeoDataFrame({"x": shapely.get_x(nearest), "y": shapely.get_y(nearest)},
                           geometry=nearest, crs=face_poly_gdf.crs)
    return out


def evaluate_candidates(alloc_demand: gpd.GeoDataFrame, candidates: gpd.GeoDataFrame,
                        weight_col: str, coverage_dist: float) -> gpd.GeoDataFrame:
    """Coverage (sum of covered weight) and access (weighted distance) of each
    candidate with respect to the demands allocated to the same facility."""
    D = cdist(xy(candidates), xy(alloc_demand))
    same = candidates["nearest_id"].to_numpy()[:, None] == alloc_demand["nearest_id"].to_numpy()[None, :]
    w = alloc_demand[weight_col].to_numpy(dtype=float)
    covered = same & (D <= coverage_dist + 0.01)
    # Sum only the selected demands, in demand order. Candidates covering the same
    # demands then tie exactly, which the non-dominance filter relies on
    # (summing over a full masked row changes rounding and breaks those ties).
    sw = np.empty(len(D))
    sd = np.empty(len(D))
    swd = np.empty(len(D))
    for i in range(len(D)):
        s, c = same[i], covered[i]
        sw[i] = w[c].sum()
        sd[i] = D[i, s].sum()
        swd[i] = (D[i, s] * w[s]).sum()
    out = candidates.copy()
    out["sum_weight"] = sw
    out["sum_distance"] = sd
    out["sum_weighted_distance"] = swd
    return out


def non_dominated_per_facility(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Non-dominated candidates (max coverage, min weighted distance) per facility."""
    parts = []
    for nid in gdf["nearest_id"].unique():
        g = gdf[gdf["nearest_id"] == nid].sort_values(
            by=["sum_weight", "sum_weighted_distance"], ascending=[False, True])
        swd = g["sum_weighted_distance"].to_numpy()
        prev_min = np.minimum.accumulate(np.concatenate([[np.inf], swd[:-1]]))
        keep = swd <= prev_min
        keep[0] = True
        parts.append(g.iloc[np.flatnonzero(keep)])
    if not parts:
        return gdf.iloc[0:0]
    return pd.concat(parts)


def generate_next_solution(nd_gdf: gpd.GeoDataFrame, partition: int, num_partitions: int) -> gpd.GeoDataFrame:
    """Lambda-interval rule: partition 0 takes the minimum-coverage end of each
    facility's non-dominated set, later partitions the maximum-coverage end."""
    q = partition / (num_partitions - 1) if num_partitions > 1 else 0
    grp = nd_gdf.groupby("nearest_id")["sum_weight"]
    idx = grp.idxmin() if q == 0 else grp.idxmax()
    return nd_gdf.loc[idx]


def mean_move_distance(prev: gpd.GeoDataFrame, nxt: gpd.GeoDataFrame) -> float:
    a = prev.set_index("nearest_id")
    b = nxt.set_index("nearest_id").reindex(a.index)
    valid = b.geometry.notna().to_numpy()
    if not valid.any():
        return float("inf")
    return float(gpd.GeoSeries(a.geometry[valid]).distance(gpd.GeoSeries(b.geometry[valid])).mean())


def evaluate_solutions(demand_gdf: gpd.GeoDataFrame, solutions: gpd.GeoDataFrame,
                       weight_col: str, coverage_dist: float) -> pd.DataFrame:
    """Objective values of every solution (grouped by 'uniq_ind') on a demand set."""
    w = demand_gdf[weight_col].to_numpy(dtype=float)
    total = w.sum()
    rows = []
    if len(solutions):
        D = cdist(xy(demand_gdf), xy(solutions))
        ids = solutions["uniq_ind"].to_numpy()
        for uid in pd.unique(ids):
            d = D[:, ids == uid].min(axis=1)
            rows.append((uid, w[d < coverage_dist].sum(), (d * w).sum()))   # same summation as before
    df = pd.DataFrame(rows, columns=["uniq_ind", "sum_expected_va", "weighted_distance"])
    df["covered_per"] = 100 * df["sum_expected_va"] / (total + 0.001)
    df["mean_dist"] = df["weighted_distance"] / (total + 0.001)
    return df


def non_dominated_solution_ids(values: pd.DataFrame, cov: str, dist: str, strict: bool) -> list:
    """Ids of non-dominated rows (maximize ``cov``, minimize ``dist``)."""
    v = values.sort_values(by=[cov, dist], ascending=[False, True])
    if v.empty:
        return []
    keep, current = [v.iloc[0]["uniq_ind"]], v.iloc[0][dist]
    for _, row in v.iloc[1:].iterrows():
        better = row[dist] < current if strict else row[dist] <= current
        if better:
            current = row[dist]
            keep.append(row["uniq_ind"])
    return keep


# --------------------------------------------------------------------------
# one optimization run and the Monte Carlo wrapper
# --------------------------------------------------------------------------
def optimize(demand_gdf: gpd.GeoDataFrame, p: Params) -> gpd.GeoDataFrame:
    """Lambda-interval heuristic on one demand set; returns its non-dominated solutions."""
    demand_gdf = demand_gdf.reset_index(drop=True)
    faces = create_intersected_areas(demand_gdf, p.coverage_dist)
    found = []
    uniq_ind = 0
    nd_empty = False
    # One iteration is a deterministic function of the current solution and of
    # whether the partition takes the minimum or maximum end of the
    # non-dominated set. Later partitions frequently repeat an earlier step
    # exactly, so steps are cached (identical results, much less computation).
    cache = {}

    def step(prev, use_min):
        key = (use_min, xy(prev).tobytes(), prev["index_id"].to_numpy().tobytes(),
               prev["nearest_id"].to_numpy().tobytes())
        if key not in cache:
            alloc_demand = find_nearest(demand_gdf, prev)
            centres = weber(alloc_demand, "nearest_id", p.weight_col)
            cand = snap_to_faces(faces, centres)
            cand = find_nearest(cand, prev)
            cand = evaluate_candidates(alloc_demand, cand, p.weight_col, p.coverage_dist)
            nd = non_dominated_per_facility(cand)
            if nd.empty:
                cache[key] = None
            else:
                nxt = generate_next_solution(nd, 0 if use_min else 1, 2).copy()
                nxt["index_id"] = range(len(nxt))
                cache[key] = nxt
        hit = cache[key]
        return None if hit is None else hit.copy()

    for _ in range(p.iterations):
        prev = demand_gdf.sample(n=p.num_fac)
        prev["index_id"] = range(len(prev))
        prev["nearest_id"] = range(len(prev))
        for part in range(p.num_partitions):
            use_min = (part / (p.num_partitions - 1) if p.num_partitions > 1 else 0) == 0
            for _ in range(p.max_iter):
                nxt = step(prev, use_min)
                nd_empty = nxt is None
                if nd_empty:
                    break
                if mean_move_distance(prev, nxt) < p.threshold_dist:
                    nxt["uniq_ind"] = uniq_ind
                    uniq_ind += 1
                    found.append(nxt)
                    break
                prev = nxt
            if nd_empty:
                break
    if not found:
        return empty_points(demand_gdf.crs)
    result = gpd.GeoDataFrame(pd.concat(found), geometry="geometry", crs=demand_gdf.crs)
    values = evaluate_solutions(demand_gdf, result, p.weight_col, p.coverage_dist)
    keep = non_dominated_solution_ids(values, "sum_expected_va", "weighted_distance", strict=False)
    return result[result["uniq_ind"].isin(keep)]


def voronoi_sample(gdf_filtered: gpd.GeoDataFrame, hull: gpd.GeoDataFrame, n_points: int) -> gpd.GeoDataFrame:
    sel = gdf_filtered.sample(n_points) if len(gdf_filtered) >= n_points else gdf_filtered
    voro = voronoi_diagram(MultiPoint(list(zip(sel.geometry.x, sel.geometry.y))))
    cells = gpd.GeoDataFrame(geometry=list(voro.geoms), crs=gdf_filtered.crs)
    return gpd.overlay(cells, hull, how="intersection")


def simulate_demands(gdf_filtered: gpd.GeoDataFrame, hull: gpd.GeoDataFrame, p: Params) -> gpd.GeoDataFrame:
    """One Monte Carlo draw: a representative demand inside each Voronoi region,
    drawn from the local distribution of coordinates and weights."""
    cells = voronoi_sample(gdf_filtered, hull, p.num_fac * p.demands_per_facility)
    gx, gy = gdf_filtered.geometry.x.to_numpy(), gdf_filtered.geometry.y.to_numpy()
    w = gdf_filtered[p.weight_col].to_numpy(dtype=float)
    pts, wts = [], []
    for poly in cells.geometry:
        inside = shapely.contains_xy(poly, gx, gy)       # same as GeoSeries.within(poly)
        if not inside.any():
            continue
        sx, sy = np.std(gx[inside]), np.std(gy[inside])
        new = Point(np.random.normal(gx[inside].mean(), sx if sx > 0 else 0.1),
                    np.random.normal(gy[inside].mean(), sy if sy > 0 else 0.1))
        if not poly.contains(new):
            new = nearest_points(poly, new)[0]
        sw = np.std(w[inside])
        pts.append(new)
        wts.append(np.random.normal(w[inside].mean(), sw if sw > 0 else 0.1))
    return gpd.GeoDataFrame({p.weight_col: wts}, geometry=pts, crs=gdf_filtered.crs)


@dataclass
class MonteCarloResult:
    solutions: gpd.GeoDataFrame        # all local non-dominated solutions (uniq_ind)
    simulated_values: pd.DataFrame     # objective values on each simulated demand set
    scenario_values: pd.DataFrame      # objective values on the scenario demands (Da)
    global_values: pd.DataFrame        # global non-dominated values on Da
    global_solutions: gpd.GeoDataFrame # global non-dominated facility locations


def run_monte_carlo(gdf_filtered: gpd.GeoDataFrame, hull: gpd.GeoDataFrame, p: Params,
                    progress=None) -> MonteCarloResult:
    """'3. Analyze': Monte Carlo simulation + global non-dominated solutions on Da."""
    if p.seed is not None:
        np.random.seed(p.seed)
    sols, sim_vals = [], []
    offset = 0
    for i in range(p.num_vor):
        sim = simulate_demands(gdf_filtered, hull, p)
        if progress:
            progress(i + 1, p.num_vor)
        if sim.empty or len(sim) < p.num_fac:
            continue
        res = optimize(sim, p)
        if res.empty:
            continue
        res = res.copy()
        res["uniq_ind"] = pd.factorize(res["uniq_ind"])[0] + offset
        offset = int(res["uniq_ind"].max()) + 1
        sols.append(res)
        sim_vals.append(evaluate_solutions(sim, res, p.weight_col, p.coverage_dist))
    if not sols:
        e = pd.DataFrame(columns=["uniq_ind", "sum_expected_va", "weighted_distance", "covered_per", "mean_dist"])
        return MonteCarloResult(empty_points(gdf_filtered.crs), e, e, e, empty_points(gdf_filtered.crs))
    solutions = gpd.GeoDataFrame(pd.concat(sols, ignore_index=True), geometry="geometry",
                                 crs=gdf_filtered.crs)
    simulated = pd.concat(sim_vals, ignore_index=True)
    scenario = evaluate_solutions(gdf_filtered, solutions, p.weight_col, p.coverage_dist)
    keep = non_dominated_solution_ids(scenario, "covered_per", "mean_dist", strict=True)
    glob_vals = scenario[scenario["uniq_ind"].isin(keep)]
    glob_sols = solutions[solutions["uniq_ind"].isin(keep)]
    return MonteCarloResult(solutions, simulated, scenario, glob_vals, glob_sols)


def solutions_for_export(sol: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Facility locations of the global non-dominated solutions, ready for a shapefile."""
    out = sol.copy()
    rename = {"sum_weight": "cov_weight", "sum_distance": "sum_dist",
              "sum_weighted_distance": "sum_wdist", "nearest_id": "fac_id", "uniq_ind": "solution"}
    return out.rename(columns=rename)
