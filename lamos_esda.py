"""
LAMOS exploratory spatial data analysis (ESDA).

The six ESDA methods listed in Table 1 of the paper, each split into a
computation function (no GUI dependency) and a ``draw_*`` function that
renders onto a ``matplotlib.figure.Figure``. The GUI embeds these figures in
pop-up windows; ``scripts/reproduce_esda_figures.py`` saves them to files.

D0 is the full demand set and Da the scenario subset selected with the
temporal/attribute filters.

1. Weighted quadrat analysis ........ quadrat_shares, quadrat_curve, draw_quadrat
2. Demand surface difference map .... surface_difference, draw_surface_difference
3. LISA (local Moran's I) ........... lisa, draw_lisa
4. Kernel density estimation ........ draw_kde_grid
5. K-S tests and Ripley's K ......... draw_ks_ripley
6. K-means clustering ............... kmeans_fit, draw_kmeans
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.figure import Figure
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp

D0_LABEL = r"$D_0$"
DA_LABEL = r"$D_a$"


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def study_extent(d0: gpd.GeoDataFrame, bg_gdf: gpd.GeoDataFrame | None = None, pad: float = 0.0):
    """Common map extent (xmin, ymin, xmax, ymax) used by every ESDA panel."""
    b = np.array(d0.total_bounds, dtype=float)
    if bg_gdf is not None and len(bg_gdf):
        g = np.array(bg_gdf.total_bounds, dtype=float)
        b = np.array([min(b[0], g[0]), min(b[1], g[1]), max(b[2], g[2]), max(b[3], g[3])])
    return (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)


def _frame(ax, extent, bg_gdf=None, title=None):
    if bg_gdf is not None and len(bg_gdf):
        bg_gdf.boundary.plot(ax=ax, color="black", linewidth=1)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)


def _coords(gdf):
    return gdf.geometry.x.to_numpy(dtype=float), gdf.geometry.y.to_numpy(dtype=float)


# --------------------------------------------------------------------------
# 1. weighted quadrat analysis
# --------------------------------------------------------------------------
def quadrat_grid(extent, n, bg_gdf=None):
    """Square quadrats: ``n`` cells across the x-range of the study area and as
    many rows as needed to cover its y-range. Only quadrats that intersect the
    study area are kept (all of them if no study area is given)."""
    import shapely

    xmin, ymin, xmax, ymax = extent
    size = (xmax - xmin) / n
    rows = max(1, int(np.ceil((ymax - ymin) / size - 1e-9)))
    ex = xmin + size * np.arange(n + 1)
    ey = ymin + size * np.arange(rows + 1)
    if bg_gdf is not None and len(bg_gdf):
        area = bg_gdf.geometry.union_all() if hasattr(bg_gdf.geometry, "union_all") else bg_gdf.unary_union
        cells = shapely.box(ex[None, :-1], ey[:-1, None], ex[None, 1:], ey[1:, None])
        keep = shapely.intersects(cells, area)
    else:
        keep = np.ones((rows, n), dtype=bool)
    return ex, ey, keep


def quadrat_shares(gdf, weight_col, ex, ey, keep):
    """Each quadrat's share (%) of the weighted demand contained in the kept quadrats."""
    x, y = _coords(gdf)
    w = gdf[weight_col].to_numpy(dtype=float)
    ix = np.clip(np.searchsorted(ex, x, side="right") - 1, 0, len(ex) - 2)
    iy = np.clip(np.searchsorted(ey, y, side="right") - 1, 0, len(ey) - 2)
    h = np.zeros(keep.shape)
    np.add.at(h, (iy, ix), w)
    h[~keep] = 0
    total = h.sum()
    return 100 * h / total if total > 0 else h


def quadrat_differences(d0, da, weight_col, extent, n, bg_gdf=None):
    """Relative absolute difference |Da - D0| / D0 (%) of every quadrat with D0 > 0."""
    ex, ey, keep = quadrat_grid(extent, n, bg_gdf)
    s0 = quadrat_shares(d0, weight_col, ex, ey, keep)
    sa = quadrat_shares(da, weight_col, ex, ey, keep)
    m = keep & (s0 > 0)
    return 100 * np.abs(sa[m] - s0[m]) / s0[m]


def quadrat_summary(d0, da, weight_col, extent, bg_gdf=None, n_range=range(1, 31)):
    """Per-resolution distributions of the quadrat differences (for the box plots)."""
    n = list(n_range)
    return n, [quadrat_differences(d0, da, weight_col, extent, k, bg_gdf) for k in n]


def draw_quadrat_summary(ax, summary, highlight=()):
    n, diffs = summary
    bp = ax.boxplot(diffs, positions=n, widths=0.6, patch_artist=True, showfliers=True,
                    medianprops=dict(color="darkorange"), flierprops=dict(markersize=3))
    for patch in bp["boxes"]:
        patch.set_facecolor("tab:blue")
    for k in highlight:
        ax.axvspan(k - 0.45, k + 0.45, color="red", alpha=0.15)
    ax.set_ylim(0, 200)
    ax.set_xlabel("Quadrat resolution (number of quadrats along the x-axis)")
    ax.set_ylabel(f"|{DA_LABEL} - {D0_LABEL}| / {D0_LABEL} (%)")
    ax.set_title("Relative absolute difference of quadrat demand shares by resolution")
    ax.grid(True, alpha=0.3)


def draw_quadrat_maps(fig: Figure, d0, da, weight_col, extent, n, bg_gdf=None, axes=None):
    """D0 | Da quadrat maps of the share of contained demand, on a shared colour scale."""
    ex, ey, keep = quadrat_grid(extent, n, bg_gdf)
    s0 = quadrat_shares(d0, weight_col, ex, ey, keep)
    sa = quadrat_shares(da, weight_col, ex, ey, keep)
    vmax = max(s0.max(), sa.max())
    if axes is None:
        axes = fig.subplots(1, 2)
    mesh = None
    for ax, s, lab, g in [(axes[0], s0, D0_LABEL, d0), (axes[1], sa, DA_LABEL, da)]:
        mesh = ax.pcolormesh(ex, ey, np.ma.masked_where(~keep, s), cmap="Reds", vmin=0, vmax=vmax,
                             edgecolors="black", linewidth=0.4)
        lim = (ex[0], ey[0], ex[-1], max(ey[-1], extent[3]))
        _frame(ax, lim, bg_gdf, f"{lab}, resolution {n}")
        ax.tick_params(labelsize=7)
    fig.colorbar(mesh, ax=list(axes), shrink=0.85, label="Proportion of contained demand (%)")
    return axes


def draw_quadrat(fig: Figure, d0, da, weight_col, extent, resolutions, bg_gdf=None, summary=None):
    """Figure 6 layout: (a) box plots for resolutions 1-30, then one D0 | Da map row
    per resolution in ``resolutions``."""
    resolutions = list(resolutions)
    if summary is None:
        summary = quadrat_summary(d0, da, weight_col, extent, bg_gdf)
    gs = gridspec.GridSpec(1 + len(resolutions), 2, figure=fig, height_ratios=[1] + [1.1] * len(resolutions))
    draw_quadrat_summary(fig.add_subplot(gs[0, :]), summary, highlight=resolutions)
    fig.axes[0].set_title("(a) " + fig.axes[0].get_title())
    for r, k in enumerate(resolutions, start=1):
        axes = [fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1])]
        draw_quadrat_maps(fig, d0, da, weight_col, extent, k, bg_gdf, axes=axes)
        axes[0].set_title(f"({'bcdefg'[r-1]}) " + axes[0].get_title())
    return summary


# --------------------------------------------------------------------------
# 2. approximated demand surface difference map (ordinary kriging)
# --------------------------------------------------------------------------
def kriged_surface(gdf, weight_col, gx, gy, max_points=800, variogram="gaussian", rng=None):
    """Ordinary-kriging surface of the demand weight on a regular grid.
    Coincident points are averaged; at most ``max_points`` locations are used."""
    from pykrige.ok import OrdinaryKriging

    x, y = _coords(gdf)
    df = pd.DataFrame({"x": x, "y": y, "w": gdf[weight_col].to_numpy(dtype=float)})
    df = df.groupby(["x", "y"], as_index=False)["w"].mean()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=rng)
    ok = OrdinaryKriging(df.x.to_numpy(), df.y.to_numpy(), df.w.to_numpy(),
                         variogram_model=variogram, pseudo_inv=True,
                         enable_plotting=False, verbose=False)
    z, _ = ok.execute("grid", gx, gy)
    return np.asarray(z, dtype=float)


def surface_difference(d0, da, weight_col, extent, bg_gdf=None, grid_n=60, max_points=800, seed=None):
    """Relative difference (Da - D0) / D0 of the kriged demand-weight surfaces,
    clipped to [-1, 1] (positive = higher demand weight in Da)."""
    rng = np.random.RandomState(seed) if seed is not None else None
    gx = np.linspace(extent[0], extent[2], grid_n)
    gy = np.linspace(extent[1], extent[3], grid_n)
    z0 = kriged_surface(d0, weight_col, gx, gy, max_points, rng=rng)
    za = kriged_surface(da, weight_col, gx, gy, max_points, rng=rng)
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.where(np.abs(z0) > 1e-9, (za - z0) / z0, np.nan)
    diff = np.clip(diff, -1, 1)
    if bg_gdf is not None and len(bg_gdf):
        import shapely
        area = bg_gdf.geometry.union_all() if hasattr(bg_gdf.geometry, "union_all") else bg_gdf.unary_union
        XX, YY = np.meshgrid(gx, gy)
        inside = shapely.contains_xy(area, XX, YY) | shapely.intersects_xy(area.boundary, XX, YY)
        diff = np.where(inside, diff, np.nan)
    return gx, gy, diff


def draw_surface_difference(fig: Figure, gx, gy, diff, bg_gdf=None, views=("top", "side")):
    """(a) top-down (planform) view, equivalent to a 2D choropleth of the
    difference, and (b) a 3D side view (rotatable in the GUI)."""
    XX, YY = np.meshgrid(gx, gy)
    finite = np.isfinite(diff)
    lim = float(np.ceil(10 * np.nanmax(np.abs(diff))) / 10) if finite.any() else 1.0
    lim = min(max(lim, 0.1), 1.0)
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    n, axes, mesh = len(views), [], None
    if "top" in views:
        ax = fig.add_subplot(1, n, 1)
        mesh = ax.pcolormesh(gx, gy, np.ma.masked_invalid(diff), cmap="RdBu_r", norm=norm, shading="auto")
        if bg_gdf is not None and len(bg_gdf):
            bg_gdf.boundary.plot(ax=ax, color="black", linewidth=1)
        ax.set_aspect("equal")
        ax.set_title("(a) Top view" if n > 1 else "")
        axes.append(ax)
    if "side" in views:
        ax3 = fig.add_subplot(1, n, n, projection="3d")
        surf = ax3.plot_surface(XX, YY, np.ma.masked_invalid(diff), cmap="RdBu_r", norm=norm,
                                rstride=1, cstride=1, linewidth=0, antialiased=False)
        if bg_gdf is not None and len(bg_gdf):
            for geom in bg_gdf.geometry:
                for ring in getattr(geom, "geoms", [geom]):
                    bx, by = ring.exterior.xy
                    ax3.plot(bx, by, zs=-lim, color="black", linewidth=1)
        ax3.set_zlim(-lim, lim)
        ax3.view_init(elev=28, azim=-60)
        ax3.set_title("(b) Side view" if n > 1 else "")
        ax3.set_zlabel(f"({DA_LABEL} - {D0_LABEL}) / {D0_LABEL}")
        mesh = surf if mesh is None else mesh
        axes.append(ax3)
    fig.colorbar(mesh, ax=axes, shrink=0.7,
                 label=f"({DA_LABEL} - {D0_LABEL}) / {D0_LABEL} of kriged demand weight")
    if n == 1:
        return
    fig.suptitle("Approximated demand surface difference map: red = higher in "
                 f"{DA_LABEL}, blue = higher in {D0_LABEL}")


# --------------------------------------------------------------------------
# 3. LISA
# --------------------------------------------------------------------------
LISA_COLORS = {"Hotspot": "red", "Coldspot": "blue", "Spatial outlier": "#bdbdbd",
               "Not significant": "white"}


def _unique_coords(gdf, eps=1e-4):
    """Coordinates with coincident points nudged apart (by eps) so that every
    demand gets its own Voronoi cell."""
    xy = np.column_stack(_coords(gdf))
    _, first, inverse = np.unique(xy, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    rank = pd.Series(np.arange(len(xy))).groupby(inverse).cumcount().to_numpy()
    xy = xy.copy()
    xy[:, 0] += eps * rank
    return xy


def voronoi_weights(xy):
    """Voronoi-contiguity (Delaunay neighbour) spatial weights, row-standardised."""
    from scipy.spatial import Delaunay
    from libpysal.weights import W

    indptr, indices = Delaunay(xy).vertex_neighbor_vertices
    neighbors = {i: indices[indptr[i]:indptr[i + 1]].tolist() for i in range(len(xy))}
    w = W(neighbors, silence_warnings=True)
    w.transform = "r"
    return w


def voronoi_cells(xy, area):
    """Voronoi cell of every point (input order), clipped to ``area``."""
    import shapely
    from shapely.geometry import MultiPoint

    mp = MultiPoint(xy)
    try:
        cells = np.array(shapely.voronoi_polygons(mp, extend_to=area.envelope.buffer(10), ordered=True).geoms)
    except TypeError:   # shapely < 2.1 has no 'ordered': match cells to points spatially
        polys = gpd.GeoDataFrame(geometry=list(shapely.voronoi_polygons(mp, extend_to=area.envelope.buffer(10)).geoms))
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]))
        j = gpd.sjoin(pts, polys, predicate="within", how="left")
        j = j[~j.index.duplicated()]
        cells = polys.geometry.to_numpy()[j["index_right"].to_numpy().astype(int)]
    return shapely.intersection(cells, area)


def lisa(gdf, weight_col, permutations=999, alpha=0.05, seed=None):
    """Global and local Moran's I of the demand weight with Voronoi-contiguity
    weights. Returns (global I, cluster label per demand)."""
    import esda

    xy = _unique_coords(gdf)
    w = voronoi_weights(xy)
    y = gdf[weight_col].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glob = esda.Moran(y, w, permutations=0)
        loc = esda.Moran_Local(y, w, permutations=permutations, seed=seed)
    sig = loc.p_sim < alpha
    labels = np.full(len(y), "Not significant", dtype=object)
    labels[sig & (loc.q == 1)] = "Hotspot"
    labels[sig & (loc.q == 3)] = "Coldspot"
    labels[sig & np.isin(loc.q, [2, 4])] = "Spatial outlier"
    return float(glob.I), pd.Series(labels, index=gdf.index, name="lisa")


def draw_lisa_panel(ax, g, result, name, extent, area, bg_gdf=None, title=None):
    moran_i, lab = result
    cells = voronoi_cells(_unique_coords(g), area)
    gpd.GeoSeries(cells).plot(ax=ax, color=[LISA_COLORS[v] for v in lab], edgecolor="grey", linewidth=0.15)
    _frame(ax, extent, bg_gdf, title or f"LISA from {name}")
    ax.text(0.98, 0.97, f"Moran's I: {moran_i:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=11)
    handles = [mpatches.Patch(facecolor=LISA_COLORS[c], edgecolor="grey",
                              label=f"{c} ({int((lab == c).sum()):,})") for c in LISA_COLORS]
    ax.legend(handles=handles, loc="upper left", fontsize=7)


def _area(bg_gdf, hull):
    src = bg_gdf if (bg_gdf is not None and len(bg_gdf)) else hull
    return src.geometry.union_all() if hasattr(src.geometry, "union_all") else src.unary_union


def draw_lisa(fig: Figure, d0, res0, da, resa, extent, bg_gdf=None, hull=None):
    """Hotspots (high values surrounded by high values) and coldspots of the
    demand weight, drawn on the Voronoi cells of the demands."""
    area = _area(bg_gdf, hull)
    for i, (g, res, name) in enumerate([(d0, res0, D0_LABEL), (da, resa, DA_LABEL)], start=1):
        draw_lisa_panel(fig.add_subplot(1, 2, i), g, res, name, extent, area, bg_gdf,
                        title=f"({'ab'[i-1]}) LISA from {name}")
    fig.suptitle("Local Moran's I of the demand weight (Voronoi contiguity, pseudo p < 0.05)")


# --------------------------------------------------------------------------
# 4. kernel density estimation (common extent and aspect ratio)
# --------------------------------------------------------------------------
def kde_figsize(n_rows, n_cols, extent, panel_width=3.0):
    """Figure size that fits n_rows x n_cols equal-aspect KDE panels without gaps."""
    ratio = (extent[3] - extent[1]) / (extent[2] - extent[0])
    return (panel_width * n_cols + 0.8, (panel_width * ratio + 0.45) * n_rows + 0.8)


def draw_kde_grid(fig: Figure, gdf, temcols, value1, extent, bg_gdf=None):
    """One figure per value of the first temporal variable; rows/columns are the
    values of the second/third temporal variables. Every panel uses the same
    spatial extent and an equal aspect ratio so panels are directly comparable."""
    c1, c2, c3 = (list(temcols) + [None, None, None])[:3]
    v2 = sorted(gdf[c2].dropna().unique()) if c2 else [None]
    v3 = sorted(gdf[c3].dropna().unique()) if c3 else [None]
    axes = fig.subplots(len(v2), len(v3), squeeze=False, sharex=True, sharey=True)
    base = gdf if (c1 is None or value1 is None) else gdf[gdf[c1] == value1]
    for r, a in enumerate(v2):
        for c, b in enumerate(v3):
            ax = axes[r, c]
            sub = base
            if c2:
                sub = sub[sub[c2] == a]
            if c3:
                sub = sub[sub[c3] == b]
            parts = [f"{c2}={a:g}" if isinstance(a, (int, float, np.number)) else f"{c2}={a}"] if c2 else []
            if c3:
                parts.append(f"{c3}={b:g}" if isinstance(b, (int, float, np.number)) else f"{c3}={b}")
            x, y = _coords(sub)
            if len(sub) >= 3 and np.std(x) > 0 and np.std(y) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sns.kdeplot(x=x, y=y, ax=ax, cmap="viridis", fill=True,
                                clip=((extent[0], extent[2]), (extent[1], extent[3])))
                ax.scatter(x, y, color="cornflowerblue", s=4)
            else:
                ax.text(0.5, 0.5, f"{len(sub)} points\n(too few for KDE)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=8)
            _frame(ax, extent, bg_gdf, ", ".join(parts))
            ax.title.set_fontsize(8)
            ax.tick_params(labelsize=6)
    for ax in axes.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.supxlabel("x", fontsize=9)
    fig.supylabel("y", fontsize=9)
    if c1 is not None and value1 is not None:
        fig.suptitle(f"{c1}={value1:g}" if isinstance(value1, (int, float, np.number)) else f"{c1}={value1}")
    return axes


# --------------------------------------------------------------------------
# 5. K-S tests, Ripley's K and Voronoi overlap
# --------------------------------------------------------------------------
def _ecdf(ax, values, label):
    v = np.sort(np.asarray(values, dtype=float))
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=label)


def _ci(a, confidence=0.95):
    a = a[~np.isnan(a)]
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, len(a) - 1)
    return m - h, m + h


def draw_ks_ripley(fig: Figure, d0, da, weight_col, bg_gdf, hull, num_points=60,
                   iterations=100, voronoi_iterations=30, seed=None, voronoi=True):
    """K-S tests (x, y, weight) between Da and the unselected demands D0 \\ Da
    (independent samples), Monte Carlo Ripley's K, and Voronoi overlap."""
    import shapely
    from astropy.stats import RipleysKEstimator
    from shapely.geometry import MultiPoint
    from shapely.ops import voronoi_diagram

    if seed is not None:
        np.random.seed(seed)
    rest = d0[~d0.index.isin(da.index)]
    gs = gridspec.GridSpec(6 if voronoi else 4, 2, figure=fig)
    parts = "Ripley's K and Voronoi overlap" if voronoi else "Ripley's K"
    fig.suptitle(f"Kolmogorov-Smirnov tests, {parts}: {DA_LABEL} vs unselected demands")

    series = [("X coordinates", rest.geometry.x, da.geometry.x),
              ("Y coordinates", rest.geometry.y, da.geometry.y),
              (weight_col, rest[weight_col], da[weight_col])]
    for i, (title, u, f) in enumerate(series):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        if len(u):
            ks, p = ks_2samp(u, f)
            _ecdf(ax, u, "Unselected demands")
            ax.text(0.65, 0.3, f"KS statistic: {ks:.4f}\nP-value: {p:.4e}", transform=ax.transAxes)
        _ecdf(ax, f, DA_LABEL)
        ax.set_xlabel("Value")
        ax.set_ylabel("ECDF")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Ripley's K (Monte Carlo samples of num_points demands)
    x_min, y_min, x_max, y_max = bg_gdf.total_bounds
    area = float(bg_gdf.geometry.area.sum())
    r = np.linspace(0, (area / 2) ** 0.5, 100)
    kest = RipleysKEstimator(area=area, x_max=x_max, y_max=y_max, x_min=x_min, y_min=y_min)

    def sample(g, n):
        return g.sample(n) if len(g) >= n else g.sample(n, replace=True)

    def karr(g):
        return kest(data=np.column_stack(_coords(g)), radii=r, mode="ripley")

    area_geom = bg_gdf.geometry.union_all() if hasattr(bg_gdf.geometry, "union_all") else bg_gdf.unary_union

    def csr_points(n):
        """n uniformly random points inside the study area (rejection sampling)."""
        pts = np.empty((0, 2))
        while len(pts) < n:
            cand = np.column_stack([np.random.uniform(x_min, x_max, 2 * n), np.random.uniform(y_min, y_max, 2 * n)])
            pts = np.vstack([pts, cand[shapely.contains_xy(area_geom, cand[:, 0], cand[:, 1])]])
        return pts[:n]

    groups = {f"{DA_LABEL}": [], "Unselected": [], "Combined": [], "CSR": []}
    for _ in range(iterations):
        m = sample(da, num_points)
        u = sample(rest, num_points) if len(rest) else m
        c = pd.concat([m.sample(len(m) // 2), u.sample(len(u) // 2)])
        groups[f"{DA_LABEL}"].append(karr(m))
        groups["Unselected"].append(karr(u))
        groups["Combined"].append(karr(c))
        groups["CSR"].append(kest(data=csr_points(num_points), radii=r, mode="ripley"))
    colors = {f"{DA_LABEL}": "cornflowerblue", "Unselected": "indianred", "Combined": "yellowgreen",
              "CSR": "black"}

    ax3 = fig.add_subplot(gs[1, 1])
    box = pd.concat([pd.DataFrame({"Group": k, "K": np.concatenate(v)}) for k, v in groups.items()])
    sns.boxplot(data=box, x="Group", y="K", hue="Group", palette=colors, legend=False, ax=ax3)
    ax3.set_title("Monte Carlo simulated Ripley's K")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2:4, :])
    for k, v in groups.items():
        arr = np.array(v)
        lo, hi = np.empty(arr.shape[1]), np.empty(arr.shape[1])
        for j in range(arr.shape[1]):
            lo[j], hi[j] = _ci(arr[:, j])
        label = ("Expected random pattern (CSR simulated in study area)" if k == "CSR" else f"{k} demands")
        ax4.plot(r, np.nanmean(arr, axis=0), color=colors[k], label=label)
        ax4.fill_between(r, lo, hi, color=colors[k], alpha=0.2)
    ax4.set_xlabel("Distance")
    ax4.set_ylabel("Ripley's K")
    ax4.set_title(f"Mean Ripley's K with 95% confidence intervals ({iterations} Monte Carlo samples of {num_points})")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    if not voronoi:
        return
    # Voronoi overlap: how much of each region of a Da sample overlaps the region
    # of the nearest generator of an unselected sample (and, as a reference,
    # of two random samples drawn from D0)
    ax5 = fig.add_subplot(gs[4:6, :])
    hull_geom = hull.geometry.iloc[0]
    hull_area = hull_geom.area

    def overlap(ga, gb):
        a, b = sample(ga, num_points), sample(gb, num_points)
        ca, cb = np.column_stack(_coords(a)), np.column_stack(_coords(b))
        def regions_by_generator(pts):
            regs = np.array(list(voronoi_diagram(MultiPoint(pts)).geoms), dtype=object)
            regs = shapely.intersection(regs, hull_geom)
            hit = shapely.covers(regs[:, None], shapely.points(pts)[None, :])   # region x generator
            out = {}
            for r_i, g_i in zip(*np.nonzero(hit)):
                out.setdefault(int(g_i), regs[r_i])
            return out
        ra, rb = regions_by_generator(ca), regions_by_generator(cb)
        nearest = np.argmin(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1), axis=1)
        parts = [ra[i].intersection(rb[j]) for i, j in enumerate(nearest) if i in ra and j in rb]
        return [p for p in parts if not p.is_empty]

    sel, ref = [], []
    for _ in range(voronoi_iterations):
        if len(rest):
            sel.extend(overlap(da, rest))
        ref.extend(overlap(d0, d0))
    if sel and ref:
        a_sel = np.array([p.area for p in sel])
        a_ref = np.array([p.area for p in ref])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ad = stats.anderson_ksamp([a_sel, a_ref], variant="midrank")
            except TypeError:
                ad = stats.anderson_ksamp([a_sel, a_ref])
        pval = getattr(ad, "pvalue", getattr(ad, "significance_level", np.nan))
        pct_sel = 100 * a_sel.sum() / (voronoi_iterations * hull_area)
        pct_ref = 100 * a_ref.sum() / (voronoi_iterations * hull_area)
        hull.plot(ax=ax5, facecolor="cornflowerblue", edgecolor="black", alpha=0.7)
        gpd.GeoSeries(sel[: num_points * 10], crs=d0.crs).plot(ax=ax5, facecolor="indianred",
                                                                edgecolor="none", alpha=0.05)
        ax5.text(0.02, 0.97, f"Overlap, {DA_LABEL} vs unselected: {pct_sel:.1f}% of area\n"
                             f"Overlap, two {D0_LABEL} samples (reference): {pct_ref:.1f}% of area\n"
                             f"Anderson-Darling statistic: {ad.statistic:.3f} (p = {pval:.3f})",
                 va="top", transform=ax5.transAxes, fontsize=9)
        ax5.legend(handles=[mpatches.Patch(color="indianred", label=f"Overlap of {DA_LABEL} regions"),
                            mpatches.Patch(color="cornflowerblue", label="Study area (convex hull)")],
                   loc="lower right")
    else:
        ax5.text(0.5, 0.5, "Not enough data for the Voronoi overlap", ha="center", transform=ax5.transAxes)
    ax5.set_title(f"Voronoi overlap ({voronoi_iterations} Monte Carlo samples)")
    ax5.set_aspect("equal")


# --------------------------------------------------------------------------
# 6. k-means clustering
# --------------------------------------------------------------------------
def kmeans_fit(gdf, weight_col, k, seed=None, n_init=10):
    """Demand-weighted k-means (k-means++ initialisation, ``n_init`` restarts)."""
    from sklearn.cluster import KMeans

    x, y = _coords(gdf)
    X = np.column_stack([x, y])
    w = np.clip(gdf[weight_col].to_numpy(dtype=float), 0, None)
    km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=seed)
    km.fit(X, sample_weight=w if w.sum() > 0 else None)
    return km.predict(X), km.cluster_centers_


def draw_kmeans(fig: Figure, d0, da, weight_col, k, extent, bg_gdf=None, seed=None):
    """K-means computed independently for D0 and Da with the same k."""
    cmap = sns.color_palette("tab10", max(k, 3))
    for i, (g, name) in enumerate([(d0, D0_LABEL), (da, DA_LABEL)], start=1):
        ax = fig.add_subplot(1, 2, i)
        labels, centres = kmeans_fit(g, weight_col, k, seed)
        x, y = _coords(g)
        for c in range(k):
            m = labels == c
            ax.scatter(x[m], y[m], s=5, color=cmap[c % len(cmap)], label=f"Cluster {c}")
        ax.scatter(centres[:, 0], centres[:, 1], marker="x", color="red", s=120, linewidths=2.5,
                   label="Cluster centre")
        _frame(ax, extent, bg_gdf, name)
        leg = ax.legend(loc="lower right", fontsize=7)
        for h in leg.legend_handles[:-1]:
            h.set_sizes([30])
        leg.legend_handles[-1].set_sizes([60])
    fig.suptitle(f"K-means clustering with {k} facilities (demand-weighted, k-means++, 10 restarts)")


# --------------------------------------------------------------------------
# '3. Analyze' result window (Figure 12)
# --------------------------------------------------------------------------
def draw_results(fig: Figure, res, gdf_filtered, bg_gdf, params):
    """Non-dominated facility locations and trade-off graphs of a Monte Carlo run
    (``lamos_core.MonteCarloResult``)."""
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0, :])
    sol = res.solutions
    if len(sol) > 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.kdeplot(x=sol.geometry.x, y=sol.geometry.y, ax=ax0, cmap="viridis", fill=True, alpha=0.5)
    if bg_gdf is not None and len(bg_gdf):
        bg_gdf.plot(ax=ax0, facecolor="none", edgecolor="black")
    gdf_filtered.plot(ax=ax0, color="cornflowerblue", edgecolor="black", label="Demand", alpha=0.5)
    if len(sol):
        sol.plot(ax=ax0, color="black", label="Dominated solutions")
    if len(res.global_solutions):
        res.global_solutions.plot(ax=ax0, color="red", marker="*", label="Non-dominated solutions",
                                  edgecolor="black", markersize=200)
    ax0.set_title(f"(a) Spatial distribution of {DA_LABEL} and non-dominated solutions")
    ax0.set_aspect("equal")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[1, 0])
    sv, gv = res.scenario_values, res.global_values
    ax1.plot(sv["covered_per"], sv["mean_dist"], "o", color="black", markersize=8, label="Local solutions")
    ax1.plot(gv["covered_per"], gv["mean_dist"], "*", color="red", markersize=12, label="Non-dominated")
    if len(gv) > 1:
        xr = gv["covered_per"].max() - gv["covered_per"].min()
        yr = gv["mean_dist"].max() - gv["mean_dist"].min()
        ax1.set_xlim(gv["covered_per"].min() - xr, gv["covered_per"].max() + xr / 4)
        ax1.set_ylim(gv["mean_dist"].min() - yr / 4, gv["mean_dist"].max() + yr)
    ax1.set_xlabel("Coverage (%)")
    ax1.set_ylabel("Mean distance")
    ax1.set_title("(b) Trade-off solutions")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
    sim = res.simulated_values
    if len(sim) > 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.kdeplot(data=sim, x="covered_per", y="mean_dist", ax=ax2, fill=True, alpha=0.5)
    ax2.scatter(sim["covered_per"], sim["mean_dist"], color="black", s=12)
    ax2.scatter(gv["covered_per"], gv["mean_dist"], marker="*", color="red", edgecolor="black", s=200)
    ax2.set_xlabel("Coverage (%)")
    ax2.set_ylabel("Mean distance")
    n = params.num_fac * params.demands_per_facility
    ax2.set_title(f"(c) Simulated trade-off solutions with {n} demands")
    ax2.grid(True, alpha=0.3)
