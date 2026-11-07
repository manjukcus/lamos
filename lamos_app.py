"""
LAMOS: Location Allocation Model Optimization System (visual analytics GUI).

Run:  python lamos_app.py

Workflow (see README.md):
  0. Area                 - load the study-area polygon shapefile
  1. Input                - load the demand point shapefile (D0)
  2. Parameter Setting    - read the model / computation parameters
  Temporal / Attribute    - define the scenario subset (Da), then 'Visualize'
  ESDA                    - open the six exploratory analyses (Table 1 of the paper)
  3. Analyze              - Monte Carlo multi-objective optimization on Da
  4. Save                 - export the non-dominated facility locations
"""
from __future__ import annotations

import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402

import lamos_core as core  # noqa: E402
import lamos_esda as esda  # noqa: E402

FONT = ("Times New Roman", 14)
TABLE_ROWS = 1000          # rows shown in the attribute table
DEFAULTS = {               # parameter values used in the paper's case study (Figure 5)
    "Weight column": "", "Number of facilities": "3", "Coverage distance": "90",
    "Number of partitions": "30", "Threshold distance": "5", "Maximum iteration": "10",
    "Number of iteration": "5", "Voronoi of iteration": "100", "Random seed": "42",
}


class ToolTip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + self.widget.winfo_height()
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1,
                 justify="left").pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def figure_window(parent, title, fig, size="1300x850"):
    """Pop-up window with a scrollable matplotlib figure and toolbar."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry(size)
    canvas = FigureCanvasTkAgg(fig, master=win)
    NavigationToolbar2Tk(canvas, win).update()
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return win, canvas


class LamosApp:
    def __init__(self, root):
        self.root = root
        root.title("LAMOS - Visual Analytics Tool")
        self.bg = None           # study area
        self.d0 = None           # all demands
        self.da = None           # scenario subset
        self.hull = None
        self.params = None
        self.result = None
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        r = self.root
        for i in range(20):
            r.grid_rowconfigure(i, weight=2)
        for i in range(9):
            r.grid_columnconfigure(i, weight=2)

        def button(text, cmd, row, col, tip, span=1, **kw):
            b = tk.Button(r, text=text, command=self._safe(cmd), font=FONT, bd=2, relief="solid", **kw)
            b.grid(row=row, column=col, columnspan=span, sticky="nsew")
            ToolTip(b, tip)
            return b

        button("0. Area", self.select_area, 0, 0, "Input the research area (polygon shapefile)")
        button("1. Input", self.select_demand, 0, 1, "The point shapefile you want to consider", span=2)
        button("2. Parameter Setting", self.read_parameters, 0, 3,
               "Weight column: the demand's weight\nNumber of facilities: the number of desired facilities\n"
               "Coverage distance: the distance that a facility can cover\n"
               "Number of partitions: how many non-dominated solutions will be considered\n"
               "Threshold distance: the algorithm stops when solutions move less than this value\n"
               "Maximum iteration: the algorithm stops after this many iterations\n"
               "Number of iteration: random restarts of the optimization\n"
               "Voronoi of iteration: number of Monte Carlo simulations\n"
               "Random seed: makes results reproducible (leave empty for random)")

        self.entries = {}
        layout = [("Weight column", 1, 0), ("Number of facilities", 2, 0), ("Coverage distance", 3, 0),
                  ("Number of partitions", 4, 0), ("Threshold distance", 1, 2), ("Maximum iteration", 2, 2),
                  ("Number of iteration", 3, 2), ("Voronoi of iteration", 4, 2), ("Random seed", 5, 0)]
        for name, row, col in layout:
            tk.Label(r, text=f"{name}: ", font=FONT).grid(row=row, column=col, sticky="nsew")
            e = tk.Entry(r, font=FONT)
            e.insert(0, DEFAULTS[name])
            e.grid(row=row, column=col + 1, sticky="nsew")
            self.entries[name] = e
        self.status = tk.Label(r, text="Load the study area (0) and the demand points (1).", font=("Arial", 10),
                               anchor="w", fg="dimgray")
        self.status.grid(row=5, column=2, columnspan=2, sticky="nsew")

        button("ESDA", self.open_esda_menu, 0, 6, "Open the six exploratory spatial data analyses in sequence")
        button("Visualize", self.visualize, 0, 7, "Update the demands following the defined conditions")

        # temporal variables: (button row, from row, to row)
        self.temporal = {}
        for k, (brow, frow, trow) in {1: (1, 2, 3), 2: (4, 5, 6), 3: (7, 8, 9)}.items():
            b = tk.Button(r, text=f"Temporal Variable ({k})", font=FONT, bd=2, relief="solid", width=20,
                          command=self._safe(lambda k=k: self.activate_temporal(k)))
            b.grid(row=brow, column=6, sticky="nsew")
            ToolTip(b, "Type a column name on the right and click to activate it")
            col_entry = tk.Entry(r, font=FONT, width=20)
            col_entry.grid(row=brow, column=7, sticky="nsew")
            scales = []
            for label, row in [("From: ", frow), ("To: ", trow)]:
                tk.Label(r, text=label, font=FONT, width=20).grid(row=row, column=5, sticky="nsew")
                s = tk.Scale(r, from_=0, to=10, orient=tk.HORIZONTAL, font=FONT, bd=2, relief="solid", width=20)
                s.grid(row=row, column=6, columnspan=3, sticky="nsew")
                scales.append(s)
            self.temporal[k] = {"entry": col_entry, "from": scales[0], "to": scales[1], "col": None}

        tk.Label(r, text="Column", font=FONT).grid(row=10, column=6, sticky="nsew")
        tk.Label(r, text="Value", font=FONT).grid(row=10, column=7, sticky="nsew")
        self.attributes = []
        for k, row in [(1, 11), (2, 12)]:
            tk.Label(r, text=f"Attribute Variable ({k}): ", font=FONT).grid(row=row, column=5, sticky="nsew")
            c, v = tk.Entry(r, font=FONT), tk.Entry(r, font=FONT)
            c.grid(row=row, column=6, sticky="nsew")
            v.grid(row=row, column=7, sticky="nsew")
            self.attributes.append((c, v))

        button("3. Analyze", self.analyze, 13, 6, "Run the Monte Carlo multi-objective optimization")
        button("4. Save", self.save, 14, 6, "Save the non-dominated facility locations as a shapefile")

        self.map_fig = Figure(figsize=(8, 3.6))
        self.map_ax = self.map_fig.add_subplot(111)
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=r)
        self.map_canvas.get_tk_widget().grid(row=6, column=0, columnspan=4, rowspan=4, sticky="nsew")

        frame = tk.Frame(r, height=200)
        frame.grid(row=11, column=0, columnspan=4, rowspan=4, sticky="nsew")
        frame.pack_propagate(False)   # keep the wide table from pushing the right-hand panel off-screen
        self.tree = ttk.Treeview(frame, show="headings", selectmode="browse")
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

    # ----------------------------------------------------------------- helpers
    def _safe(self, func):
        """Run a callback with a busy cursor and show errors instead of failing silently."""
        def wrapper(*args):
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            try:
                func(*args)
            except UserWarning as w:
                messagebox.showwarning("LAMOS", str(w))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                messagebox.showerror("LAMOS", f"{type(e).__name__}: {e}")
            finally:
                self.root.config(cursor="")
        return wrapper

    def set_status(self, text):
        self.status.config(text=text)
        self.root.update_idletasks()

    def require(self, demand=True, area=False, params=False):
        if demand and self.d0 is None:
            raise UserWarning("Load the demand point shapefile first (1. Input).")
        if area and self.bg is None:
            raise UserWarning("Load the study-area shapefile first (0. Area).")
        if params and self.params is None:
            self.read_parameters()

    def fill_table(self, gdf):
        self.tree.delete(*self.tree.get_children())
        cols = [c for c in gdf.columns if c != gdf.geometry.name] + [gdf.geometry.name]
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=90, stretch=False)
        for _, row in gdf[cols].head(TABLE_ROWS).iterrows():
            self.tree.insert("", "end", values=[str(v) for v in row])

    def draw_map(self, points=None, color="black", label=None):
        ax = self.map_ax
        ax.clear()
        if self.bg is not None:
            self.bg.plot(ax=ax, color="none", edgecolor="black")
        if points is not None and len(points):
            points.plot(ax=ax, color=color, alpha=0.5, markersize=12)
            ax.text(0.97, 0.95, label or f"Number of points: {len(points):,}", ha="right", va="top",
                    transform=ax.transAxes, color="blue" if color != "black" else "black", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        ax.set_aspect("equal")
        self.map_fig.tight_layout()
        self.map_canvas.draw()

    # ----------------------------------------------------------------- 0, 1, 2
    def select_area(self):
        path = filedialog.askopenfilename(filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")])
        if path:
            self.bg = gpd.read_file(path)
            self.draw_map(self.d0)
            self.set_status(f"Study area loaded: {path.split('/')[-1]}")

    def select_demand(self):
        path = filedialog.askopenfilename(filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")])
        if path:
            d0 = gpd.read_file(path)
            if not (d0.geom_type == "Point").all():
                raise UserWarning("The demand shapefile must contain point geometries.")
            self.d0 = d0.reset_index(drop=True)
            self.da = None
            self.hull = core.convex_hull_area(self.d0)
            self.draw_map(self.d0)
            self.fill_table(self.d0)
            self.set_status(f"{len(self.d0):,} demands loaded. Set the parameters (2).")

    def read_parameters(self):
        self.require()
        e = {k: v.get().strip() for k, v in self.entries.items()}
        weight = e["Weight column"]
        if weight not in self.d0.columns:
            raise UserWarning(f"Weight column '{weight}' is not in the demand data.")
        if not pd.api.types.is_numeric_dtype(self.d0[weight]):
            raise UserWarning(f"Weight column '{weight}' must be numeric.")
        try:
            self.params = core.Params(
                weight_col=weight, num_fac=int(e["Number of facilities"]),
                coverage_dist=float(e["Coverage distance"]), num_partitions=int(e["Number of partitions"]),
                threshold_dist=float(e["Threshold distance"]), max_iter=int(e["Maximum iteration"]),
                iterations=int(e["Number of iteration"]), num_vor=int(e["Voronoi of iteration"]),
                seed=int(e["Random seed"]) if e["Random seed"] else None)
        except ValueError as err:
            raise UserWarning(f"Check the parameter values: {err}") from err
        self.set_status("Parameters set. Define the scenario and click Visualize.")

    # --------------------------------------------------------------- filtering
    def activate_temporal(self, k):
        self.require()
        t = self.temporal[k]
        col = t["entry"].get().strip()
        if not col:
            t["col"] = None
            return
        if col not in self.d0.columns or not pd.api.types.is_numeric_dtype(self.d0[col]):
            raise UserWarning(f"Temporal variable '{col}' must be a numeric column of the demand data.")
        v = self.d0[col].dropna()
        lo, hi = float(v.min()), float(v.max())
        integer = np.allclose(v, np.round(v))
        res = 1 if integer else (hi - lo) / 100 or 1
        for s, val in [(t["from"], lo), (t["to"], hi)]:
            s.config(from_=lo, to=hi, resolution=res)
            s.set(val)
        t["col"] = col

    def scenario(self):
        """Apply the temporal ranges and attribute values to D0."""
        self.require()
        g = self.d0
        for t in self.temporal.values():
            if t["col"]:
                g = g[(g[t["col"]] >= t["from"].get()) & (g[t["col"]] <= t["to"].get())]
        for c_entry, v_entry in self.attributes:
            col, val = c_entry.get().strip(), v_entry.get().strip()
            if not col or val == "":
                continue
            if col not in g.columns:
                raise UserWarning(f"Attribute variable '{col}' is not in the demand data.")
            if pd.api.types.is_numeric_dtype(g[col]):
                g = g[g[col] == float(val)]
            else:
                g = g[g[col].astype(str) == val]
        if len(g) == 0:
            raise UserWarning("No demands satisfy the selected conditions.")
        return g

    def visualize(self):
        self.da = self.scenario()
        self.draw_map(self.da, color="cornflowerblue")
        self.set_status(f"Scenario subset Da: {len(self.da):,} of {len(self.d0):,} demands.")

    def _ready(self, area=True):
        """Re-read the parameters and re-apply the scenario so every analysis uses
        the values currently shown in the main window."""
        self.require(area=area)
        self.read_parameters()
        self.visualize()
        return self.d0, self.da, esda.study_extent(self.d0, self.bg)

    # -------------------------------------------------------------------- ESDA
    def open_esda_menu(self):
        self._ready()
        win = tk.Toplevel(self.root)
        win.title("ESDA")
        items = [("1. Weighted quadrat analysis", self.esda_quadrat),
                 ("2. Demand surface difference map", self.esda_surface),
                 ("3. LISA (local Moran's I)", self.esda_lisa),
                 ("4. Kernel density estimation", self.esda_kde),
                 ("5. K-S tests and Ripley's K", self.esda_ks),
                 ("6. K-means clustering", self.esda_kmeans)]
        tk.Label(win, text="Re-open a single view:", font=FONT, anchor="w").pack(fill=tk.X)
        for text, cmd in items:
            tk.Button(win, text=text, font=FONT, width=34, anchor="w", command=self._safe(cmd)).pack(fill=tk.X)
        # one press of 'ESDA' presents all six views as sequential pop-up windows (Section 6-3 of the paper)
        for _, cmd in items:
            self._safe(cmd)()
            self.root.update()

    def esda_quadrat(self):
        d0, da, ext = self._ready()
        w = self.params.weight_col
        self.set_status("Weighted quadrat analysis ...")
        summary = esda.quadrat_summary(d0, da, w, ext, self.bg)
        fig = Figure(figsize=(13, 9), layout="constrained")
        win, canvas = figure_window(self.root, "Quadrat Analysis", fig)

        def redraw(n):
            fig.clear()
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])
            esda.draw_quadrat_summary(fig.add_subplot(gs[0, :]), summary, highlight=[int(n)])
            esda.draw_quadrat_maps(fig, d0, da, w, ext, int(n), self.bg,
                                   axes=[fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])])
            canvas.draw()

        bar = tk.Frame(win)
        bar.pack(side=tk.TOP, fill=tk.X, before=canvas.get_tk_widget())
        tk.Label(bar, text="Grid Resolution", font=FONT).pack(side=tk.LEFT)
        s = tk.Scale(bar, from_=1, to=30, orient=tk.HORIZONTAL, length=400, font=FONT)
        s.pack(side=tk.LEFT)
        s.set(7)
        s.config(command=redraw)
        redraw(7)
        self.set_status("Weighted quadrat analysis done.")

    def esda_surface(self):
        d0, da, ext = self._ready()
        self.set_status("Kriging the demand surfaces ...")
        gx, gy, diff = esda.surface_difference(d0, da, self.params.weight_col, ext, self.bg, seed=self.params.seed)
        fig = Figure(figsize=(15, 6.5), layout="constrained")
        esda.draw_surface_difference(fig, gx, gy, diff, self.bg)
        figure_window(self.root, "Approximated demand surface difference map (drag to rotate)", fig)
        self.set_status("Demand surface difference map done.")

    def esda_lisa(self):
        d0, da, ext = self._ready()
        self.set_status("LISA (999 permutations) ...")
        r0 = esda.lisa(d0, self.params.weight_col, seed=self.params.seed)
        ra = esda.lisa(da, self.params.weight_col, seed=self.params.seed)
        fig = Figure(figsize=(16, 7), layout="constrained")
        esda.draw_lisa(fig, d0, r0, da, ra, ext, self.bg, self.hull)
        figure_window(self.root, "LISA", fig)
        self.set_status("LISA done.")

    def esda_kde(self):
        d0, da, ext = self._ready()
        cols = [t["col"] for t in self.temporal.values() if t["col"]]
        if not cols:
            raise UserWarning("Activate at least one temporal variable for the KDE plots.")
        win = tk.Toplevel(self.root)
        win.title("KDE Analyzer")
        win.geometry("1300x850")
        outer = tk.Canvas(win)
        vsb = ttk.Scrollbar(win, orient="vertical", command=outer.yview)
        hsb = ttk.Scrollbar(win, orient="horizontal", command=outer.xview)
        outer.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(outer)
        outer.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: outer.configure(scrollregion=outer.bbox("all")))
        first = cols[0]
        values = sorted(da[first].dropna().unique()) if len(cols) > 1 else [None]
        for v in values:
            n2 = da[cols[1]].nunique() if len(cols) > 1 else 1
            n3 = da[cols[2]].nunique() if len(cols) > 2 else 1
            fig = Figure(figsize=esda.kde_figsize(n2, n3, ext), layout="constrained")
            if len(cols) > 1:
                esda.draw_kde_grid(fig, da, cols, v, ext, self.bg)
            else:
                esda.draw_kde_grid(fig, da, [None, first], None, ext, self.bg)
            c = FigureCanvasTkAgg(fig, master=inner)
            c.draw()
            c.get_tk_widget().pack()
        self.set_status("KDE plots done (common extent and aspect ratio).")

    def esda_ks(self):
        d0, da, ext = self._ready()
        self.set_status("K-S tests, Ripley's K and Voronoi overlap ...")
        fig = Figure(figsize=(16, 24), layout="constrained")
        esda.draw_ks_ripley(fig, d0, da, self.params.weight_col, self.bg, self.hull,
                            num_points=self.params.num_fac * self.params.demands_per_facility,
                            seed=self.params.seed)
        win = tk.Toplevel(self.root)
        win.title("K-S Test Report")
        win.geometry("1300x850")
        outer = tk.Canvas(win)
        vsb = ttk.Scrollbar(win, orient="vertical", command=outer.yview)
        outer.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(outer)
        outer.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: outer.configure(scrollregion=outer.bbox("all")))
        c = FigureCanvasTkAgg(fig, master=inner)
        c.draw()
        c.get_tk_widget().pack()
        self.set_status("K-S test report done.")

    def esda_kmeans(self):
        d0, da, ext = self._ready()
        fig = Figure(figsize=(15, 6.5), layout="constrained")
        win, canvas = figure_window(self.root, "K-means Clustering Visualization", fig, size="1400x750")
        bar = tk.Frame(win)
        bar.pack(side=tk.TOP, fill=tk.X, before=canvas.get_tk_widget())
        label = tk.Label(bar, font=FONT)

        def redraw(k):
            k = int(k)
            label.config(text=f"Number of Facilities: {k}")
            fig.clear()
            esda.draw_kmeans(fig, d0, da, self.params.weight_col, k, ext, self.bg, seed=self.params.seed)
            canvas.draw()

        label.pack(side=tk.LEFT)
        s = tk.Scale(bar, from_=2, to=10, orient=tk.HORIZONTAL, length=300, font=FONT)
        s.pack(side=tk.LEFT)
        s.set(min(max(self.params.num_fac, 2), 10))
        s.config(command=redraw)
        redraw(s.get())

    # ------------------------------------------------------------ 3. and 4.
    def analyze(self):
        d0, da, _ = self._ready(area=False)
        p = self.params
        if len(da) < p.num_fac:
            raise UserWarning("The scenario subset has fewer demands than facilities.")
        self.result = core.run_monte_carlo(
            da, self.hull, p, progress=lambda i, n: self.set_status(f"Monte Carlo simulation {i}/{n} ..."))
        if self.result.global_solutions.empty:
            raise UserWarning("The analysis produced no solutions; check the parameters.")
        fig = Figure(figsize=(14, 13), layout="constrained")
        esda.draw_results(fig, self.result, da, self.bg, p)
        figure_window(self.root, "Results", fig, size="1200x1000")
        g = self.result.global_values
        self.set_status(f"{len(g)} non-dominated solutions: coverage {g.covered_per.min():.1f}-"
                        f"{g.covered_per.max():.1f}%, mean distance {g.mean_dist.min():.1f}-{g.mean_dist.max():.1f}.")

    def save(self):
        if self.result is None or self.result.global_solutions.empty:
            raise UserWarning("Run the analysis (3. Analyze) first.")
        path = filedialog.asksaveasfilename(defaultextension=".shp", filetypes=[("Shapefile", "*.shp")])
        if path:
            core.solutions_for_export(self.result.global_solutions).to_file(path)
            self.set_status(f"Saved {path.split('/')[-1]}")


def main():
    root = tk.Tk()
    LamosApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
