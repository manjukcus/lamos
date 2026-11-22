"""
Reference copy of the optimization routines of the original LAMOS notebook
(lamos_app.py before this revision), used only by tests/test_equivalence.py to
check that lamos_core.py reproduces the original algorithm exactly.

Changes from the original, kept to the minimum needed to run it:
- string-literal escaping errors introduced by the notebook export were fixed;
- analysis() passes the allocated demand table (alloc_demand_gdf) to
  calculate_sum_per_row(); the original passed demand_gdf, which raised
  KeyError: 'nearest_id' when '3. Analyze' was run.
Module-level parameters (weight_col, num_fac, ...) are set by the test.
"""
# fmt: off
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint, box, Polygon
from shapely.ops import polygonize, unary_union, nearest_points, voronoi_diagram
from scipy import stats
def nanmean_confidence_interval(data, confidence=0.95):
    """
    Calculate the mean and 95% confidence interval of a dataset, ignoring NaN values.
    """
    cleaned_data = data[~np.isnan(data)]  # Remove NaN values
    n = len(cleaned_data)
    m, se = np.nanmean(cleaned_data), stats.sem(cleaned_data, nan_policy='omit')
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def select_bg():
    global bg_gdf
    bg_gdf = filedialog.askopenfilename(filetypes = (("shapefile files","*.shp"),("all files","*.*")))
    if bg_gdf:
        visualize_shapefile1(bg_gdf)

def select_shapefile():
    global demand_gdf, gdf_convexhull, gdf
    demand_gdf_path = filedialog.askopenfilename(filetypes = (("shapefile files","*.shp"),("all files","*.*")))
    if demand_gdf_path:
        demand_gdf = gpd.read_file(demand_gdf_path)
        gdf = demand_gdf # Initialize gdf
        visualize_shapefile2(demand_gdf_path)
        gdf_convexhull = gpd.GeoDataFrame(geometry=[Polygon(gdf.unary_union.convex_hull.buffer(5).exterior)])
        gdf_convexhull.crs = demand_gdf.crs

def add_parameters():
    global num_fac, demand_buffer_distance, maximum_lambda, threshold_dist, max_iter, iteration, weight_col, temcol1, temcol2, temcol3, temp_sel1, temp_sel2, temp_sel3, temp_sel4, temp_sel5, temp_sel6, attrcol1, attrval1, attrcol2, attrval2, num_vor
    weight_col = weight_col_entry.get()
    num_fac = int(num_fac_entry.get())
    demand_buffer_distance = float(demand_buffer_distance_entry.get())
    maximum_lambda = int(maximum_lambda_entry.get())
    threshold_dist = float(threshold_dist_entry.get())
    max_iter = int(max_iter_entry.get())
    iteration = int(iteration_entry.get())
    num_vor = int(vor_entry.get())
    temcol1 = temp1_col_entry.get()
    temcol2 = temp2_col_entry.get()
    temcol3 = temp3_col_entry.get()
    temp_sel1= None
    temp_sel2= None
    temp_sel3= None
    temp_sel4= None
    temp_sel5= None
    temp_sel6= None
    attrcol1 = add1_col_entry.get()
    attrval1 = add1_val_entry.get()
    if attrcol1 and attrval1 and attrcol1 in gdf.columns: # Check if gdf and column exist
        attrval1 = type(gdf[attrcol1].iloc[0])(attrval1)
    attrcol2 = add2_col_entry.get()
    attrval2 = add2_val_entry.get()
    if attrcol2 and attrval2 and attrcol2 in gdf.columns: # Check if gdf and column exist
        attrval2 = type(gdf[attrcol2].iloc[0])(attrval2)

def filter_file(gdf_in, attrcol1=None, attrval1=None, attrcol2=None, attrval2=None):#, sample_n=20):
    #global gdf

    # Create a dictionary with column-value pairs
    conditions = {attrcol1: attrval1, attrcol2: attrval2}

    # Remove None values
    conditions = {k: v for k, v in conditions.items() if k is not None and v is not None}

    # Apply conditions
    gdf_new = gdf_in

    for col, val in conditions.items():
        if col in gdf_new.columns:
            gdf_new = gdf_new[gdf_new[col] == val]

    gdf = gdf_new#.sample(n=sample_n)#, replace=False)
    return(gdf)

def save_shapefile():
    global non_result_gdf_tot
    filename = filedialog.asksaveasfilename(defaultextension=".shp")
    if filename:
        # Check for columns before dropping
        cols_to_drop = ['nearest_point', 'poly_geometry']
        cols_exist = [col for col in cols_to_drop if col in non_result_gdf_tot.columns]
        non_result_gdf_tot_to_save = non_result_gdf_tot.drop(columns=cols_exist)
        non_result_gdf_tot_to_save.to_file(filename)


def update_tree_columns():
    if 'gdf' in globals() and gdf is not None:
        tree["columns"] = list(gdf.columns)
        for col in gdf.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
fig, ax = None, None

def visualize_shapefile1(filename=None):
    global bg_gdf, ax, canvas, tree, fig
    if filename is not None:
        bg_gdf = gpd.read_file(filename)

        # Create the main figure and canvas
        fig, ax = plt.subplots(figsize = (10,4))
        bg_gdf.plot(ax = ax, color='none', edgecolor='black')

        # Create the main figure and canvas
        fig.subplots_adjust(top=1)
        if canvas:
            canvas.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=6, column=0, columnspan=4, rowspan=4, sticky='nsew')

        # Create a Frame for the Treeview widget
        frame = tk.Frame(root)
        frame.grid(row=11, column=0, columnspan=4, rowspan = 4, sticky='nsew')

        # Create a Treeview widget
        tree = ttk.Treeview(frame, selectmode="browse", xscrollcommand=lambda x: tree.xview(x))  # Add xscrollcommand to allow horizontal scrolling
        tree["show"] = "headings"

        if 'gdf' in globals() and gdf is not None:
            update_tree_columns()  # Initial tree setup

            # Add data to the Treeview widget
            for index, row in gdf.iterrows():
                tree.insert("", "end", values=list(row))

        # Add the Treeview widget to the window
        tree.place(relheight=1, relwidth=1)

        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        hsb.place(relx=0, rely=1, relwidth=1, anchor='sw')  # Use the place method to control the position of the scrollbar
        tree.configure(xscrollcommand=hsb.set)

        def on_select(event):
            if 'gdf' not in globals() or gdf is None:
                return
            selected_item = tree.selection()[0]  # Get the selected item
            values = tree.item(selected_item)["values"]  # Get the values of the selected item
            for i, value in enumerate(values):
                gdf.iloc[int(selected_item), i] = value  # Update the corresponding row in the GeoDataFrame

        tree.bind("<<TreeviewSelect>>", on_select)

def visualize_shapefile2(filename):
    global gdf, ax, canvas, tree, fig
    gdf = gpd.read_file(filename)

    # Create the main figure and canvas if they don't exist
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize = (10,4))
    gdf.plot(ax=ax, color = "black", alpha = 0.5)
    num_points1 = len(gdf)
    ax.text(0.97, 0.95, f'Number of points: {num_points1}',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes, color='black', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    fig.subplots_adjust(top=1)
    if canvas:
        canvas.get_tk_widget().destroy()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=6, column=0, columnspan=4, rowspan=4, sticky='nsew')

    # Create a Frame for the Treeview widget
    frame = tk.Frame(root)
    frame.grid(row=11, column=0, columnspan=4, rowspan = 4, sticky='nsew')

    # Create a Treeview widget
    tree = ttk.Treeview(frame, selectmode="browse", xscrollcommand=lambda x: tree.xview(x))  # Add xscrollcommand to allow horizontal scrolling
    tree["show"] = "headings"
    update_tree_columns()  # Initial tree setup

    # Add data to the Treeview widget
    for index, row in gdf.iterrows():
        tree.insert("", "end", values=list(row))

    # Add the Treeview widget to the window
    tree.place(relheight=1, relwidth=1)

    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    hsb.place(relx=0, rely=1, relwidth=1, anchor='sw')  # Use the place method to control the position of the scrollbar
    tree.configure(xscrollcommand=hsb.set)

    def on_select(event):
        selected_item = tree.selection()[0]  # Get the selected item
        values = tree.item(selected_item)["values"]  # Get the values of the selected item
        for i, value in enumerate(values):
            gdf.iloc[int(selected_item), i] = value  # Update the corresponding row in the GeoDataFrame

    tree.bind("<<TreeviewSelect>>", on_select)


def visualize_gdf():
    global gdf_filtered, canvas
    if 'gdf' not in globals() or gdf is None:
        print("GDF not loaded")
        return

    plt.clf()
    # Filter the gdf
    conditions = [
        (temcol1, temp_sel1, temp_sel2),
        (temcol2, temp_sel3, temp_sel4),
        (temcol3, temp_sel5, temp_sel6)
    ]
    gdf_filtered = gdf
    for col, min_val, max_val in conditions:
        if col in gdf.columns:
            if min_val is not None:
                gdf_filtered = gdf_filtered[gdf_filtered[col]>=min_val]
            if max_val is not None:
                gdf_filtered = gdf_filtered[gdf_filtered[col]<=max_val]

    conditions = [
        (attrcol1, attrval1),
        (attrcol2, attrval2)
    ]

    for col, val in conditions:
        if col in gdf.columns:
            if val is not None:
                gdf_filtered = gdf_filtered[gdf_filtered[col] == val]

    # Perform Kriging
    #OK = OrdinaryKriging(gdf_filtered.geometry.x, gdf_filtered.geometry.y, gdf_filtered[weight_col], variogram_model='gaussian')
    #x = np.linspace(min(gdf.geometry.x), max(gdf.geometry.x), 100)
    #y = np.linspace(min(gdf.geometry.y), max(gdf.geometry.y), 100)
    #z, ss = OK.execute('grid', x, y)

    # Create the main figure and canvas
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        ax.clear() # Clear existing plot

    # Plot Kriging results
    #ax.imshow(z, origin='lower', aspect='auto',
    #          extent=[min(gdf.geometry.x), max(gdf.geometry.x), min(gdf.geometry.y), max(gdf.geometry.y)], alpha=0.5)

    if 'bg_gdf' in globals() and bg_gdf is not None:
        bg_gdf.plot(ax = ax, color='none', edgecolor='black')

    # Plot points
    gdf_filtered.plot(ax=ax, color='cornflowerblue', alpha=0.5)
    num_points2 = len(gdf_filtered)

# Add text to the top right of the plot
    ax.text(0.97, 0.95, f'Number of points: {num_points2}',
            verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes, color='blue', fontsize=10)
    # Update the canvas
    if canvas:
        canvas.figure = fig
        canvas.draw()
        canvas.get_tk_widget().grid(row=6, column=0, columnspan=4, rowspan=4, sticky='nsew')

    # plt.show() # This might open a separate window, canvas.draw() is usually enough for tkinter

#intersecting the polygon after generating the buffer area
def create_intersected_areas(gdf, buffer_distance):
    gdf_copy = gdf.copy()  # Create a copy of gdf to avoid modifying the original DataFrame
    gdf_copy['geometry'] = gdf_copy['geometry'].apply(Point)
    gdf_copy['geometry'] = gdf_copy['geometry'].buffer(buffer_distance)
    unioned_polygon = unary_union(gdf_copy.geometry)
    unioned_gdf = gpd.GeoDataFrame(geometry=[unioned_polygon])

    #intersected_areas = gpd.overlay(unioned_gdf, gdf_copy, how='intersection')

    #return intersected_areas

    lines = gdf_copy.boundary

    lines_gdf = gpd.GeoDataFrame(geometry=lines)

    unioned_linestring = unary_union(lines_gdf.geometry)

    unioned_lines_gdf = gpd.GeoDataFrame(geometry=[unioned_linestring])

    combined_lines = unary_union([unioned_polygon.boundary, unioned_linestring])

    cut_polygons = list(polygonize(combined_lines))

    cut_region_gdf = gpd.GeoDataFrame(geometry=cut_polygons, crs = gdf.crs)
    cut_region_gdf['index_id'] = range(len(cut_region_gdf))
    return cut_region_gdf

#generating the random points on the polygon area
def generate_random(number, polygon):
    polygon_geom = unary_union(polygon.geometry) # Get the geometry
    points = []
    minx, miny, maxx, maxy = polygon_geom.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon_geom.contains(pnt):
            points.append(pnt)

    # Use the crs from the input polygon GeoDataFrame
    gdf_out = gpd.GeoDataFrame(geometry=points, crs=polygon.crs)
    gdf_out['index_id'] = range(len(gdf_out))
    return gdf_out


#allocating the demand points to the centers
def find_nearest(source_gdf, target_gdf):
    nearest_ids = []  # Create a list to store the nearest ids
    nearest_points_list = []
    nearest_distances = []

    for i, row in source_gdf.iterrows():
        source_point = row.geometry
        distances = target_gdf.distance(source_point)
        min_idx = distances.idxmin()
        min_distance = distances.min()
        nearest_ids.append(target_gdf.loc[min_idx, 'index_id'])
        nearest_points_list.append(target_gdf.loc[min_idx].geometry)
        nearest_distances.append(min_distance)

    source_gdf_out = source_gdf.copy()
    source_gdf_out['nearest_id'] = nearest_ids
    source_gdf_out['nearest_point'] = nearest_points_list
    source_gdf_out['distance'] = nearest_distances

    return source_gdf_out

#calculating the coverage and access of points
def calculate_sum_per_row(allocated_demand_gdf, allocated_face_point_gdf):
    sum_weight_list = []  # Create a list to store the sum of 'ExpectedVa' for each row
    sum_distance_list = []  # Create a list to store the sum of distances for each row
    sum_weighted_distance_list = []  # Create a list to store the sum of weighted distances for each row

    allocated_face_point_gdf_out = allocated_face_point_gdf.copy()

    for i, row in allocated_face_point_gdf_out.iterrows():
        distances = allocated_demand_gdf.geometry.apply(lambda x: row.geometry.distance(x))
        #scrollbar1_label = Label(root, text="From: ", font=font)
        #scrollbar1_label.grid(row=2, column=5, sticky='nsew')

        filtered_gdf_in = allocated_demand_gdf[(distances <= demand_buffer_distance+0.01) & (allocated_demand_gdf['nearest_id'] == row['nearest_id'])]
        filtered_gdf_all = allocated_demand_gdf[(allocated_demand_gdf['nearest_id'] == row['nearest_id'])]

        sum_weight = filtered_gdf_in[weight_col].sum()
        sum_distance = distances[filtered_gdf_all.index].sum()  # Calculate the sum of distances for the filtered rows
        sum_weighted_distance = (distances[filtered_gdf_all.index] * filtered_gdf_all[weight_col]).sum()  # Calculate the sum of weighted distances for the filtered rows

        sum_weight_list.append(sum_weight)
        sum_distance_list.append(sum_distance)
        sum_weighted_distance_list.append(sum_weighted_distance)

    allocated_face_point_gdf_out['sum_weight'] = sum_weight_list
    allocated_face_point_gdf_out['sum_distance'] = sum_distance_list  # Add the list of sums of distances as a new column in allocated_face_point_gdf
    allocated_face_point_gdf_out['sum_weighted_distance'] = sum_weighted_distance_list  # Add the list of sums of weighted distances as a new column in allocated_face_point_gdf

    return allocated_face_point_gdf_out

#deriving the weber point
def weber(gdf, uniq_id):
    # Create an empty DataFrame to store the results
    result_gdf = gpd.GeoDataFrame()

    # Calculate the weighted center for each unique 'nearest_id' value
    for nearest_id in gdf[uniq_id].unique():
        # Filter the rows with the current 'nearest_id' value
        filtered_gdf = gdf[gdf[uniq_id] == nearest_id]

        # Calculate the weighted sums of the x and y coordinates
        weighted_sum_x = (filtered_gdf.geometry.x * filtered_gdf[weight_col]).sum()
        weighted_sum_y = (filtered_gdf.geometry.y * filtered_gdf[weight_col]).sum()

        # Calculate the total sum of the weights
        total_weight = filtered_gdf[weight_col].sum()

        # Calculate the weighted center coordinates
        center_x = weighted_sum_x / (total_weight+0.001)
        center_y = weighted_sum_y / (total_weight+0.001)

        # Create a new DataFrame with the results for the current 'nearest_id' value
        result_row = gpd.GeoDataFrame({'index_id': [nearest_id], 'center_x': [center_x], 'center_y': [center_y]}, geometry=gpd.points_from_xy([center_x], [center_y]),
                                     crs = gdf.crs)

        # Append the results to the result DataFrame
        result_gdf = pd.concat([result_gdf, result_row])

    return result_gdf

#finding polygon's nearest point to the weber point
def find_nearest2(face_poly_gdf, solution_weber):
    # Create an empty list to store the results
    nearest_points_list = []

    # Iterate over each polygon
    for poly in face_poly_gdf.geometry:
        # Initialize the minimum distance to a large number
        min_dist = float('inf')
        nearest_pt = None

        # Iterate over each point
        for pt in solution_weber.geometry:
            # Find the nearest point on the polygon to the current point
            nearest = nearest_points(poly, pt)[0]

            # Calculate the distance between the point and the nearest point on the polygon
            dist = pt.distance(nearest)

            # If this distance is less than the current minimum distance, update the minimum distance and the nearest point
            if dist < min_dist:
                min_dist = dist
                nearest_pt = nearest

        # Add the nearest point and the original polygon to the results list
        nearest_points_list.append((nearest_pt, poly))

    # Convert the list to a GeoDataFrame
    nearest_points_gdf = gpd.GeoDataFrame(nearest_points_list, columns=['nearest_point', 'poly_geometry'], geometry='nearest_point',
                                          crs = face_poly_gdf.crs)

    nearest_points_gdf['x'] = nearest_points_gdf['nearest_point'].x
    nearest_points_gdf['y'] = nearest_points_gdf['nearest_point'].y

    # Create a new 'geometry' column from the x and y coordinates
    nearest_points_gdf['geometry'] = gpd.points_from_xy(nearest_points_gdf['x'], nearest_points_gdf['y'])

    return nearest_points_gdf

#sorting for the solution
def find_non_dominated_solutions(gdf):
    #global result_gdf
    # Initialize an empty dataframe to store the non-dominated solutions
    non_dominated_gdf = pd.DataFrame()

    # Iterate over each unique 'nearest_id'
    for nearest_id in gdf['nearest_id'].unique():
        temp_gdf = gdf[gdf['nearest_id'] == nearest_id]

        # Sort the temporary dataframe by 'sum_weight' (descending) and 'sum_weighted_distance' (ascending)
        temp_gdf_sorted = temp_gdf.sort_values(by=['sum_weight', 'sum_weighted_distance'], ascending=[False, True])

        # Initialize the current maximum 'sum_weight' and minimum 'sum_weighted_distance'
        min_sum_weighted_distance = temp_gdf_sorted.iloc[0]['sum_weighted_distance']

        # Add the first row (which is non-dominated by definition) to the non-dominated solutions dataframe
        non_dominated_gdf = pd.concat([non_dominated_gdf, temp_gdf_sorted.iloc[[0]]])

        # Iterate over the rest of the rows
        for i in range(1, len(temp_gdf_sorted)):
            row = temp_gdf_sorted.iloc[i]

            # If the current row is non-dominated, update the current maximum 'sum_weight' and minimum 'sum_weighted_distance', and add the row to the non-dominated solutions dataframe
            if row['sum_weighted_distance'] <= min_sum_weighted_distance:
                min_sum_weighted_distance = row['sum_weighted_distance']
                non_dominated_gdf = pd.concat([non_dominated_gdf, temp_gdf_sorted.iloc[[i]]])

    return non_dominated_gdf

#filtering out the gdf using quantile
def generate_gdf(gdf, quantile):
    # Define quantile value
    q = quantile / (maximum_lambda-1) if maximum_lambda > 1 else 0

    if q == 0:
        # If quantile is 0, select the row with the smallest 'sum_weight' for each 'nearest_id'
        idx = gdf.groupby('nearest_id')['sum_weight'].idxmin()
    else:
        # If quantile is not 0 (e.g., 1 if maximum_lambda is 2), select the row with the largest 'sum_weight' for each 'nearest_id'
        idx = gdf.groupby('nearest_id')['sum_weight'].idxmax()

    # Use the indices to select the corresponding rows from the dataframe
    new_gdf = gdf.loc[idx]

    return new_gdf

#calculating the values of the results
def calculate_values(demand_gdf, gdf):
    # Create an empty DataFrame to hold the results
    result_df = pd.DataFrame(columns=['uniq_ind', 'sum_expected_va', 'weighted_distance'])

    demand_gdf_copy = demand_gdf.copy()

    # For each unique 'uniq_ind' in result_gdf
    for uniq_ind in gdf['uniq_ind'].unique():
        # Select the rows with the current 'uniq_ind'
        result_subset = gdf[gdf['uniq_ind'] == uniq_ind]

        # Calculate the distance from each point in demand_gdf to the nearest point in result_subset
        demand_gdf_copy['distance'] = demand_gdf_copy.geometry.apply(lambda x: result_subset.distance(x).min())

        # Filter the rows in demand_gdf where the distance is less than 30
        close_points = demand_gdf_copy[demand_gdf_copy['distance'] < demand_buffer_distance].copy()

        # Calculate the sum of ExpectedVa for these points
        sum_expected_va = close_points[weight_col].sum()

        # Calculate the shortest distance to result_subset multiplied by ExpectedVa
        weighted_distance = (demand_gdf_copy['distance'] * demand_gdf_copy[weight_col]).sum()

        # Append the results to result_df
        new_row = pd.DataFrame({'uniq_ind': [uniq_ind], 'sum_expected_va': [sum_expected_va], 'weighted_distance': [weighted_distance]})
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    result_df['covered_per'] = 100*result_df["sum_expected_va"] / (demand_gdf_copy[weight_col].sum() + 0.001)
    result_df['mean_dist'] = result_df["weighted_distance"] / (demand_gdf_copy[weight_col].sum() + 0.001)

    return result_df

# non-dominated solutions
def non_dominated_solutions(gdf):
    # Initialize an empty dataframe to store the non-dominated solutions
    non_dominated_gdf = pd.DataFrame()

    # Sort the temporary dataframe by 'sum_weight' (descending) and 'sum_weighted_distance' (ascending)
    gdf_sorted = gdf.sort_values(by=['sum_expected_va', 'weighted_distance'], ascending=[False, True])

    if gdf_sorted.empty:
        return non_dominated_gdf

    # Initialize the current maximum 'sum_weight' and minimum 'sum_weighted_distance'
    min_sum_weighted_distance = gdf_sorted.iloc[0]['weighted_distance']

    # Add the first row (which is non-dominated by definition) to the non-dominated solutions dataframe
    non_dominated_gdf = pd.concat([non_dominated_gdf, gdf_sorted.iloc[[0]]])

    # Iterate over the rest of the rows
    for i in range(1, len(gdf_sorted)):
        row = gdf_sorted.iloc[i]

        # If the current row is non-dominated, update the current maximum 'sum_weight' and minimum 'sum_weighted_distance', and add the row to the non-dominated solutions dataframe
        if row['weighted_distance'] < min_sum_weighted_distance:
            min_sum_weighted_distance = row['weighted_distance']
            non_dominated_gdf = pd.concat([non_dominated_gdf, gdf_sorted.iloc[[i]]])

    return non_dominated_gdf

#index of calculation
def calculate_mean_distance(gdf1, gdf2):
    # Ensure the GeoDataFrames are indexed by 'nearest_id'
    gdf1_indexed = gdf1.set_index('nearest_id')
    gdf2_indexed = gdf2.set_index('nearest_id')

    # Align gdf2 to gdf1's index
    gdf2_aligned = gdf2_indexed.reindex(gdf1_indexed.index)

    # Drop rows where alignment failed (NaN geometries)
    valid_indices = gdf2_aligned.geometry.notna()
    gdf1_aligned = gdf1_indexed[valid_indices]
    gdf2_aligned = gdf2_aligned[valid_indices]

    if gdf1_aligned.empty or gdf2_aligned.empty:
        return float('inf') # Return a large number if no common indices

    # Calculate the distance between corresponding points
    distances = gdf1_aligned.distance(gdf2_aligned)

    # Return the mean distance
    return distances.mean()

def non_dom_coords(gdf, result_gdf):
    result_gdf2 = calculate_values(gdf, result_gdf)
    result_gdf2_sorted = result_gdf2.sort_values(by=['sum_expected_va', 'weighted_distance'], ascending=[False, True])

    if result_gdf2_sorted.empty:
        return gpd.GeoDataFrame(geometry=[]) # Return empty GeoDataFrame

    min_sum_weighted_distance = result_gdf2_sorted.iloc[0]['weighted_distance']
    uniq_id_list = [result_gdf2_sorted.iloc[0]['uniq_ind']] # Add the first one

    for i in range(1, len(result_gdf2_sorted)):
        row = result_gdf2_sorted.iloc[i]
        if row['weighted_distance'] <= min_sum_weighted_distance:
            min_sum_weighted_distance = row['weighted_distance']
            uniq_id_list.append(row['uniq_ind'])

    result_gdf3 = result_gdf[result_gdf['uniq_ind'].isin(uniq_id_list)]
    return(result_gdf3)

def analysis(demand_gdf):
    #global result_gdf
    result_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(dtype=Point))
    result_gdf.crs = demand_gdf.crs

    #intersection
    face_poly_gdf = create_intersected_areas(demand_gdf, demand_buffer_distance)

    uniq_ind = 0

    for k in range(0,iteration):
        #generate random initial points
        solution_prev = demand_gdf.sample(n=num_fac)
        solution_prev['index_id'] = range(len(solution_prev))
        solution_prev['nearest_id'] = range(len(solution_prev))

        for quant in range(0, maximum_lambda):
            for i in range(0,max_iter):
            #allocate demands and fac points
                alloc_demand_gdf = find_nearest(demand_gdf, solution_prev)
                solution_weber = weber(alloc_demand_gdf, "nearest_id")
                alloc_face_point_gdf = find_nearest2(face_poly_gdf, solution_weber)
                alloc_face_point_gdf = find_nearest(alloc_face_point_gdf, solution_prev)
                alloc_face_point_gdf.geometry = alloc_face_point_gdf['geometry']

            #assess the solution
                alloc_face_point_gdf = calculate_sum_per_row(alloc_demand_gdf, alloc_face_point_gdf)

            #find non-dominated solutions
                non_face_point_gdf = find_non_dominated_solutions(alloc_face_point_gdf)

                if non_face_point_gdf.empty:
                    break # Break inner loop if no non-dominated solutions

            #quantile based solution search
                solution_next = generate_gdf(non_face_point_gdf, quant)
                solution_next['index_id'] = range(len(solution_next))

                if calculate_mean_distance(solution_prev, solution_next) < threshold_dist:
                    solution_next['uniq_ind'] = uniq_ind
                    uniq_ind = uniq_ind + 1
                    result_gdf = pd.concat([result_gdf, solution_next])
                    break
                else:
                    solution_prev = solution_next

            if non_face_point_gdf.empty:
                break # Break outer loop

    result_gdf_non_dom = non_dom_coords(demand_gdf, result_gdf)
    #result_gdf = result_gdf[result_gdf['uniq_ind'] > 0]
    return(result_gdf_non_dom)


def voronoi_rand(gdf_filtered):
    global gdf_voro, num_points
    num_points = num_fac*20
    if len(gdf_filtered) >= num_points:
        selected_points = gdf_filtered.sample(num_points)
    else:
        selected_points = gdf_filtered

    coordinates = selected_points.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()

    voro = voronoi_diagram(MultiPoint(coordinates))
    gdf_voro = gpd.GeoDataFrame(geometry=list(voro.geoms))
    gdf_voro.crs = gdf_filtered.crs # Set CRS
    gdf_voro = gpd.overlay(gdf_voro, gdf_convexhull, how='intersection')
    return gdf_voro

def create_new_points(row, gdf_filtered, weight_col):
    global new_point_gdf
    sel_polygon = row.geometry

    sel_gdf_filtered = gdf_filtered[gdf_filtered.within(sel_polygon)]

    if sel_gdf_filtered.empty:
        # Handle empty case: maybe return None or an empty GeoDataFrame
        return gpd.GeoDataFrame(columns=[weight_col, 'geometry'], crs=gdf_filtered.crs)


    mean_x = np.mean(sel_gdf_filtered.geometry.x)
    std_x = np.std(sel_gdf_filtered.geometry.x)
    mean_y = np.mean(sel_gdf_filtered.geometry.y)
    std_y = np.std(sel_gdf_filtered.geometry.y)

    # Generate new x and y coordinates based on Gaussian distribution
    new_x = np.random.normal(mean_x, std_x if std_x > 0 else 0.1)
    new_y = np.random.normal(mean_y, std_y if std_y > 0 else 0.1)

    # Create new point
    new_point = Point(new_x, new_y)

    # If the new point is outside the polygon, move it to the nearest location inside the polygon
    if not sel_polygon.contains(new_point):
        new_point = nearest_points(sel_polygon, new_point)[0]

    new_point_gdf = gpd.GeoDataFrame(geometry=[new_point])
    weight_mean = np.mean(sel_gdf_filtered[weight_col])
    weight_std = np.std(sel_gdf_filtered[weight_col])
    new_point_gdf[weight_col] = np.random.normal(weight_mean, weight_std if weight_std > 0 else 0.1)
    new_point_gdf.crs = gdf_filtered.crs

    return new_point_gdf

def create_new_points_apply():
    global all_new_points_gdf
    gdf_voro = voronoi_rand(gdf_filtered)
    all_new_points_list = gdf_voro.apply(create_new_points, axis=1, args=(gdf_filtered, weight_col))

    # Filter out empty GeoDataFrames
    all_new_points_list = [gdf for gdf in all_new_points_list if not gdf.empty]

    if not all_new_points_list:
        # Handle case where all results were empty
        all_new_points_gdf = gpd.GeoDataFrame(columns=[weight_col, 'geometry'], crs=gdf_filtered.crs)
    else:
        all_new_points_gdf = pd.concat(all_new_points_list, ignore_index=True)


