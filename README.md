LAMOS: Location Allocation Model Optimization System

LAMOS is a desktop application designed for spatial analysis and location allocation modeling. It provides a graphical user interface (GUI) built with Tkinter to help users load spatial data, set parameters for location analysis, perform exploratory spatial data analysis (ESDA), and run optimization models to find non-dominated solutions for facility locations.

Features

Interactive Map Display: Load and visualize area shapefiles (background) and demand point shapefiles.

Data Filtering: Filter demand points based on temporal ranges (up to 3 variables) and specific attributes (up to 2 variables).

Exploratory Spatial Data Analysis (ESDA):

Generate Kernel Density Estimation (KDE) plots to visualize spatial distributions based on temporal filters.

Perform comparative analysis between filtered and unfiltered data, including:

Kolmogorov-Smirnov (KS) tests on coordinates and weight columns.

Ripley's K analysis with Monte Carlo simulations.

Voronoi intersection analysis to compare spatial specialization.

Location Allocation Modeling:

Run an optimization algorithm (based on Weber problem and non-dominated sorting) to find optimal facility locations.

Set key parameters like the number of facilities, coverage distance, and iteration counts.

Results Visualization:

View dominated and non-dominated (optimal) solutions on a map.

Analyze trade-offs between coverage percentage and mean distance using scatter plots and KDE plots.

Save Results: Export the final non-dominated solution points as a shapefile.

Prerequisites

Before you begin, ensure you have the following installed on your Windows machine:

Git: Download & Install Git

Python 3: Download & Install Python (Make sure to check "Add Python to PATH" during installation)

How to Run

You can easily clone, install, and run the application by using the provided batch file.

Download the run.bat file from this repository.

Save it to a folder where you want to install the project (e.g., your Desktop or Documents folder).

Double-click run.bat.

This script will automatically perform the following steps:

Clone this repository (manjukcus/lamos) into a new folder named lamos.

Create a Python virtual environment (venv) inside that folder.

Activate the virtual environment.

Install all the required Python packages from requirements.txt.

Run the application (lamos_app.py).

Manual Installation (Alternative)

If you prefer to install the application manually:

# 1. Clone the repository
git clone [https://github.com/manjukcus/lamos.git](https://github.com/manjukcus/lamos.git)

# 2. Navigate into the project directory
cd lamos

# 3. Create a virtual environment
python -m venv venv

# 4. Activate the virtual environment
.\venv\Scripts\activate.bat

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the application
python lamos_app.py


Files in this Repository

lamos_app.py: The main Python application source code.

requirements.txt: A list of all Python libraries required by the application.

run.bat: A Windows batch script to automate installation and execution.

README.md: This file.