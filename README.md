# LiDAR Catenary Detection

A modular Python pipeline to process **LiDAR point clouds**, detect **cables and wires**, fit **best-fit planes**, project them into 2D, and estimate **catenary curves** with quantitative evaluation.

---

## Features
- **Cable Clustering** using DBSCAN.  
- **Wire Clustering** within each cable using per-cable optimized `eps`.  
- **Plane Fitting** via PCA.  
- **2D Projection** of wire points for curve fitting.  
- **Catenary Fitting** using nonlinear optimization with RMSE evaluation.  
- **Visualization Tools** for 3D clusters, planes, 2D projections, and catenary fits.  
- **Reproducible & Scalable** design with modular Python scripts.  

---

## Project Structure
lidar_catenary/

│── clustering.py # Cable & wire clustering (DBSCAN + eps optimization)

│── plane_fit.py # Fit best-fit planes using PCA

│── projection.py # Project 3D wire points into 2D

│── catenary.py # Fit catenary curves and compute RMSE

│── visualize.py # Visualization utilities (3D/2D plots, planes, catenaries)

│── main.py # Entry point: runs the full pipeline

│── data/ # Place your .parquet LiDAR files here



---

## Setup & Installation

### Clone Repository
```bash
git clone https://github.com/Pooja-72/lidar_catenary.git
cd lidar_catenary
```

## Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib scipy kneed
```

## Add Data
Place your .parquet files inside the data/ directory.

---

## Running the pipeline

Run the main script:
```bash
python main.py
```

You will be prompted to choose a dataset.

The pipeline will:

- Cluster cables.
- Detect wires per cable.
- Fit best-fit planes.
- Project wires into 2D.
- Fit catenary curves.
- Output plots + summary tables.

---

## Example Outputs

- Wire Count Summary per Cable
- Plane equations for first 5 wires
- Catenary fit plots overlayed with wire points
- Top 3 wires ranked by RMSE

---

## Evaluation Criteria

This project demonstrates:
- Code Quality & Organisation — Modular, well-documented scripts.
- Reproducibility — Works on any .parquet dataset.
- Best Practices — Version control, maintainability, modular design.
- Performance & Scalability — DBSCAN + PCA handle large datasets efficiently.
- Clarity — Visualizations & logs make results interpretable.
- Logical Approach — Complex problem broken into reproducible steps.