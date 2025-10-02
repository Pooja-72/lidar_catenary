import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa
from .catenary import catenary


# --- Cluster Visualizations ---

def plot_3d_clusters(df, col='cluster', title="3D Clusters of LiDAR Wires"):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['x'], df['y'], df['z'], c=df[col], cmap='tab10', s=1)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.colorbar(scatter, label=col)
    plt.show()


def plot_k_distance(k_distances, knee=None, eps=None):
    plt.figure(figsize=(10, 5))
    plt.plot(k_distances, label='k-distances')
    if knee and knee.knee is not None:
        plt.axvline(knee.knee, color='r', linestyle='--', label=f"Knee @ {knee.knee}")
    if eps is not None:
        plt.axhline(eps, color='g', linestyle='--', label=f"eps ≈ {eps:.3f}")
    plt.title("k-distance with KneeLocator")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_knee_curve(k_distances, knee=None, eps=None, title="k-distance with KneeLocator"):
    plot_k_distance(k_distances, knee, eps)


def plot_wire_clusters(cable_df, cable_id):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(cable_df['x'], cable_df['y'], cable_df['z'], c=cable_df['wire_cluster'], cmap='tab20', s=2)
    ax.set_title(f"Wire Clusters in Cable {cable_id}")
    plt.colorbar(scatter, label="Wire Cluster ID")
    plt.show()


def plot_wires_2d(cable_df, cable_id, axis_x='x', axis_y='y'):
    wire_ids = [wid for wid in cable_df['wire_cluster'].unique() if wid != -1]
    num_wires = len(wire_ids)
    print(f"📦 Cable {cable_id} → {num_wires} wires")
    plt.figure(figsize=(10, 6))
    plt.scatter(cable_df[axis_x], cable_df[axis_y], c=cable_df['wire_cluster'], cmap='tab20', s=3)
    plt.title(f"Cable {cable_id} - Wire Clusters ({axis_x} vs {axis_y})")
    plt.grid(True)
    plt.show()


def plot_all_wires_2d(df, eps_by_cable, axis_x='x', axis_y='y', min_samples=5):
    from sklearn.cluster import DBSCAN
    all_points = []
    for cable_id in sorted(df['cluster'].unique()):
        if cable_id == -1 or cable_id not in eps_by_cable:
            continue
        cable_df = df[df['cluster'] == cable_id].copy()
        eps = eps_by_cable[cable_id]
        cable_df['wire_cluster'] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(cable_df[['x', 'y', 'z']])
        cable_df['global_wire_id'] = cable_df['wire_cluster'].apply(lambda w: f"C{cable_id}_W{w}" if w != -1 else f"C{cable_id}_Noise")
        all_points.append(cable_df)

    all_combined = pd.concat(all_points, ignore_index=True)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(all_combined[axis_x], all_combined[axis_y], c=pd.factorize(all_combined['global_wire_id'])[0], cmap='tab20', s=2)
    plt.title("All Cables and Wires in 2D")
    plt.colorbar(scatter, label="Global Wire Cluster ID")
    plt.grid(True)
    plt.show()


# --- Plane Visualizations ---

def plot_wire_with_plane_3d(wire_df, center, normal_vector, pca, plane_size=5):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(wire_df['x'], wire_df['y'], wire_df['z'], s=5, color='blue', label='Wire Points')
    axis1, axis2 = pca.components_[0], pca.components_[1]
    u, v = np.meshgrid(np.linspace(-plane_size, plane_size, 10), np.linspace(-plane_size, plane_size, 10))
    plane_points = center + u[..., None]*axis1 + v[..., None]*axis2
    ax.plot_surface(plane_points[:, :, 0], plane_points[:, :, 1], plane_points[:, :, 2], alpha=0.3, color='orange')
    plt.title("Wire with Best-Fit Plane")
    plt.show()


def plot_wire_projection_2d(projected_df, cable_id, wire_id):
    plt.figure(figsize=(8, 5))
    plt.scatter(projected_df['u'], projected_df['v'], s=3, color='darkgreen')
    plt.title(f"2D Projection — Cable {cable_id}, Wire {wire_id}")
    plt.grid(True)
    plt.show()


# --- Catenary Visualizations ---

def plot_catenary(points, params, title="Catenary Fit"):
    if not params:
        print("⚠️ Fit failed")
        return
    x0, y0, c = params
    x_vals = np.linspace(points[:, 0].min(), points[:, 0].max(), 500)
    y_vals = catenary(x_vals, c, x0, y0)
    plt.figure(figsize=(8, 5))
    plt.scatter(points[:, 0], points[:, 1], s=3, color='red', label="Points")
    plt.plot(x_vals, y_vals, color='blue', linewidth=2, label="Catenary Fit")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()


def plot_wire_with_plane(wire_df, center, pca, wire_label="Wire"):
    """Simplified wrapper that uses PCA axes to plot a plane with wire points."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(wire_df['x'], wire_df['y'], wire_df['z'], s=5, label='Wire Points')

    # Plane mesh
    axis1, axis2 = pca.components_[0], pca.components_[1]
    u, v = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    plane = center + u[..., None]*axis1 + v[..., None]*axis2
    ax.plot_surface(plane[:, :, 0], plane[:, :, 1], plane[:, :, 2], alpha=0.3, color='orange')

    ax.set_title(f"Best-Fit Plane — {wire_label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()
