import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def cluster_point_cloud(df, eps=0.6, min_samples=6):
    """
    Clusters a 3D point cloud using DBSCAN.

    Parameters:
        df (pd.DataFrame): DataFrame with 'x', 'y', 'z' columns
        eps (float): DBSCAN epsilon parameter (distance threshold)
        min_samples (int): DBSCAN minimum samples per cluster

    Returns:
        pd.DataFrame: Original DataFrame with a new 'cluster' column
        int: Number of clusters detected (excluding noise)
    """
    coords = df[['x', 'y', 'z']].dropna()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)
    df = coords.copy()
    df['cluster'] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✅ Clusters found (excluding noise): {n_clusters}")
    return df, n_clusters



def estimate_optimal_eps(df, k=5):
    """
    Estimate optimal eps using k-distance and KneeLocator.
    """
    coords = df[['x', 'y', 'z']].dropna().values
    distances, _ = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(coords)
    k_distances = np.sort(distances[:, k-1])

    knee = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
    if knee.knee is not None:
        return k_distances[knee.knee], k_distances, knee
    return k_distances[int(len(k_distances) * 0.9)], k_distances, knee


def cluster_wires(cable_df, eps=0.4, min_samples=6):
    """
    Cluster wires inside a single cable.
    """
    cable_df = cable_df.copy()
    cable_df['wire_cluster'] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(cable_df[['x', 'y', 'z']])

    unique_wires = [wid for wid in cable_df['wire_cluster'].unique() if wid != -1]
    return cable_df, len(unique_wires)


def cluster_wires_per_cable(df, eps_by_cable, min_samples=5):
    """
    Cluster wires inside each cable using per-cable eps values.
    """
    df = df.copy()
    summary = []

    for cable_id in sorted(df['cluster'].unique()):
        if cable_id == -1 or cable_id not in eps_by_cable:
            continue

        cable_df = df[df['cluster'] == cable_id].copy()
        eps = eps_by_cable[cable_id]
        cable_df['wire_cluster'] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(cable_df[['x', 'y', 'z']])

        df.loc[cable_df.index, 'wire_cluster'] = cable_df['wire_cluster']
        unique_wires = [w for w in cable_df['wire_cluster'].unique() if w != -1]

        summary.append({'Cable ID': cable_id, 'Wire Count': len(unique_wires)})
        print(f"🧵 Cable {cable_id} → Wires: {len(unique_wires)} (eps = {eps:.3f})")

    return df, summary


def optimize_eps_per_cable(df, k=5, plot=False):
    """
    Estimate optimal eps for each cable cluster separately.
    """
    from lidar_catenary.visualize import plot_knee_curve

    eps_by_cable = {}
    for cable_id in sorted(df['cluster'].unique()):
        if cable_id == -1:
            continue

        cable_df = df[df['cluster'] == cable_id]
        eps, k_distances, knee = estimate_optimal_eps(cable_df, k=k)
        eps_by_cable[cable_id] = eps

        if plot:
            plot_knee_curve(k_distances, knee, eps, title=f"Cable {cable_id}")

        print(f"✅ Cable {cable_id}: Optimal eps = {eps:.3f}")

    return eps_by_cable


def summarize_wire_counts(summary):
    """
    Convert summary list of dicts into a DataFrame report.
    """
    summary_df = pd.DataFrame(summary)
    print("\n📊 Wire Count per Cable Cluster:")
    return summary_df
