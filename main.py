import pandas as pd
from sklearn.cluster import DBSCAN

# --- Imports from lidar_catenary ---
from lidar_catenary.clustering import (
    cluster_point_cloud,
    estimate_optimal_eps,
    cluster_wires,
    summarize_wire_counts,
    optimize_eps_per_cable,
    cluster_wires_per_cable,
)
from lidar_catenary.visualize import (
    plot_k_distance,
    plot_knee_curve,
    plot_3d_clusters,
    plot_wire_clusters,
    plot_wires_2d,
    plot_wire_with_plane_3d,
    plot_wire_with_plane,
    plot_wire_projection_2d,
    plot_catenary,
)
from lidar_catenary.plane_fit import (
    fit_best_fit_plane_3d,
    fit_plane_pca,
    get_plane_equation,
)
from lidar_catenary.projection import project_wire_to_2d_plane
from lidar_catenary.catenary import fit_catenary_pipeline


# ==========================
# 🚀 1. Cable-Level Clustering
# ==========================
df, num_clusters = cluster_point_cloud(df, eps=0.6, min_samples=6)
print(f"Clusters found: {num_clusters}")

eps, k_distances, knee = estimate_optimal_eps(df, k=5)
plot_k_distance(k_distances, knee, eps)
print(f"📏 Estimated optimal eps: {eps:.3f}" if eps else "⚠️ No knee point found")

plot_3d_clusters(df, col="cluster")

# ==========================
# 🧵 2. Wire-Level Clustering
# ==========================
summary = []
for cable_id in sorted(df['cluster'].unique()):
    if cable_id == -1:
        continue
    cable_df, num_wires = cluster_wires(df[df['cluster'] == cable_id], eps=0.4, min_samples=6)
    summary.append({'Cable Cluster ID': cable_id, 'Wire Count': num_wires})
    print(f"🔍 Cable {cable_id}: {num_wires} wires")
    plot_wire_clusters(cable_df, cable_id)

summary_df = summarize_wire_counts(summary)
print(summary_df)

# ==========================
# 📊 3. Optimize eps per cable
# ==========================
eps_by_cable = optimize_eps_per_cable(df, k=5, plot=True)
df, summary = cluster_wires_per_cable(df, eps_by_cable, min_samples=5)

for cable_id in sorted(df['cluster'].unique()):
    if cable_id != -1 and cable_id in eps_by_cable:
        plot_wires_2d(df[df['cluster'] == cable_id], cable_id, axis_x='x', axis_y='y')

# ==========================
# 📐 4. Plane fitting
# ==========================
wire_counter = 0
for cable_id in sorted(df['cluster'].unique()):
    if cable_id == -1 or cable_id not in eps_by_cable:
        continue

    cable_df = df[df['cluster'] == cable_id].copy()
    cable_df['wire_cluster'] = DBSCAN(eps=eps_by_cable[cable_id], min_samples=5).fit_predict(cable_df[['x', 'y', 'z']])

    for wire_id in sorted(cable_df['wire_cluster'].unique()):
        if wire_id == -1:
            continue
        wire_df = cable_df[cable_df['wire_cluster'] == wire_id]

        # Fit plane
        center, normal, pca = fit_plane_pca(wire_df)
        a, b, c, d = get_plane_equation(center, normal)

        print(f"\n📐 Wire #{wire_counter+1} — Cable {cable_id}, Wire {wire_id}")
        print(f"Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        # Plot
        plot_wire_with_plane(wire_df, center, pca, wire_label=f"Wire #{wire_counter+1}")

        wire_counter += 1
        if wire_counter >= 5:
            break
    if wire_counter >= 5:
        break

# ==========================
# 🔄 5. Projection to 2D
# ==========================
wire_counter = 0
for cable_id in sorted(df['cluster'].unique()):
    if cable_id == -1 or cable_id not in eps_by_cable:
        continue
    cable_df = df[df['cluster'] == cable_id].copy()
    cable_df['wire_cluster'] = DBSCAN(eps=eps_by_cable[cable_id], min_samples=5).fit_predict(cable_df[['x','y','z']])

    for wire_id in sorted(cable_df['wire_cluster'].unique()):
        if wire_id == -1:
            continue
        wire_df = cable_df[cable_df['wire_cluster'] == wire_id]
        projected_df, _, _ = project_wire_to_2d_plane(wire_df)

        plot_wire_projection_2d(projected_df, cable_id, wire_id)

        wire_counter += 1
        if wire_counter >= 5:
            break
    if wire_counter >= 5:
        break

# ==========================
# ⛓ 6. Catenary fitting
# ==========================
 #Plot a few (fixed version with recomputed wire clusters)
for res in results_df.head(5).to_dict(orient="records"):
    cable_id, wire_id = res["cable_id"], res["wire_id"]
    label = f"Cable {cable_id}, Wire {wire_id}"

    # Recompute wire clusters for this cable (since df doesn’t keep wire_cluster from pipeline)
    cable_df = df[df['cluster'] == cable_id].copy()
    cable_df['wire_cluster'] = DBSCAN(eps=eps_by_cable[cable_id], min_samples=5).fit_predict(
        cable_df[['x','y','z']]
    )

    # Extract wire points
    wire_df = cable_df[cable_df['wire_cluster'] == wire_id]

    # Project to 2D
    projected_df, _, _ = project_wire_to_2d_plane(wire_df)
    points = projected_df[['u','v']].to_numpy()

    # Plot catenary fit
    params = (res["x0"], res["y0"], res["c"])
    plot_catenary(points, params, title=f"{label} Catenary Fit")

# ==========================
# 🏆 7. Top 3 catenaries
# ==========================
top3 = results_df.sort_values(by="c", ascending=False).head(3)
print("\n📊 Top 3 Catenary Fits by Curvature (c):")
print(top3[['cable_id', 'wire_id', 'x0', 'y0', 'c']].to_string(index=False))
