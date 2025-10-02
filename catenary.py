import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .projection import project_wire_to_2d_plane


def catenary(x, c, x0, y0):
    """Catenary equation in 2D."""
    return y0 + c * (np.cosh((x - x0) / c) - 1)


def find_param(points):
    """Find lowest point (x0, y0)."""
    y_min_idx = np.argmin(points[:, 1])
    return points[y_min_idx, 0], points[y_min_idx, 1]


def loss(c, points, x0, y0):
    """MSE loss for catenary fit."""
    y_pred = catenary(points[:, 0], c[0], x0, y0)
    return np.mean((points[:, 1] - y_pred) ** 2)


def fit_catenary(points, initial_c=10.0):
    """Fit catenary curve using optimization, return params + RMSE."""
    x0, y0 = find_param(points)

    result = minimize(lambda c: loss(c, points, x0, y0),
                      x0=[initial_c], bounds=[(0.1, None)])
    if not result.success:
        return None

    c = result.x[0]

    # Compute RMSE
    y_pred = catenary(points[:, 0], c, x0, y0)
    rmse = np.sqrt(np.mean((points[:, 1] - y_pred) ** 2))

    return x0, y0, c, rmse




def fit_catenary_pipeline(df, eps_by_cable=None, level="wire", max_items=5, top_n=None):
    """
    Fit catenary curves at wire or cable level.

    Returns:
        DataFrame with parameters [cable_id, wire_id, x0, y0, c, rmse]
    """
    from sklearn.cluster import DBSCAN

    results = []
    counter = 0

    for cable_id in sorted(df['cluster'].unique()):
        if cable_id == -1:
            continue
        cable_df = df[df['cluster'] == cable_id].copy()

        if level == "cable":
            projected_df, _, _ = project_wire_to_2d_plane(cable_df)
            params = fit_catenary(projected_df[['u', 'v']].to_numpy())
            if params:
                x0, y0, c, rmse = params
                results.append({"cable_id": cable_id, "x0": x0, "y0": y0, "c": c, "rmse": rmse})
            counter += 1

        elif level == "wire":
            if eps_by_cable is None and "wire_cluster" not in df.columns:
                raise ValueError("eps_by_cable must be provided for wire-level fitting if wire_cluster not in df.")

            # ✅ Reuse existing wire clusters if available
            if "wire_cluster" in cable_df.columns:
                clusters = sorted(cable_df['wire_cluster'].unique())
            else:
                eps = eps_by_cable[cable_id]
                cable_df['wire_cluster'] = DBSCAN(eps=eps, min_samples=5).fit_predict(cable_df[['x', 'y', 'z']])
                clusters = sorted(cable_df['wire_cluster'].unique())

            for wire_id in clusters:
                if wire_id == -1:
                    continue
                wire_df = cable_df[cable_df['wire_cluster'] == wire_id]
                projected_df, _, _ = project_wire_to_2d_plane(wire_df)
                params = fit_catenary(projected_df[['u', 'v']].to_numpy())
                if params:
                    x0, y0, c, rmse = params
                    results.append({
                        "cable_id": cable_id,
                        "wire_id": wire_id,
                        "x0": x0, "y0": y0, "c": c, "rmse": rmse
                    })
                counter += 1
                if counter >= max_items:
                    break
        else:
            raise ValueError("level must be 'wire' or 'cable'")

        if counter >= max_items:
            break

    df_results = pd.DataFrame(results)
    if top_n:
        df_results = df_results.sort_values(by="rmse").head(top_n)  # sort by best fit
    return df_results
