import numpy as np
from sklearn.decomposition import PCA


def fit_best_fit_plane_3d(wire_df):
    """
    Fit a best-fit plane using PCA.
    """
    coords = wire_df[['x', 'y', 'z']].dropna().values
    center = coords.mean(axis=0)
    coords_centered = coords - center

    pca = PCA(n_components=3)
    pca.fit(coords_centered)

    normal = pca.components_[2]  # third component = normal to plane
    return center, normal, pca


def fit_plane_pca(wire_df):
    """Alias for fit_best_fit_plane_3d (kept for readability)."""
    return fit_best_fit_plane_3d(wire_df)


def get_plane_equation(center, normal):
    """
    Return coefficients of plane equation ax + by + cz + d = 0.
    """
    a, b, c = normal
    x0, y0, z0 = center
    d = -(a * x0 + b * y0 + c * z0)
    return a, b, c, d
