import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def project_wire_to_2d_plane(wire_df):
    """
    Project 3D wire points onto 2D best-fit plane using PCA.
    """
    coords = wire_df[['x', 'y', 'z']].dropna().values
    center = coords.mean(axis=0)
    coords_centered = coords - center

    pca = PCA(n_components=2)
    projected = pca.fit_transform(coords_centered)

    projected_df = pd.DataFrame(projected, columns=['u', 'v'])
    return projected_df, pca, center
