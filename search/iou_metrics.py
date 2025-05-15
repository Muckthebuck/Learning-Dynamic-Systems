import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy.linalg import inv, det
from matplotlib.patches import Polygon
from sklearn.decomposition import PCA

def visualise_and_compare_clusters(X, Y, pca_components=2, n_samples=50000, title_prefix="", is_plot_results=False):
    """For n-dimensional X and Y with k samples: X and Y should be a kxn NumPy ndarray"""
    """X: """
    def get_bounding_box(A, B):
        all_points = np.vstack([A, B])
        return all_points.min(axis=0), all_points.max(axis=0)

    def sample_uniform(mins, maxs, n_samples):
        return np.random.uniform(mins, maxs, (n_samples, len(mins)))

    def in_hull(points, delaunay):
        return delaunay.find_simplex(points) >= 0

    def convex_hull_iou_original_space(A, B, n_samples):
        hull_A = ConvexHull(A)
        hull_B = ConvexHull(B)
        del_A = Delaunay(A[hull_A.vertices])
        del_B = Delaunay(B[hull_B.vertices])
        mins, maxs = get_bounding_box(A, B)
        samples = sample_uniform(mins, maxs, n_samples)
        in_A = in_hull(samples, del_A)
        in_B = in_hull(samples, del_B)
        intersection = np.sum(in_A & in_B)
        union = np.sum(in_A | in_B)
        return intersection / union

    def mahalanobis_distance(A, B):
        mu_A, mu_B = A.mean(0), B.mean(0)
        pooled_cov = np.cov(np.vstack((A, B)).T)
        inv_cov = inv(pooled_cov)
        delta = mu_A - mu_B
        return np.sqrt(delta.T @ inv_cov @ delta)

    def bhattacharyya_distance(A, B):
        mu_A, mu_B = A.mean(0), B.mean(0)
        cov_A = np.cov(A.T)
        cov_B = np.cov(B.T)
        cov_avg = 0.5 * (cov_A + cov_B)
        inv_cov_avg = inv(cov_avg)
        term1 = 0.125 * ((mu_A - mu_B).T @ inv_cov_avg @ (mu_A - mu_B))
        term2 = 0.5 * np.log(det(cov_avg) / np.sqrt(det(cov_A) * det(cov_B)))
        return term1 + term2

    # Compute metrics in original space
    iou = convex_hull_iou_original_space(X, Y, n_samples=n_samples)
    mahal = mahalanobis_distance(X, Y)
    bhat = bhattacharyya_distance(X, Y)


    if is_plot_results:
        # PCA projection for visualisation
        pca = PCA(n_components=pca_components)
        XY_pca = pca.fit_transform(np.vstack((X, Y)))
        X_pca = XY_pca[:len(X)]
        Y_pca = XY_pca[len(X):]

        # Visualisation
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, label="Cluster X", color="blue")
        ax.scatter(Y_pca[:, 0], Y_pca[:, 1], alpha=0.4, label="Cluster Y", color="green")

        hull_X = ConvexHull(X_pca)
        hull_Y = ConvexHull(Y_pca)
        poly_X = Polygon(X_pca[hull_X.vertices], color='blue', alpha=0.2)
        poly_Y = Polygon(Y_pca[hull_Y.vertices], color='green', alpha=0.2)
        ax.add_patch(poly_X)
        ax.add_patch(poly_Y)

        ax.legend()
        ax.set_title(f"{title_prefix}Convex IoU = {iou:.3f}, Mahalanobis = {mahal:.3f}, Bhattacharyya = {bhat:.3f}")
        plt.tight_layout()
        plt.grid(True)
        plt.axis("equal")
        plt.show()

    return iou, mahal, bhat


if __name__ == "__main__":
    # Example usage with 4D data
    from sklearn.datasets import make_blobs
    X_4d, _ = make_blobs(n_samples=300, centers=[(2, 0, 2, 0)], cluster_std=1.0, random_state=0)
    Y_4d, _ = make_blobs(n_samples=300, centers=[(2, 0, 2, 0)], cluster_std=1.0, random_state=1)

    iou, mahal, bhat = visualise_and_compare_clusters(X_4d, Y_4d, title_prefix="4D (original space overlap, PCA for plot)\n")