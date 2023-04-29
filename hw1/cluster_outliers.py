import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

SEED = 42
np.random.seed(SEED)

def estimate_clusters(data, max_clusters=10):
    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=SEED)
        labels = kmeans.fit_predict(data)

        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters

def find_kmeans_outliers(kmeans, data, std_multiplier):
    distances = kmeans.transform(data)
    min_distances = np.min(distances, axis=1)
    threshold = np.std(min_distances) * std_multiplier
    outliers = np.where(min_distances > threshold)[0]
    return outliers

def plot_kmeans_clusters_outliers(data, kmeans, outliers, ax):
    ax.axes.set_aspect("equal")
    labels = kmeans.labels_
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, color in zip(unique_labels, colors):
        cluster_indices = np.where(labels == k)[0]
        non_outlier_indices = np.setdiff1d(cluster_indices, outliers)
        non_outlier_data = data[non_outlier_indices]
        ax.plot(non_outlier_data[:, 0], non_outlier_data[:, 1], 'o', c=tuple(color), markeredgecolor='k', markersize=14, label=f"Cluster {k + 1}")

    # outliers in black
    ax.plot(data[outliers, 0], data[outliers, 1], 'o', c="black", markeredgecolor='k', markersize=6, label="Outliers")

    ax.set_title("K-means clustering with outlier detection")
    ax.legend()

def find_gmm_outliers(gmm, data, percentile):
    means = gmm.means_
    covariances = gmm.covariances_
    n_components = gmm.n_components
    
    min_mahalanobis = np.full(data.shape[0], np.inf)

    for k in range(n_components):
        cov = covariances[k]
        mean = means[k]
        
        # compute Mahalanobis distance
        cov_inv = np.linalg.pinv(cov)
        distances = cdist(data, mean[np.newaxis, :], metric="mahalanobis", VI=cov_inv)
        min_mahalanobis = np.minimum(min_mahalanobis, distances.ravel())
    
    threshold = np.percentile(min_mahalanobis, 100 - percentile)
    outliers = np.where(min_mahalanobis > threshold)[0]
    return outliers

def plot_gmm_clusters_outliers(data, gmm, outliers, ax):
    ax.axes.set_aspect("equal")
    labels = gmm.predict(data)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, color in zip(unique_labels, colors):
        cluster_indices = np.where(labels == k)[0]
        non_outlier_indices = np.setdiff1d(cluster_indices, outliers)
        non_outlier_data = data[non_outlier_indices]
        ax.plot(non_outlier_data[:, 0], non_outlier_data[:, 1], 'o', c=tuple(color), markeredgecolor='k', markersize=14, label=f"Cluster {k + 1}")

    # outliers in black
    ax.plot(data[outliers, 0], data[outliers, 1], 'o', c="black", markeredgecolor='k', markersize=6, label="Outliers")

    ax.set_title("GMM clustering with outlier detection")
    ax.legend()

def plot_dbscan_outliers(dbscan, labels, ax):
    ax.axes.set_aspect("equal")
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels) - 1))

    outliers = np.array([]).reshape(0, 2)
    for i, k in enumerate(unique_labels):
        class_member_mask = (labels == k)

        if k == -1:
            outliers = np.concatenate((outliers, data[class_member_mask & ~core_samples_mask]), axis=0)
        else:
            xy = data[class_member_mask & core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(colors[i]), markeredgecolor='k', markersize=14, label=f"Cluster {i + 1}")

    xy = outliers
    ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor="black", markeredgecolor='k', markersize=6, label=f"Outliers")

    ax.set_title("DBSCAN clustering with outlier detection")
    ax.legend()

# create 4 clusters of data
data, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=SEED)

# create 10 additional outliers
n_outliers = 10
outliers_range = (-10, 10)
outliers_kmeans = np.random.uniform(low=outliers_range[0], high=outliers_range[1], size=(n_outliers, 2))

# merge clusters with outliers
data = np.vstack((data, outliers_kmeans))

# standardize features for better performance to zero mean and unit variance
scaler = StandardScaler()
data = scaler.fit_transform(data)

# estimate the optimal number of clusters
optimal_clusters = estimate_clusters(data)

# K-means clustering with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_clusters, init='k-means++', n_init=10, random_state=SEED)
kmeans_optimal.fit(data)
# find outliers further from the cluster centers than 3 standard deviations of the distances
outliers_kmeans = find_kmeans_outliers(kmeans_optimal, data, 3.0)

# GMM clustering with the optimal number of clusters
gmm = GaussianMixture(n_components=optimal_clusters, random_state=SEED)
gmm.fit(data)
# find outliers from the cluster centers using Mahalanobis distance and a 1 percentile threshold
outliers_gmm = find_gmm_outliers(gmm, data, 1.0)

# DBSCAN clustering
dbscan = DBSCAN()
labels = dbscan.fit_predict(data)

figure, axes = plt.subplots(1, 3, figsize=(15, 5))
plot_kmeans_clusters_outliers(data, kmeans_optimal, outliers_kmeans, axes[0])
plot_gmm_clusters_outliers(data, gmm, outliers_gmm, axes[1])
plot_dbscan_outliers(dbscan, labels, axes[2])
plt.subplots_adjust(left=0.035, bottom=0.01, right=0.99, top=0.99, wspace=0.12)
plt.savefig("outliers_anomalies/outliers", dpi=500)
plt.show()
