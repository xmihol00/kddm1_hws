import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

SEED = 1

def estimate_clusters(data, max_clusters=10):
    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=SEED)
        labels = kmeans.fit_predict(data)

        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters

data = pd.read_csv("clustering-dataset.csv")

# get the min and max values of each column
min_max = {}
for col in data.columns:
    min_max[col] = [data[col].min(), data[col].max()]

print("Min and max values of each dimension:")
print(min_max)
print()

# get the number of missing values in each column
print("Number of missing values in each dimension:")
print(data.isnull().sum())
print()

# plot histograms of each column in a single figure
_, axis = plt.subplots(2, 3, figsize=(12, 8))
axis[0, 0].hist(data['0'], bins=30)
axis[0, 0].set_title("dimension 1")
axis[0, 0].set_xlabel("Range of values")
axis[0, 0].set_ylabel("Count")

axis[0, 1].hist(data['1'], bins=30)
axis[0, 1].set_title("dimension 2")
axis[0, 1].set_xlabel("Range of values")
axis[0, 1].set_ylabel("Count")

axis[1, 0].hist(data['2'], bins=30)
axis[1, 0].set_title("dimension 3")
axis[1, 0].set_xlabel("Range of values")
axis[1, 0].set_ylabel("Count")

axis[1, 1].hist(data['3'], bins=30)
axis[1, 1].set_title("dimension 4")
axis[1, 1].set_xlabel("Range of values")
axis[1, 1].set_ylabel("Count")

axis[0, 2].hist(data['4'], bins=30)
axis[0, 2].set_title("dimension 5")
axis[0, 2].set_xlabel("Range of values")
axis[0, 2].set_ylabel("Count")

# clear the unused subplot
axis[1, 2].axis('off')

plt.subplots_adjust(left=0.06, bottom=0.07, right=0.98, top=0.95, wspace=0.25, hspace=0.27)
plt.savefig("cluster/dimension_histograms", dpi=500)
plt.show()

# use PCA to reduce the dimensionality of the data to 2 dimensions
pca = PCA(n_components=2)
data_PCA = pca.fit_transform(data)

# plot the data in 2D
plt.figure(figsize=(8, 6))
plt.scatter(data_PCA[:, 0], data_PCA[:, 1], s=10)
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.tight_layout()
plt.savefig("cluster/pca_plain_2d", dpi=500)
plt.show()

# use the silhouette score to determine the optimal number of clusters
optimal_clusters = estimate_clusters(data)
print(f"Optimal number of clusters: {optimal_clusters}")

_, axis = plt.subplots(2, 2, figsize=(12, 10))

# map labels to different colors
colors = {-1: "black", 0: "red", 1: "blue", 2: "green", 3: "orange", 4: "magenta", 5: "yellow"}

# K-means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=SEED)
kmeans.fit(data)

# print number of points in each cluster
print("K-means number of points in each cluster:")
print(pd.Series(kmeans.labels_).value_counts())
print("colors:", [f"{label}: {colors[label]}" for label in pd.Series(kmeans.labels_).value_counts().index])
print()

labels = [colors[label] for label in kmeans.labels_]

# plot the data in 2D with the clusters colored
axis[0, 0].scatter(data_PCA[:, 0], data_PCA[:, 1], c=labels, s=10)
axis[0, 0].set_title("K-means")
axis[0, 0].set_xlabel("1st principal component")
axis[0, 0].set_ylabel("2nd principal component")

# GMM clustering with 4 clusters
gmm = GaussianMixture(n_components=4, random_state=SEED)
gmm.fit(data)
prediction = gmm.predict(data)

# print number of points in each cluster
print("GMM number of points in each cluster:")
print(pd.Series(prediction).value_counts())
print("colors:", [f"{label}: {colors[label]}" for label in pd.Series(prediction).value_counts().index])
print()

labels = [colors[label] for label in gmm.predict(data)]

# plot the data in 2D with the clusters colored
axis[0, 1].scatter(data_PCA[:, 0], data_PCA[:, 1], c=labels, s=10)
axis[0, 1].set_title("GMMs")
axis[0, 1].set_xlabel("1st principal component")
axis[0, 1].set_ylabel("2nd principal component")

# DBSCAN clustering with 4 clusters
dbscan = DBSCAN(eps=6.5, min_samples=10, metric="euclidean")
dbscan.fit(data)

# print number of points in each cluster
print("DBSCAN number of points in each cluster:")
print(pd.Series(dbscan.labels_).value_counts())
print("colors:", [f"{label}: {colors[label]}" for label in pd.Series(dbscan.labels_).value_counts().index])
print()

labels = [colors[label] for label in dbscan.labels_]

# plot the data in 2D with the clusters colored
axis[1, 0].scatter(data_PCA[:, 0], data_PCA[:, 1], c=labels, s=10)
# selected = data_PCA[dbscan.labels_ == 0]
# axis[1, 0].scatter(selected[:, 0], selected[:, 1], c="red", s=10)
axis[1, 0].set_title("DBSCAN")
axis[1, 0].set_xlabel("1st principal component")
axis[1, 0].set_ylabel("2nd principal component")

hierarchical_cluster = AgglomerativeClustering(n_clusters=4, metric="euclidean")
prediction = hierarchical_cluster.fit_predict(data)

# print number of points in each cluster
print("Hierarchical number of points in each cluster:")
print(pd.Series(prediction).value_counts())
print("colors:", [f"{label}: {colors[label]}" for label in pd.Series(prediction).value_counts().index])
print()

labels = [colors[label] for label in prediction]

# plot the data in 2D with the clusters colored
axis[1, 1].scatter(data_PCA[:, 0], data_PCA[:, 1], c=labels, s=10)
axis[1, 1].set_title("Agglomerative Clustering")
axis[1, 1].set_xlabel("1st principal component")
axis[1, 1].set_ylabel("2nd principal component")

plt.subplots_adjust(left=0.06, bottom=0.05, right=0.98, top=0.97, wspace=0.2, hspace=0.2)
plt.savefig("cluster/cluster_comparison", dpi=500)
plt.show()

if False:
    # cluster using K-means with 4 clusters
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=SEED)
    kmeans.fit(data_PCA[:, :2])

    # plot the data in 2D with the clusters colored
    plt.scatter(data_PCA[:, 0], data_PCA[:, 1], c=kmeans.labels_, s=10)
    plt.title("Data reduced to 2 dimensions using PCA with 4 clusters")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.savefig("cluster/4_clusters_colored_PCA", dpi=500)
    plt.show()

    # cluster PCA data using GMM with 4 clusters
    gmm = GaussianMixture(n_components=4, random_state=SEED)
    gmm.fit(data_PCA[:, :2])

    # plot the data in 2D with the clusters colored
    plt.scatter(data_PCA[:, 0], data_PCA[:, 1], c=gmm.predict(data_PCA), s=10)
    plt.title("Data reduced to 2 dimensions using PCA with 4 clusters")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.savefig("cluster/4_clusters_colored_PCA_gmm", dpi=500)
    plt.show()

    # cluster PCA data using DBSCAN with 4 clusters
    dbscan = DBSCAN()
    dbscan.fit(data_PCA[:, :2])

    # plot the data in 2D with the clusters colored
    plt.scatter(data_PCA[:, 0], data_PCA[:, 1], c=dbscan.labels_, s=10)
    plt.title("Data reduced to 2 dimensions using PCA with 4 clusters")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.savefig("cluster/4_clusters_colored_PCA_dbscan", dpi=500)
    plt.show()

    # cluster PCA data using OPTICS with 4 clusters
    optics = OPTICS(min_samples=10)
    optics.fit(data_PCA[:, :2])

    # plot the data in 2D with the clusters colored
    plt.scatter(data_PCA[:, 0], data_PCA[:, 1], c=optics.labels_, s=10)
    plt.title("Data reduced to 2 dimensions using PCA with 4 clusters")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.savefig("cluster/4_clusters_colored_PCA_optics", dpi=500)
    plt.show()
