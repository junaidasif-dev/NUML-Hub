import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = load_iris()
data = iris.data[:, :2]  # Using only sepal_length and sepal_width as features

# Applying K-means clustering with 3 clusters (as there are 3 types of iris in the dataset)
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
labels = kmeans.labels_

# Scatter plot to visualize the clusters
plt.figure(figsize=(8, 6))

# Plotting points with color-coded clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k')
plt.title('K-means Clustering on Iris Dataset')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Plotting the centroids of the clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='Centroids')

plt.legend()
plt.show()
