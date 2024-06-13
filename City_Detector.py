import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

class kMeansClustering:
    
    def __init__(self,k=3, random_state=None):
        self.k = k
        self.centroids = None
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)

    @staticmethod
    def euclidean_dist(data_point, centroids):
        return np.sqrt(np.sum((centroids-data_point)**2, axis=1))

    def fit(self, X, max_it=100):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for i in range(max_it):
            y = []

            for data_point in X:
                distances = kMeansClustering.euclidean_dist(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)
            cluster_indices = []

            for j in range(self.k):
                cluster_indices.append(np.argwhere(y == j))

            cluster_centres = []
            for j, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centres.append(self.centroids[j])
                else:
                    cluster_centres.append(np.mean(X[indices], axis=0)[0])

            if np.max(self.centroids - np.array(cluster_centres)) < 0.001:
                break
            else:
                self.centroids = np.array(cluster_centres)

        return y, self.centroids

    def calculate_wcss(self, data_points):
        wcss = []
        for k in range(1, 11):
            self.k = k
            clusters, centroids = self.fit(data_points)
            wcss_current_cluster = 0
            for cluster in range(self.k):
                wcss_current_cluster += np.sum((data_points[clusters == cluster] - centroids[cluster]) ** 2)
            wcss.append(wcss_current_cluster)
        return wcss

    def optimal_number_of_clusters(self, wcss):
        slopes = [0]
        for i in range(1, len(wcss) - 1):
            slope_behind = (wcss[i] - wcss[i-1])
            slope_ahead = (wcss[i+1] - wcss[i])
            slopes.append(abs(slope_ahead - slope_behind))
        return slopes.index(max(slopes)) + 1

    def final_optimality(self, wcss):
        optimal = []
        for i in range(20):
            optimal.append(self.optimal_number_of_clusters(wcss))
        return np.argmax(np.bincount(optimal))

    def plot_clusters(self, X, y, centroids):
        plt.figure(figsize=(8, 6))

        for i in range(self.k):
            cluster_points = X[y == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'CITY {i+1}')

        plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='CITY CENTRES')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('SATELLITE IMAGE OF CITIES')
        plt.legend()
        plt.grid(True)
        plt.show()


img_path = "3.png"
img = Image.open(img_path)
grayscale_img = img.convert("L")
pixels = list(grayscale_img.getdata())

width, height = grayscale_img.size
img_array = np.array(pixels).reshape(height, width)

y = []
for i in range(img_array.shape[0]):
    for j in range(img_array.shape[1]):
        if img_array[i][j] != 0:
            y.append([j, 64 - i])

data_points = np.array(y)


random_state = 42

kmeans = kMeansClustering(random_state=random_state)
wcss = kmeans.calculate_wcss(data_points)
optimal_k = kmeans.final_optimality(wcss)

kmeans = kMeansClustering(k=optimal_k, random_state=random_state)
clusters, centroids = kmeans.fit(data_points)
kmeans.plot_clusters(data_points, clusters, centroids)

        #ELBOW GRAPH PLOTTING:
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-cluster sum of square(WCSS)')
plt.title('The Elbow Method showing the optimal k')
plt.grid(True)
plt.show()

        # CREATING A DATAFRAME THAT DISPLAYS THE DISTANCE BETWEEN ALL THE CENTROIDS IN A TABULAR MANNER:

distances = []
for i in range(len(centroids)):
    row = []
    for j in range(len(centroids)):
        row.append(np.linalg.norm(centroids[i] - centroids[j]))
    distances.append(row)

centroid_labels = [f"CITY {i + 1}" for i in range(len(centroids))]
dist_df = pd.DataFrame(distances, columns=centroid_labels, index=centroid_labels)

print("DISTANCES BETWEEN EACH OF THE CITIES:")
print(dist_df)
