import sys
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import symnmf

EPSILON = 1e-4  # Defines the precision for determining the convergence of centroids.
ITER_NUM = 300  # Default number of iterations if none is specified by the user.
GENERAL_ERROR = "An Error Has Occurred"


def calculate_distance(vector1, vector2):
    """
    Calculates the Euclidean distance between two vectors.

    Args:
        vector1 (list): The first vector.
        vector2 (list): The second vector.

    Returns:
        float: The Euclidean distance between the two vectors.
    """
    sum_squares = 0
    for i in range(len(vector1)):
        sum_squares += (vector1[i] - vector2[i]) ** 2
    return sum_squares ** 0.5


def get_closest_centroid_index(centroids, data_point):
    """
    Finds the index of the closest centroid to a given data point.

    Args:
        centroids (list): List of centroid vectors.
        data_point (list): The data point to find the closest centroid for.

    Returns:
        int: The index of the closest centroid.
    """
    distances = [calculate_distance(data_point, centroid) for centroid in centroids]
    return distances.index(min(distances))


def find_clusters(k, data_points):
    """
    Performs the K-means clustering algorithm on a given set of data points.

    Args:
        k (int): Number of clusters to form.
        data_points (list): The data points to cluster.

    Returns:
        list: A list of clusters, where each cluster is a list of data points.
    """
    centroids = data_points[:k]
    clusters = [[] for _ in range(k)]

    for iteration_number in range(ITER_NUM):
        epsilon_counter = 0

        for data_point in data_points:
            i = get_closest_centroid_index(centroids, data_point)
            clusters[i].append(data_point)

        for i, cluster in enumerate(clusters):
            if len(cluster) > 0:
                new_centroid = [round(dim / len(cluster), 4) for dim in map(sum, zip(*cluster))]
                if calculate_distance(centroids[i], new_centroid) < EPSILON:
                    epsilon_counter += 1
                centroids[i] = new_centroid
            else:
                epsilon_counter += 1

        if iteration_number < ITER_NUM - 1:
            res = clusters
            clusters = [[] for _ in range(k)]

        if epsilon_counter == k:
            return res

    return res


def construct_cluster_array(clusters, data_points):
    """
    Constructs an array where each element represents the cluster number to which the corresponding data point belongs.

    Args:
        clusters (list): The clusters formed by the clustering algorithm.
        data_points (list): The original data points.

    Returns:
        list: An array where each element represents the cluster number for the corresponding data point.
    """
    cluster_arr = [0] * len(data_points)
    for i, data_point in enumerate(data_points):
        break_out = False
        for j, cluster in enumerate(clusters):
            for vector in cluster:
                if data_point == vector:
                    cluster_arr[i] = j
                    break_out = True
                    break
            if break_out:
                break
    return cluster_arr


def main():
    if len(sys.argv) != 3:
        print(GENERAL_ERROR)
        quit()

    k = int(sys.argv[1])
    file_name = sys.argv[2]
    file_df = pd.read_csv(file_name, header=None)
    if k <= 1 or k >= file_df.shape[0]:
        print(GENERAL_ERROR)
        quit()

    dim = file_df.shape[1]
    data_points = file_df.values.tolist()

    n = file_df.shape[0]
    try:
        sym_mat = symnmf.get_symnmf(n, k, dim, data_points)
        sym_clusters = np.argmax(sym_mat, axis=1)
        kmeans_clusters = construct_cluster_array(find_clusters(k, data_points), data_points)
        print("nmf: {:.4f}".format(silhouette_score(data_points, sym_clusters)))
        print("kmeans: {:.4f}".format(silhouette_score(data_points, kmeans_clusters)))
    except RuntimeError:
        print(GENERAL_ERROR)
        quit()


if __name__ == "__main__":
    main()
