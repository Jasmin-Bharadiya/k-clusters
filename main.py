

"""Implement an algorithm to partition n observations into k clusters in
 which each observation belongs to the cluster with the nearest mean (the centroid).
"""

import numpy as np

def calc_assignments(centroids: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Calculates distance of each sample from `data` to each centroid and returns
    a list of which centroid is closest to each sample.

    :param centroids: K centroids of dimension N
    :param data: M samples of dimension N
    :returns: M assignments to closest centroid
    """
    distance = np.linalg.norm(data[:,np.newaxis] - centroids, axis=2)
    
    return np.argmin(distance, axis=1)


def generate_centroids(data: np.ndarray, assignments: np.ndarray, K: int) -> np.ndarray:
    """
    Generates new centroids given the data and assignments, where each centroid is
    the mean of all data points which are assigned to it.
    
    :param data: Data to generate centroids from
    :param assignments: The assignment of each data sample to the closest centroid
    :param K: number of centroids to generate
    :returns: K centroids
    """
    centroids = np.array([data[assignments == k].mean(axis=0) for k in range(K)])
    return centroids



def generate_data(mus: np.ndarray, sigmas: np.ndarray, M: int) -> np.ndarray:
    """
    :param mus: KxN mean
    :param sigma: KxM variance
    :returns: MxN samples
    """
    N = mus.shape[1]
    return np.concatenate([
        np.random.randn(M, N) * sigma + mu
        for mu, sigma in zip(mus, sigmas)
    ])


def main():
    np.random.seed(42)
    data = generate_data(
        mus=np.array([(1, 1), (20, 20), (30, 30)]),
        sigmas=np.array([(0.01, 0.01), (0.01, 0.01), (0.01, 0.01)]),
        M=5
    )

    centroids = np.array([
        (0, 0),
        (15, 15),
        (35, 35)
    ])

    assignments = calc_assignments(centroids, data)
    centroids = generate_centroids(data, assignments, 3)
    print(centroids)



if __name__ == "__main__":
    main()

