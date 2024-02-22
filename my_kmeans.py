import numpy as np

class my_KMeans():
    def __init__(self, k=5, max_iterations=100, balanced=False):
        super().__init__()
        self.k = k
        self.max_iterations = max_iterations
        self.balanced = balanced

    def fit(self, X):
        X = np.array(X)

        if self.balanced:
            centroids = self.initialize_centroids_balanced(X)
        else:
            centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iterations):
            cluster_assignments = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

            new_centroids = np.array([X[cluster_assignments == i].mean(axis=0) for i in range(self.k)])

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return cluster_assignments.tolist(), centroids.tolist()

        
    def initialize_centroids_balanced(self, X):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        step = n_samples // self.k
        centroids_indices = [indices[i*step:(i+1)*step] for i in range(self.k)]
        centroids = np.array([X[indices].mean(axis=0) for indices in centroids_indices])
        return centroids
