import numpy as np


class KNNBaseline:
    def __init__(self, data, user_knn=True, ld1=3, ld2=5, k=200):
        self.matrix = data.matrix
        self.rated = data.rated
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.min_rating = data.min_rating
        self.max_rating = data.max_rating
        self.mean_rating = data.mean_rating
        self.ld1 = ld1
        self.ld2 = ld2
        self.k = k
        self.user_knn = user_knn

        self.n = self.n_users if self.user_knn else self.n_items
        self.ld3 = 25
        self.ld4 = 10

        self.baseline = None
        self.bu = None
        self.bi = None
        self.c = None
        self.s = None
        self.neighbor = [None] * self.n

    def __compute_baseline(self):
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)
        yi = np.sum(self.rated, axis=0) + self.ld3
        yu = np.sum(self.rated, axis=1) + self.ld4
        for i in range(5):
            self.bi = np.sum((self.matrix.T - self.mean_rating - self.bu) * self.rated.T, axis=1) / yi
            self.bu = np.sum((self.matrix - self.mean_rating - self.bi) * self.rated, axis=1) / yu
        self.baseline = np.add.outer(self.bu, self.bi) + self.mean_rating

    def __compute_similarity(self):
        if self.user_knn:
            diff = (self.matrix - self.baseline) * self.rated
            self.c = np.dot(self.rated, self.rated.T)
            x = np.dot(diff, diff.T)
            y = np.sqrt(np.dot(diff ** 2, self.rated.T))
            y = y * y.T
            z = self.c / (self.c + 100)
            self.s = x / (y + 1e-8) * z
        else:
            diff = (self.matrix.T - self.baseline.T) * self.rated.T
            self.c = np.dot(self.rated.T, self.rated)
            x = np.dot(diff, diff.T)
            y = np.sqrt(np.dot(diff ** 2, self.rated))
            y = y * y.T
            z = self.c / (self.c + 100)
            self.s = x / (y + 1e-8) * z

    def __compute_neighbor(self):
        for x in range(self.n):
            y = np.where(self.c[x] >= self.ld1)[0].astype(np.int32)
            y = y[y != x]
            sim = self.s[x][y]
            ids = np.argsort(sim)[-self.k:]
            self.neighbor[x] = y[ids]

    def compute(self):
        self.__compute_baseline()
        self.__compute_similarity()
        self.__compute_neighbor()

    def add(self, u, i, r):
        self.matrix[u][i] = r
        self.rated[u][i] = 1

    def predict(self, u, i):
        if self.user_knn:
            r = self.matrix[self.neighbor[u], i] - self.baseline[self.neighbor[u], i]
            s = self.s[u][self.neighbor[u]] * self.rated[self.neighbor[u], i]
        else:
            r = self.matrix[u, self.neighbor[i]] - self.baseline[u, self.neighbor[i]]
            s = self.s[i][self.neighbor[i]] * self.rated[u, self.neighbor[i]]
        result = (r * s).sum() / (np.abs(s).sum() + 1e-8) + self.baseline[u][i]
        result = max(result, self.min_rating)
        result = min(result, self.max_rating)
        return result

    def predict_group(self, x):
        if self.neighbor[x].size == 0:
            return np.array([]), np.array([])
        matrix = self.matrix if self.user_knn else self.matrix.T
        baseline = self.baseline if self.user_knn else self.baseline.T
        rated = self.rated if self.user_knn else self.rated.T
        mask = np.sum(rated[self.neighbor[x]], axis=0)
        mask = mask * (1 - rated[x])
        y = np.where(mask >= self.ld2)[0].astype(np.int32)
        s = self.s[x][self.neighbor[x]]
        rated_y = rated[self.neighbor[x]][:, y]
        ratings_y = (matrix[self.neighbor[x]][:, y] - baseline[self.neighbor[x]][:, y]) * rated_y
        results = np.dot(s, ratings_y) / (np.dot(np.abs(s), rated_y) + 1e-8) + baseline[x][y]
        results[results < self.min_rating] = self.min_rating
        results[results > self.max_rating] = self.max_rating
        return y, results
