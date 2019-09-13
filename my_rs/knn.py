import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class KNN:
    def __init__(self, data, user_knn=True, k=30):
        self.matrix = data.rating.copy()
        self.rated = data.rated.copy()
        self.att_items = data.attr_items.copy()
        self.att_users = data.attr_users.copy()
        self.n_users = self.matrix.shape[0]
        self.n_items = self.matrix.shape[1]
        self.min_rating = data.min_rating
        self.max_rating = data.max_rating
        self.user_knn = user_knn
        self.k = k

        self.n_x = self.n_users if self.user_knn else self.n_items
        self.n_y = self.n_items if self.user_knn else self.n_users
        self.similarity = None
        self.baseline = None

    def __compute_baseline(self):
        ratings = self.matrix[self.rated == 1]
        mean_rating = ratings.mean()
        b_u = np.zeros(self.n_users)
        b_i = np.zeros(self.n_items)
        ld2 = 10
        ld3 = 20
        p_i = np.sum(self.rated, axis=0) + ld2
        p_u = np.sum(self.rated, axis=1) + ld3
        for i in range(5):
            b_i = np.sum((self.matrix.T - mean_rating - b_u) * self.rated.T, axis=1) / p_i
            b_u = np.sum((self.matrix - mean_rating - b_i) * self.rated, axis=1) / p_u
        self.baseline = np.add.outer(b_u, b_i) + mean_rating

    def __compute_similarity(self):
        if self.user_knn:
            diff = (self.matrix - self.baseline) * self.rated
            c = np.dot(self.rated, self.rated.T)
            x = np.dot(diff, diff.T)
            y = np.sqrt(np.dot(diff ** 2, self.rated.T))
            y = y * y.T
            z = c / c.shape[0]
            self.similarity = x / (y + 1e-8) * z
        else:
            diff = (self.matrix.T - self.baseline.T) * self.rated.T
            c = np.dot(self.rated.T, self.rated)
            x = np.dot(diff, diff.T)
            y = np.sqrt(np.dot(diff ** 2, self.rated))
            y = y * y.T
            z = c / c.shape[0]
            self.similarity = x / (y + 1e-8) * z
        self.similarity = np.abs(self.similarity)
        # self.similarity = cosine_similarity(self.matrix) if self.user_knn else cosine_similarity(self.matrix.T)

    def compute(self):
        self.__compute_baseline()
        self.__compute_similarity()

    def add(self, u, i, r):
        self.matrix[u][i] = r
        self.rated[u][i] = 1

    def predict(self, u, i):
        if self.rated[u][i] == 1:
            return self.matrix[u][i]
        if self.user_knn:
            rated_i = self.rated[:, i]
            u_rated_i = np.where(rated_i == 1)[0].astype(np.int32)
            sims = self.similarity[u][u_rated_i]
            ids = np.argsort(sims)[-self.k:]
            u_rated_i = u_rated_i[ids]
            sims = sims[ids]
            ratings = self.matrix[u_rated_i, i] - self.baseline[u_rated_i, i]
            result = (ratings * sims).sum() / (np.abs(sims).sum() + 1e-8) + self.baseline[u][i]
        else:
            rated_u = self.rated[u, :]
            i_rated_u = np.where(rated_u == 1)[0].astype(np.int32)
            sims = self.similarity[i][i_rated_u]
            ids = np.argsort(sims)[-self.k:]
            i_rated_u = i_rated_u[ids]
            sims = sims[ids]
            ratings = self.matrix[u, i_rated_u] - self.baseline[u, i_rated_u]
            result = (ratings * sims).sum() / (np.abs(sims).sum() + 1e-8) + self.baseline[u][i]

        result = max(result, self.min_rating)
        result = min(result, self.max_rating)
        return result

    def predict_group(self, x, m):
        neighbor_x = np.argsort(self.similarity[x])[::-1]
        neighbor_x = neighbor_x[neighbor_x != x][:50]
        sim = self.similarity[x, neighbor_x]
        matrix = self.matrix if self.user_knn else self.matrix.T
        rated = self.rated if self.user_knn else self.rated.T
        baseline = self.baseline if self.user_knn else self.baseline.T

        unrated_y = np.where(rated[x] == 0)[0]
        mask = rated[neighbor_x][:, unrated_y]
        acc = np.add.accumulate(mask)
        acc = np.where(acc > self.k, 0.0, 1.0)
        mask = mask * acc
        confident = np.dot(sim, mask)
        ids = np.argsort(confident)[-m:]
        y = unrated_y[ids]

        rating_y = matrix[neighbor_x][:, y] - baseline[neighbor_x][:, y]
        mask_y = mask[:, ids]
        predict = np.dot(sim, rating_y * mask_y) / (np.dot(np.abs(sim), mask_y) + 1e-8) + baseline[x, y]
        return y, predict

