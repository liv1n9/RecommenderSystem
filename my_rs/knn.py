import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class KNN:
    def __init__(self, data, user_knn=True, k=30):
        self.matrix = data.matrix.copy()
        self.rated = data.rated.copy()
        self.att_items = data.att_items.copy()
        self.att_users = data.att_users.copy()
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.min_rating = data.min_rating
        self.max_rating = data.max_rating
        self.user_knn = user_knn
        self.k = k

        self.n_x = self.n_users if self.user_knn else self.n_items
        self.n_y = self.n_items if self.user_knn else self.n_users
        self.s = None
        self.m = np.zeros(self.n_x)

    def __compute_utility(self):
        for x in range(self.n_x):
            y = np.where(self.rated[x] == 1)[0] if self.user_knn else np.where(self.rated[:, x] == 1)[
                0]
            r = self.matrix[x, y] if self.user_knn else self.matrix[y, x]
            self.m[x] = np.mean(r) if r.size > 0 else 0.0

    def __compute_similarity(self):
        if self.user_knn:
            c = np.dot(self.rated, self.rated.T)
            self.s = cosine_similarity(self.matrix) * c / self.n_items
        else:
            c = np.dot(self.rated.T, self.rated)
            self.s = cosine_similarity(self.matrix.T) * c / self.n_users

    def compute(self):
        self.__compute_utility()
        self.__compute_similarity()

    def add(self, u, i, r):
        self.matrix[u][i] = r
        self.rated[u][i] = 1

    def confident(self, u, i):
        if self.user_knn:
            u_rated_i = np.where(self.rated[:, i] == 1)[0]
            u_rated_i = u_rated_i[u_rated_i != u]
            sim = self.s[u, u_rated_i]
        else:
            i_rated_u = np.where(self.rated[u, :] == 1)[0]
            i_rated_u = i_rated_u[i_rated_u != i]
            sim = self.s[i, i_rated_u]
        sim = np.sort(sim)[-self.k:]
        return sim.mean() if sim.size > 0 else 0

    def predict(self, u, i):
        if self.rated[u][i] == 1:
            return self.matrix[u][i]
        if self.user_knn:
            rated_i = self.rated[:, i]
            u_rated_i = np.where(rated_i == 1)[0].astype(np.int32)
            sims = self.s[u][u_rated_i]
            ids = np.argsort(sims)[-self.k:]
            u_rated_i = u_rated_i[ids]
            sims = sims[ids]
            ratings = self.matrix[u_rated_i, i] - self.m[u_rated_i]
            result = (ratings * sims).sum() / (np.abs(sims).sum() + 1e-8) + self.m[u]
        else:
            rated_u = self.rated[u, :]
            i_rated_u = np.where(rated_u == 1)[0].astype(np.int32)
            sims = self.s[i][i_rated_u]
            ids = np.argsort(sims)[-self.k:]
            i_rated_u = i_rated_u[ids]
            sims = sims[ids]
            ratings = self.matrix[u, i_rated_u] - self.m[i_rated_u]
            result = (ratings * sims).sum() / (np.abs(sims).sum() + 1e-8) + self.m[i]

        result = max(result, self.min_rating)
        result = min(result, self.max_rating)
        return result

    def predict_group(self, x, m):
        neighbor_x = np.argsort(self.s[x])[::-1]
        neighbor_x = neighbor_x[neighbor_x != x][:50]
        sim = self.s[x, neighbor_x]
        matrix = self.matrix if self.user_knn else self.matrix.T
        rated = self.rated if self.user_knn else self.rated.T

        unrated_y = np.where(rated[x] == 0)[0]
        mask = rated[neighbor_x][:, unrated_y]
        acc = np.add.accumulate(mask)
        acc = np.where(acc > self.k, 0, 1)
        mask = mask * acc
        confident = np.dot(sim, mask)
        ids = np.argsort(confident)[-m:]
        y = unrated_y[ids]

        rating_y = matrix[neighbor_x][:, y] - np.array([self.m[neighbor_x]]).T
        mask_y = mask[:, ids]
        predict = np.dot(sim, rating_y * mask_y) / (np.dot(np.abs(sim), mask_y) + 1e-8) + self.m[x]
        predict[predict < self.min_rating] = self.min_rating
        predict[predict > self.max_rating] = self.max_rating
        return y, predict

