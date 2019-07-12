import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix


class UserCF:
    def __init__(self, r_train, r_test, k=20, sim_func=cosine_similarity):
        self.r_train = r_train
        self.r_test = r_test
        self.k = k
        self.sim_func = sim_func
        self.n_users = int(np.max(self.r_train[:, 0])) + 1
        self.n_items = int(np.max(self.r_train[:, 1])) + 1
        self.r_bar = None
        self.mu = None
        self.s = None

    def fit(self):
        users = self.r_train[:, 0]
        self.r_bar = self.r_train.copy()
        self.mu = np.zeros(self.n_users)
        for u in range(self.n_users):
            ids = np.where(users == u)[0].astype(np.int32)
            ratings = self.r_train[ids, 2]
            self.mu[u] = np.mean(ratings) if ids.size > 0 else 0
            self.r_bar[ids, 2] = ratings - self.mu[u]

        self.r_bar = coo_matrix((self.r_bar[:, 2], (self.r_bar[:, 0], self.r_bar[:, 1])), (self.n_users, self.n_items)).toarray()
        self.s = self.sim_func(self.r_bar, self.r_bar)

    def predict(self, u, i):
        ids = np.where(self.r_train[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.r_train[ids, 0]).astype(np.int32)
        sim = self.s[u, users_rated_i]
        nns = np.argsort(sim)[-self.k:]
        nearest_s = sim[nns]
        r = self.r_bar[users_rated_i[nns], i]
        return (r * nearest_s).sum() / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def evaluate(self):
        se = 0
        n_tests = self.r_test.shape[0]
        for i in range(n_tests):
            p = self.predict(self.r_test[i, 0], self.r_test[i, 1])
            se += (p - self.r_test[i][2]) ** 2

        rmse = np.sqrt(se / n_tests)
        print('RMSE', rmse)




