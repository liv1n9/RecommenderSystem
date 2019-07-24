from cf import CF
from co_training import CoTraining
import numpy as np
from sklearn.model_selection import KFold
import time


class Evaluation:
    def __init__(self, data, k_fold=10, keep=15):
        self.data = data
        self.k_fold = k_fold
        self.keep = keep
        self.r_trains = [None] * self.k_fold
        self.r_tests = [None] * self.k_fold
        self.n_users = int(np.max(self.data[:, 0])) + 1
        self.n_items = int(np.max(self.data[:, 1])) + 1
        self.rmse = [dict() for i in range(self.k_fold)]
        self.predict_time = [dict() for i in range(self.k_fold)]
        self.training_time = [dict() for i in range(self.k_fold)]
        self.mean_rmse = dict()
        self.variance_rmse = dict()
        self.mean_predict_time = dict()
        self.mean_training_time = dict()

    def split(self):
        np.random.seed = 26051996
        np.random.shuffle(self.data)
        adj_list = [[] for i in range(self.n_users)]
        for x in range(self.data.shape[0]):
            u = int(self.data[x][0])
            i = int(self.data[x][1])
            r = float(self.data[x][2])
            adj_list[u].append((i, r))

        kf = KFold(n_splits=self.k_fold, random_state=26051996, shuffle=True)
        kf.get_n_splits(adj_list)

        for i, index in enumerate(kf.split(adj_list)):
            temp_train = []
            temp_test = []
            for u in index[0]:
                for x in range(len(adj_list[u])):
                    temp_train.append([u, adj_list[u][x][0], adj_list[u][x][1]])
            for u in index[1]:
                for x in range(len(adj_list[u])):
                    if x < self.keep:
                        temp_test.append([u, adj_list[u][x][0], adj_list[u][x][1]])
                    else:
                        temp_train.append([u, adj_list[u][x][0], adj_list[u][x][1]])
            self.r_trains[i] = np.array(temp_train)
            self.r_tests[i] = np.array(temp_test)

    def __evaluate(self, rs, name, fold):
        t1 = time.process_time_ns()
        rs.fit()
        t1 = time.process_time_ns() - t1
        self.training_time[fold][name] = t1 / 1e6

        t1 = time.process_time_ns()

        # se = 0
        # n_tests = self.r_tests[fold].shape[0]
        # for x in range(n_tests):
        #     u = int(self.r_tests[fold][x][0])
        #     i = int(self.r_tests[fold][x][1])
        #     pred = rs.predict(u, i)
        #     r = float(self.r_tests[fold][x][2])
        #     se += (pred - r) ** 2
        # self.rmse[fold][name] = np.sqrt(se / n_tests)

        n_users = int(np.max(self.r_tests[fold][:, 0])) + 1
        users = self.r_tests[fold][:, 0]
        rmse = 0.0
        total = 0
        for u in range(n_users):
            ids = np.where(users == u)[0].astype(np.int32)
            se = 0.0
            for x in ids:
                u = int(self.r_tests[fold][x][0])
                i = int(self.r_tests[fold][x][1])
                r = float(self.r_tests[fold][x][2])
                pred = rs.predict(u, i)
                se += (pred - r) ** 2
            if ids.size > 0:
                rmse += np.sqrt(se / ids.size)
                total += 1
        self.rmse[fold][name] = rmse / total

        t1 = time.process_time_ns() - t1
        self.predict_time[fold][name] = t1 / 1e6

    def evaluate(self):
        for fold in range(self.k_fold):
            print('Fold:', fold)
            rs = CF(r_train=self.r_trains[fold], user_cf=True)
            self.__evaluate(rs, 'User-KNN', fold)
            rs = CF(r_train=self.r_trains[fold], user_cf=False)
            self.__evaluate(rs, 'Item-KNN', fold)
            crs = CoTraining(r_train=self.r_trains[fold], loop=10, user_first=True)
            self.__evaluate(crs, 'User-Item-Co-Training', fold)
            crs = CoTraining(r_train=self.r_trains[fold], loop=10, user_first=False)
            self.__evaluate(crs, 'Item-User-Co-Training', fold)
            crs = CoTraining(r_train=self.r_trains[fold], loop=1000, user_first=True)
            self.__evaluate(crs, 'User-Item-Co-Training - no loop', fold)
            crs = CoTraining(r_train=self.r_trains[fold], loop=1000, user_first=False)
            self.__evaluate(crs, 'Item-User-Co-Training - no loop', fold)

        for i in range(self.k_fold):
            for name, t in self.training_time[i].items():
                self.mean_training_time[name] = self.mean_training_time.setdefault(name, 0) + t
        for name, t in self.mean_training_time.items():
            self.mean_training_time[name] = t / self.k_fold

        for i in range(self.k_fold):
            for name, t in self.predict_time[i].items():
                self.mean_predict_time[name] = self.mean_predict_time.setdefault(name, 0) + t
        for name, t in self.mean_predict_time.items():
            self.mean_predict_time[name] = t / self.k_fold

        for i in range(self.k_fold):
            for name, t in self.rmse[i].items():
                self.mean_rmse[name] = self.mean_rmse.setdefault(name, 0) + t
        for name, t in self.mean_rmse.items():
            self.mean_rmse[name] = t / self.k_fold

        for i in range(self.k_fold):
            for name, t in self.rmse[i].items():
                self.variance_rmse[name] = self.variance_rmse.setdefault(name, 0) + (t - self.mean_rmse[name]) ** 2
        for name, t in self.variance_rmse.items():
            self.variance_rmse[name] = t / self.k_fold

        print(self.mean_training_time)
        print(self.mean_predict_time)
        print(self.mean_rmse)
        print(self.variance_rmse)
