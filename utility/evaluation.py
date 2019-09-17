import time
import numpy as np
from sklearn.model_selection import KFold
from my_rs.knn import KNN

from recommender_system.knn.knn_co_training import KNNCoTraining
from utility.preprocessing import Data

f_items = './data/u.item'
f_users = './data/u.user'


class Evaluation:
    def __init__(self, r_data, k_fold=10, keep=10):
        self.r_data = r_data
        self.k_fold = k_fold
        self.keep = keep

        self.r_trains = [None] * self.k_fold
        self.r_tests = [None] * self.k_fold
        self.n_users = int(np.max(self.r_data[:, 0])) + 1
        self.n_items = int(np.max(self.r_data[:, 1])) + 1

        self.predict_time = [dict() for i in range(self.k_fold)]
        self.mean_predict_time = dict()
        self.training_time = [dict() for i in range(self.k_fold)]
        self.mean_training_time = dict()
        self.loop = [dict() for i in range(self.k_fold)]
        self.mean_loop = dict()
        self.pred_percent = [dict() for i in range(self.k_fold)]
        self.mean_pred_percent = dict()
        self.rmse = [dict() for i in range(self.k_fold)]
        self.mean_rmse = dict()
        self.variance_rmse = dict()

    def split(self):
        np.random.seed = 26051996
        np.random.shuffle(self.r_data)
        adj_list = [None] * self.n_users
        users = self.r_data[:, 0]
        for u in range(self.n_users):
            ids = np.where(users == u)[0].astype(np.int32)
            adj_list[u] = self.r_data[ids][:, [1, 2]]
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
                        temp_train.append([u, adj_list[u][x][0], adj_list[u][x][1]])
                    else:
                        temp_test.append([u, adj_list[u][x][0], adj_list[u][x][1]])
            self.r_trains[i] = np.array(temp_train)
            self.r_tests[i] = np.array(temp_test)

    def __evaluate(self, rs, name, fold, co_training=False):
        t1 = time.process_time()
        rs.compute()
        t1 = time.process_time() - t1
        if co_training:
            self.loop[fold][name] = rs.t
        self.training_time[fold][name] = t1

        t1 = time.process_time()
        n_users = int(np.max(self.r_tests[fold][:, 0])) + 1
        users = self.r_tests[fold][:, 0]
        rmse = 0.0
        t_total = 0.0
        for u in range(n_users):
            ids = np.where(users == u)[0].astype(np.int32)
            se = 0.0
            total = 0.0
            for x in ids:
                u = int(self.r_tests[fold][x][0])
                i = int(self.r_tests[fold][x][1])
                r = float(self.r_tests[fold][x][2])
                pred = rs.predict(u, i)
                se += abs(pred - r)
                total += 1
            if total > 0:
                rmse += se / total
                t_total += 1
        self.rmse[fold][name] = rmse / t_total
        if co_training:
            self.pred_percent[fold][name] = rs.total / (rs.data.rating.shape[0] * rs.data.rating.shape[1])
        t1 = time.process_time() - t1
        self.predict_time[fold][name] = t1

    def evaluate(self):
        print('keep:', self.keep)
        for fold in range(self.k_fold):
            print('Fold:', fold)
            if self.keep == 5:
                l1 = 4
            elif self.keep == 10:
                l1 = 7
            else:
                l1 = 14
            data = Data(self.r_trains[fold], self.r_tests[fold], f_items, f_users)
            data.process()
            rs = KNN(data, user_knn=True)
            self.__evaluate(rs, 'User-KNN', fold)

            data = Data(self.r_trains[fold], self.r_tests[fold], f_items, f_users)
            data.process()
            rs = KNN(data, user_knn=False)
            self.__evaluate(rs, 'Item-KNN', fold)

            data = Data(self.r_trains[fold], self.r_tests[fold], f_items, f_users)
            data.process()
            rs = KNNCoTraining(data, user_knn=False, content_base=False, a0=l1)
            self.__evaluate(rs, 'Co-training_U-I_' + str(l1), fold, co_training=True)

            data = Data(self.r_trains[fold], self.r_tests[fold], f_items, f_users)
            data.process()
            rs = KNNCoTraining(data, user_knn=True, content_base=False, a0=l1)
            self.__evaluate(rs, 'Co-training_I-U_' + str(l1), fold, co_training=True)

            data = Data(self.r_trains[fold], self.r_tests[fold], f_items, f_users)
            data.process()
            rs = KNNCoTraining(data, user_knn=False, content_base=True, a0=l1)
            self.__evaluate(rs, 'Co-training_U-I_Content' + str(l1), fold, co_training=True)

            data = Data(self.r_trains[fold], self.r_tests[fold], f_items, f_users)
            data.process()
            rs = KNNCoTraining(data, user_knn=True, content_base=True, a0=l1)
            self.__evaluate(rs, 'Co-training_I-U_Content' + str(l1), fold, co_training=True)

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

        for i in range(self.k_fold):
            for name, t in self.loop[i].items():
                self.mean_loop[name] = self.mean_loop.setdefault(name, 0) + t
        for name, t in self.mean_loop.items():
            self.mean_loop[name] = t / self.k_fold

        for i in range(self.k_fold):
            for name, t in self.pred_percent[i].items():
                self.mean_pred_percent[name] = self.mean_pred_percent.setdefault(name, 0) + t
        for name, t in self.mean_pred_percent.items():
            self.mean_pred_percent[name] = t / self.k_fold

        f = open('evaluation_' + str(self.keep) + '.txt', 'w+')
        print(self.mean_training_time, file=f)
        print(self.mean_predict_time, file=f)
        print(self.mean_loop, file=f)
        print(self.mean_pred_percent, file=f)
        print(self.mean_rmse, file=f)
        print(self.variance_rmse, file=f)
        f.close()
