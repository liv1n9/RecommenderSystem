import time

import numpy as np
from recommender_system.knn.knn import KNN
from recommender_system.knn.knn_co_training import KNNCoTraining

# from my_rs.knn import KNN
# from my_rs.knn_co_training import KNNCoTraining

from utility.preprocessing import Data

if __name__ == '__main__':
    # train_file = '../recommender_system/data/ua.base'
    # test_file = '../recommender_system/data/ua.test'
    train_file = '../recommender_system/data/training.txt'
    test_file = '../recommender_system/data/testing.txt'
    i_items = '../recommender_system/data/u.item'
    i_users = '../recommender_system/data/u.user'
    sep = '\t'

    data = Data(train_file, test_file, i_items, i_users, sep)
    data.process()

    rs = KNNCoTraining(loop=20, data=data, content_base=True, ld1=14, ld2=0.5, ld3=10, ld4=0.6, ld5=0.75)
    # rs = KNN(data=data, user_knn=False, content_base=True)
    t1 = time.process_time_ns()
    rs.compute()
    t1 = time.process_time_ns() - t1
    print('Training time:', t1 / 1e6)

    se = 0.0
    total = 0
    n_tests = data.r_test.shape[0]
    for j in range(n_tests):
        u = int(data.r_test[j][0])
        i = int(data.r_test[j][1])
        r = float(data.r_test[j][2])
        pred = rs.predict(u, i)
        se += (pred - r) ** 2
        total += 1

    print('RMSE:', np.sqrt(se / n_tests))
    print(total, n_tests)
