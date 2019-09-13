import time

import numpy as np
from recommender_system.knn.knn import KNN
from recommender_system.knn.knn_co_training import KNNCoTraining

# from my_rs.knn import KNN
# from my_rs.knn_co_training import KNNCoTraining

from utility.preprocessing import Data

if __name__ == '__main__':
    train_file = '../recommender_system/data/ua.base'
    test_file = '../recommender_system/data/ua.test'
    # train_file = '../recommender_system/data/training.txt'
    # test_file = '../recommender_system/data/testing.txt'
    f_items = '../recommender_system/data/u.item'
    f_users = '../recommender_system/data/u.user'
    sep = '\t'

    data = Data(train_file, test_file, f_items, f_users, sep)
    data.process()

    rs = KNNCoTraining(data=data, user_knn=True, content_base=True, loop=10)
    # rs = KNN(data=data, user_knn=False, a0=14, c0=0.5, c1=1)
    # rs = KNN(data=data, user_knn=False, content_base=True)
    # rs = KNNCoTraining(data=data)
    t1 = time.process_time()
    rs.compute()
    t1 = time.process_time() - t1
    print('Training time:', t1)

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
