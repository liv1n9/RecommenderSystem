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
    f_items = '../recommender_system/data/u.item'
    f_users = '../recommender_system/data/u.user'
    sep = '\t'

    data = Data(train_file, test_file, f_items, f_users, sep)
    data.process()

    rs = KNNCoTraining(data=data, user_knn=True, content_base=True, a0=4, loop=10)
    t1 = time.process_time()
    rs.compute()
    t1 = time.process_time() - t1
    print('Training time:', t1)

    se = 0.0
    total = 0
    n_tests = data.r_test.shape[0]
    users = data.r_test[:, 0]
    t_total = 0.0
    ev = 0.0
    for u in range(data.rating.shape[0]):
        ids = np.where(users == u)[0].astype(np.int32)
        se = 0.0
        total = 0.0
        for x in ids:
            u = int(data.r_test[x, 0])
            i = int(data.r_test[x, 1])
            r = float(data.r_test[x, 2])
            pred = rs.predict(u, i)
            se += abs(pred - r)
            total += 1
        if total > 0:
            t_total += 1
            ev += se / total

    print('EV:', ev / t_total)
