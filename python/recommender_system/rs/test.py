import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from rs.preprocessing import Data
from rs.knn import KNN

if __name__ == '__main__':
    train_file = 'ua.base'
    test_file = 'ua.test'
    sep = '\t'
    r_cols = ['user_id', 'item_id', 'rating', 'temp_1']
    r_train = pd.read_csv(train_file, sep=sep, names=r_cols)
    r_test = pd.read_csv(test_file, sep=sep, names=r_cols)
    r_train = r_train.values.astype(np.float32)
    r_test = r_test.values.astype(np.float32)
    temp_1 = np.concatenate((r_train[:, 0], r_test[:, 0]))
    temp_2 = np.concatenate((r_train[:, 1], r_test[:, 1]))

    le = LabelEncoder()
    le.fit(temp_1)
    r_train[:, 0] = le.transform(r_train[:, 0])
    r_test[:, 0] = le.transform(r_test[:, 0])
    le.fit(temp_2)
    r_train[:, 1] = le.transform(r_train[:, 1])
    r_test[:, 1] = le.transform(r_test[:, 1])

    data = Data(r_train, r_test)
    data.process()
    rs = KNN(data, user_knn=False)
    rs.compute()

    se = 0.0
    total = 0
    for j in range(r_test.shape[0]):
        u = int(r_test[j][0])
        i = int(r_test[j][1])
        r = float(r_test[j][2])
        pred = rs.predict(u, i)
        se += (pred - r) ** 2
        total += 1
    print('RMSE:', np.sqrt(se / total))
