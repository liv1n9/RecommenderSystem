import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

DTYPE_INT = np.int32
DTYPE_DOUBLE = np.float64


class Data:
    def __init__(self, r_train, r_test, f_items, f_users, sep=None, data_name='100k'):
        if sep is None:
            self.r_train = r_train.astype(DTYPE_DOUBLE)
            self.r_test = r_test.astype(DTYPE_DOUBLE)
        else:
            r_cols = ['user_id', 'item_id', 'rating', 'temp_1']
            dtype = {'user_id': DTYPE_INT, 'item_id': DTYPE_INT, 'rating': DTYPE_DOUBLE}
            self.r_train = pd.read_csv(r_train, sep=sep, names=r_cols, dtype=dtype).values[:, [0, 1, 2]]
            self.r_test = pd.read_csv(r_test, sep=sep, names=r_cols, dtype=dtype).values[:, [0, 1, 2]]

        self.f_items = f_items
        self.f_users = f_users
        self.data_name = data_name

    def __process_info(self):
        if self.data_name == '100k':
            self.attr_items = pd.read_csv(self.f_items, sep='|', encoding='latin-1').values[:, 6:25].astype(DTYPE_DOUBLE)

            temp = pd.read_csv(self.f_users, sep='|', encoding='latin-1')
            col_transform = ColumnTransformer(
                [('user_sex_category', OneHotEncoder(categories='auto', drop=['M'], sparse=False), ['user_sex']),
                 ('user_job_category', OneHotEncoder(categories='auto', drop=['other'], sparse=False), ['user_job'])], )
            self.attr_users = col_transform.fit_transform(temp).astype(DTYPE_DOUBLE)
        else:
            pass

    def __process_matrix(self):
        le = LabelEncoder()
        temp_1 = np.concatenate((self.r_train[:, 0], self.r_test[:, 0]))
        temp_2 = np.concatenate((self.r_train[:, 1], self.r_test[:, 1]))

        le.fit(temp_1)
        self.r_train[:, 0] = le.transform(self.r_train[:, 0])
        self.r_test[:, 0] = le.transform(self.r_test[:, 0])

        le.fit(temp_2)
        self.r_train[:, 1] = le.transform(self.r_train[:, 1])
        self.r_test[:, 1] = le.transform(self.r_test[:, 1])

        self.min_rating = np.min(self.r_train[:, 2])
        self.max_rating = np.max(self.r_train[:, 2])

        n_users = int(max(np.max(self.r_train[:, 0]), np.max(self.r_test[:, 0]))) + 1
        n_items = int(max(np.max(self.r_train[:, 1]), np.max(self.r_test[:, 1]))) + 1

        self.rating = coo_matrix((self.r_train[:, 2], (self.r_train[:, 0], self.r_train[:, 1])), (n_users, n_items)).toarray()
        self.rated = coo_matrix((np.ones(self.r_train.shape[0]), (self.r_train[:, 0], self.r_train[:, 1])), (n_users, n_items)).toarray()

    def process(self):
        self.__process_info()
        self.__process_matrix()
