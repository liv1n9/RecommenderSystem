import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class Data:
    def __init__(self, r_train, r_test, i_items, i_users, sep=None, data_name='100k'):
        if sep is None:
            self.r_train = r_train.astype(np.float32)
            self.r_test = r_test.astype(np.float32)
        else:
            r_cols = ['user_id', 'item_id', 'rating', 'temp_1']
            dtype = {'user_id': np.int32, 'item_id': np.int32, 'rating': np.float32}
            self.r_train = pd.read_csv(r_train, sep=sep, names=r_cols, dtype=dtype).values[:, [0, 1, 2]]
            self.r_test = pd.read_csv(r_test, sep=sep, names=r_cols, dtype=dtype).values[:, [0, 1, 2]]
        self.i_items = i_items
        self.i_users = i_users
        self.data_name = data_name

        self.matrix = None
        self.rated = None
        self.info_items = None
        self.info_users = None
        self.n_users = None
        self.n_items = None
        self.min_rating = None
        self.max_rating = None
        self.mean_rating = None

    def __process_info(self):
        if self.data_name == '100k':
            self.info_items = pd.read_csv(self.i_items, sep='|', encoding='latin-1').values[:, 6:25]

            temp = pd.read_csv(self.i_users, sep='|', encoding='latin-1')
            col_transform = ColumnTransformer(
                [('user_sex_category', OneHotEncoder(categories='auto', drop=['M'], sparse=False), ['user_sex']),
                 ('user_job_category', OneHotEncoder(categories='auto', drop=['other'], sparse=False), ['user_job'])], )
            self.info_users = col_transform.fit_transform(temp)

    def __process_matrix(self):
        temp_1 = np.concatenate((self.r_train[:, 0], self.r_test[:, 0]))
        temp_2 = np.concatenate((self.r_train[:, 1], self.r_test[:, 1]))

        le = LabelEncoder()
        le.fit(temp_1)
        self.r_train[:, 0] = le.transform(self.r_train[:, 0])
        self.r_test[:, 0] = le.transform(self.r_test[:, 0])
        le.fit(temp_2)
        self.r_train[:, 1] = le.transform(self.r_train[:, 1])
        self.r_test[:, 1] = le.transform(self.r_test[:, 1])

        self.n_users = int(max(np.max(self.r_train[:, 0]), np.max(self.r_test[:, 0]))) + 1
        self.n_items = int(max(np.max(self.r_train[:, 1]), np.max(self.r_test[:, 1]))) + 1
        self.min_rating = np.min(self.r_train[:, 2])
        self.max_rating = np.max(self.r_train[:, 2])
        self.mean_rating = np.mean(self.r_train[:, 2])

        self.rated = coo_matrix((np.ones(self.r_train.shape[0]), (self.r_train[:, 0], self.r_train[:, 1])),
                                (self.n_users, self.n_items)).toarray()
        self.matrix = coo_matrix((self.r_train[:, 2], (self.r_train[:, 0], self.r_train[:, 1])),
                                 (self.n_users, self.n_items)).toarray()

    def process(self):
        self.__process_info()
        self.__process_matrix()
