import pandas as pd

from rs.evaluation import Evaluation
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    # File data MovieLens 100K
    data_file = 'u.data'
    sep = '\t'

    # Đọc dữ liệu
    r_cols = ['user_id', 'item_id', 'rating', 'temp_1']
    r_data = pd.read_csv(data_file, sep=sep, names=r_cols)
    r_data = r_data.values[:, [0, 1, 2]]

    # Rời rạc hoá user_id và item_id về miền giá trị [0, number_of_user) và [0, number_of_items)
    le = LabelEncoder()
    r_data[:, 0] = le.fit_transform(r_data[:, 0])
    r_data[:, 1] = le.fit_transform(r_data[:, 1])

    # Đánh giá dữ liệu
    evaluation = Evaluation(r_data)
    evaluation.split()
    evaluation.evaluate()
