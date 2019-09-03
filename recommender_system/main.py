import pandas as pd

from utility.evaluation import Evaluation
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    data_file = '1m.data'
    sep = ' '

    r_cols = ['user_id', 'item_id', 'rating', 'temp_1']
    r_data = pd.read_csv(data_file, sep=sep, names=r_cols)
    r_data = r_data.values[:, [0, 1, 2]]

    le = LabelEncoder()
    r_data[:, 0] = le.fit_transform(r_data[:, 0])
    r_data[:, 1] = le.fit_transform(r_data[:, 1])

    evaluation = Evaluation(r_data, keep=20)
    evaluation.split()
    evaluation.evaluate()

    # evaluation = Evaluation(r_data, keep=10)
    # evaluation.split()
    # evaluation.evaluate()
    #
    # evaluation = Evaluation(r_data, keep=20)
    # evaluation.split()
    # evaluation.evaluate()
