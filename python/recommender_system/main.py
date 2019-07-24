from evaluation import Evaluation
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    data = pd.read_csv('u.data', sep='\t', names=r_cols)
    r_data = data.values[:, [0, 1, 2]]
    temp = np.concatenate((r_data[:, 0], r_data[:, 1]))
    le = LabelEncoder()
    le.fit(temp)

    r_data[:, 0] = le.transform(r_data[:, 0])
    r_data[:, 1] = le.transform(r_data[:, 1])

    print('keep:', 10)
    evaluation = Evaluation(data=r_data, k_fold=10, keep=10)
    evaluation.split()
    evaluation.evaluate()

    print()
    print('keep:', 15)
    evaluation = Evaluation(data=r_data, k_fold=10)
    evaluation.split()
    evaluation.evaluate()

    print()
    print('keep:', 20)
    evaluation = Evaluation(data=r_data, k_fold=10, keep=20)
    evaluation.split()
    evaluation.evaluate()



