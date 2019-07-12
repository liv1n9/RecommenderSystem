from user_cf import UserCF
import pandas as pd
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings_base = pd.read_csv('ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('ua.test', sep='\t', names=r_cols)

r_train = ratings_base.values
r_test = ratings_test.values


def runtime(f):
    duration = time.process_time_ns()
    f()
    duration = time.process_time_ns() - duration
    return duration / 1e9


rs = UserCF(r_train, r_test)
t1 = runtime(rs.fit)
print('training_time', t1)
t2 = runtime(rs.evaluate)
print('predict_time', t2)
