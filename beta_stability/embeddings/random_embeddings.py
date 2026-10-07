import numpy as np
from numpy.linalg import norm 
import time
import scipy
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances_chunked
import logging


class RandomEmbeddings(object):
    def __init__(self):
        self.rng = np.random.default_rng()

    def get_score(self, id1, id2):
        return np.clip(self.rng.normal(0.5, 0.01), 0, 1)



    