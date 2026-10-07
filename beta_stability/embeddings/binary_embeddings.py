import numpy as np
from numpy.linalg import norm 
import time
import scipy
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances_chunked
import logging


class BinaryEmbeddings(object):
    def __init__(self,
                 ids,
                 df,
                 filter_key,
                 id_key='uuid'):
        """
        Initialize BinaryEmbedding with embeddings, ids, and their corresponding labels
        
        Parameters:
        - embeddings: numpy array of embedding vectors
        - ids: dictionary mapping uuids to indices
        - labels: labels corresponding to the embeddings
        """
        self.uuids = ids
        self.ids = list(ids.keys())
        self.labels = [df.loc[df[id_key] == self.uuids[id], filter_key].values[0] for id in self.ids]
        self.num_requests = 0

    def get_score(self, id1, id2):
        """
        Calculate score between two embeddings based on their label similarity
        
        Parameters:
        - id1: first embedding id
        - id2: second embedding id
        
        Returns:
        - Score: 1 if in same class, 0 if in different classes
        """
        label1 = self.labels[self.ids.index(id1)]
        label2 = self.labels[self.ids.index(id2)]
        self.num_requests += 1
        
        return 1 if label1 == label2 else 0
    def final_print(self):
        logger = logging.getLogger("beta_stability")
        logger.info(f"Binary embeddings number of requests: {self.num_requests}")



    