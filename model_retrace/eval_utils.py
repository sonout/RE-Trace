import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
import time
from sklearn import svm
from tqdm import tqdm

class Dataset:
    def __init__(self, filepath, max_len):
        self.filepath = filepath
        self.max_len = max_len
        self.trajs, self.trajs_len =self.read_data(filepath, max_len)

    def read_data(self, filepath, max_len):
        trajs = []
        trajs_len = []
        with open(filepath) as f:
            for traj in f:
                traj = [int(point) for point in traj.strip().split(" ")]
                if len(traj) > max_len:
                    traj = traj[:max_len]
                    traj_len = max_len
                else:
                    traj_len = len(traj)
                    traj = traj + [0] * (max_len-traj_len)
                trajs_len.append(traj_len)
                trajs.append(traj)
        return trajs, trajs_len

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, i):
        return torch.tensor(self.trajs[i], dtype=torch.long), \
               torch.tensor(self.trajs_len[i], dtype=torch.long)


    

from scipy.spatial import distance
def execute_test_loop_k(own_set, total_set, k=1):
    start = time.time()


    # Dot product
    dot_product = np.dot(total_set, own_set.T)

    # Calculate the magnitude of the vectors
    total_set_magnitude = np.linalg.norm(total_set, axis=1)
    own_set_magnitude = np.linalg.norm(own_set, axis=1)

    # Calculate cosine similarity
    cosine_similarity = dot_product / (total_set_magnitude[:, None] * own_set_magnitude)
    k_highest_dot_products = np.partition(cosine_similarity, -k)[:, -k:]

    # Flip the scale of the dot products
    k_highest_dot_products = np.max(k_highest_dot_products) - k_highest_dot_products

    # Euklidean distance
    euclidean_distances = distance.cdist(total_set, own_set, 'euclidean')
    k_smallest_distances = np.partition(euclidean_distances, k)[:, :k]      

    # Normalize k_highest_dot_products
    #k_highest_dot_products -= np.min(k_highest_dot_products, axis=0)
    #k_highest_dot_products /= np.max(k_highest_dot_products, axis=0)

    # Normalize k_smallest_distances
    k_smallest_distances -= np.min(k_smallest_distances, axis=0)
    k_smallest_distances /= np.max(k_smallest_distances, axis=0)

    distances = np.concatenate((k_highest_dot_products, k_smallest_distances), axis=1)

    print("Distance calculation finished.")
    # Convert to DF
    end = time.time()
    train_time = end - start
    print(train_time) 
    return distances, train_time

def train_predict(distances_train, distances_test, labels):

    clf = svm.SVC(gamma='scale')
    clf.fit(distances_train, labels)

    predictions = clf.predict(distances_test)
    return predictions
