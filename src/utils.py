import time
import numpy as np
import pandas as pd
import random

from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm
from joblib import Parallel, delayed
#from tqdm.notebook import tqdm
from tqdm import tqdm
import h5py
from scipy.spatial import distance



##################### For Evaluation ###################
def execute_test_loop(compare_traces_function, own_set, total_set, n_jobs = 20):
    start = time.time()

    distance_list = Parallel(n_jobs=n_jobs)(delayed(compare_traces_function)(own_set,  trace) for trace in tqdm(total_set))      

    # Convert to DF
    distances = np.asarray(distance_list)
    #distance_df = pd.DataFrame(np.column_stack([distances, labels]), columns =['distance', 'label']) 

    end = time.time()
    train_time = end - start
    print(train_time) 
    return distances, train_time

def execute_test_loop_k(own_set, total_set, k=1):
    start = time.time()

    euclidean_distances = distance.cdist(total_set, own_set, 'euclidean')
    k_smallest_distances = np.partition(euclidean_distances, k)[:, :k]      

    print("Distance calculation finished.")
    # Convert to DF
    end = time.time()
    train_time = end - start
    print(train_time) 
    return k_smallest_distances, train_time


from scipy.spatial import distance
def execute_test_loop_retrace(own_set, total_set, k=1):
    start = time.time()

    # Dot product
    dot_product = np.dot(total_set, own_set.T)

    # Calculate the magnitude of the vectors
    # total_set_magnitude = np.linalg.norm(total_set, axis=1)
    # own_set_magnitude = np.linalg.norm(own_set, axis=1)

    # Calculate cosine similarity
    #cosine_similarity = dot_product / (total_set_magnitude[:, None] * own_set_magnitude)
    k_highest_dot_products = np.partition(dot_product, -k)[:, -k:]

    # Flip the scale of the dot products
    #k_highest_dot_products = np.max(k_highest_dot_products) - k_highest_dot_products

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
    end = time.time()
    train_time = end - start
    print(train_time) 
    return distances, train_time



def eval_per_attack(predictions, description, labels):
    descr_df = pd.DataFrame({'attack': description, 'predictions': predictions, 'labels': labels})
    # Group, but only starting after train set
    grouped = descr_df.groupby(['attack'])

    # Iterate over groups
    res_dict = {}
    for name, group in grouped:
        ########## Evaluate Predictions #############
        accuracy = accuracy_score(group['labels'], group['predictions'])
        f1 = f1_score(group['labels'], group['predictions'])
        print(name)
        print(f"Acc: {accuracy:.3f}   F1: {f1:.3f}")
        res_dict[name] = [accuracy, f1]
    return res_dict

### SVM ####

def train_predict_svm(distances_train, distances_test, labels, clf = svm.SVC(gamma='scale')):

    clf.fit(distances_train, labels)

    predictions = clf.predict(distances_test)
    return predictions

##### MLP #####

import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model with two hidden layers
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        hidden_size1 = 2 * input_size
        hidden_size2 = hidden_size1
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Define the train and predict function using the MLP model
def train_predict_mlp(distances_train, distances_test, labels, output_size=2, epochs=100):
    # infere input size from distances_train
    input_size = distances_train.shape[1]

    clf = MLP(input_size, output_size)
    # Convert the inputs to tensors
    distances_train = torch.tensor(distances_train, dtype=torch.float)
    distances_test = torch.tensor(distances_test, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=0.01)

    # Train the model for 100 epochs
    for epoch in range(epochs):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = clf(distances_train)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    # Predict the labels for the test set
    predictions = clf(distances_test).argmax(dim=1).tolist()
    return predictions




##################### For Trajectory Processing ###################

def attack_on_set(trajectory_set, attack_function):
    return [attack_function(t).tolist() for t in trajectory_set]


def create_modified_set(set, attack_name, attack_function, is_own_set = True):
    # If its a hybrid attack we could have multiple attacks in a list
    if isinstance(attack_function, list):
        modified_set = set
        for attack in attack_function:
            modified_set = attack_on_set(modified_set, attack)
    else:
        modified_set = attack_on_set(set, attack_function)

    # Create Labels
    if is_own_set:
        # Labels is 1
        labels = np.ones(len(modified_set), dtype=bool).tolist()
    else:
        # Label is 0
        labels = np.zeros(len(modified_set), dtype=bool).tolist()

    # Create Attack Description
    attacks_descr = [attack_name] * len(modified_set)

    return modified_set, labels, attacks_descr


def create_attacked_sets(own_set, other_set, attack_name, attack_function, amount_other_attacked):
    # If its a hybrid attack we could have multiple attacks in a list
    if isinstance(attack_function, list):
        attacks_own = own_set
        attacks_other = random.sample(other_set, amount_other_attacked)
        for attack in attack_function:
            attacks_own = attack_on_set(attacks_own, attack)
            attacks_other = attack_on_set(attacks_other, attack)
    else:
        attacks_own = attack_on_set(own_set, attack_function)
        rand_sample_other_set = random.sample(other_set, amount_other_attacked)
        attacks_other = attack_on_set(rand_sample_other_set, attack_function)


    # Create Labels
    label_own = np.ones(len(attacks_own), dtype=bool).tolist()
    label_others = np.zeros(len(attacks_other), dtype=bool).tolist()

    # Create Attack Description
    attacks_descr = [attack_name] * len(attacks_own +  attacks_other)

    return attacks_own + attacks_other, label_own + label_others, attacks_descr

# Different Setting: We first create all modified trajectories 
# Later we add #NUM other trajectories, so that we can vary the amount of other trajectories we added, e.g. #NUM = 10.000, 50.000, 100.000 
def create_attacked_sets2(own_set, attack_name, attack_function):
    # If its a hybrid attack we could have multiple attacks in a list
    if isinstance(attack_function, list):
        attacks_own = own_set
        for attack in attack_function:
            attacks_own = attack_on_set(attacks_own, attack)
    else:
        attacks_own = attack_on_set(own_set, attack_function)

    # Create Labels
    label_own = np.ones(len(attacks_own), dtype=bool).tolist()

    # Create Attack Description
    attacks_descr = [attack_name] * len(attacks_own)

    return attacks_own, label_own, attacks_descr

#def create_other_attacked(other_set, attack_name, attack_function)


def read_train_file(filepath, max_traj_len, n_views=20):
        print(f"Importing file: {filepath}")
        data_list = []
        data_len = []
        with open(filepath) as f:
            multi_views = []
            multi_views_len = []
            for idx, traj in tqdm(enumerate(f)):
                if idx%n_views == 0:
                    if multi_views != []:
                        # delete the identity map
                        multi_views = multi_views[1:]
                        multi_views_len = multi_views_len[1:]
                        data_list.append(multi_views)
                        data_len.append(multi_views_len)
                    multi_views = []
                    multi_views_len = []
                traj = [int(point) for point in traj.strip().split(" ")]
                if len(traj) > max_traj_len:
                    traj = traj[:max_traj_len]
                    traj_len = max_traj_len
                else:
                    traj_len = len(traj)
                    traj = traj + [0]*(max_traj_len-traj_len)
                multi_views.append(traj)
                multi_views_len.append(traj_len)
        return data_list, data_len

def read_train_file2(self, filepath, max_traj_len):
        print(f"Importing file: {filepath}")
        trajectories = []
        trajectory_lengths = []
        with open(filepath, 'r') as file:
            reader = csv.reader(file, delimiter = ' ')
            for row in tqdm(reader):
                # Convert from String to Int
                row = [int(i) for i in row]
                # If length over limit, cut
                if len(row) > max_traj_len:
                    traj = row[:max_traj_len]
                    traj_len = max_traj_len
                # Else pad
                else:
                    traj_len = len(row)
                    traj = row + [0]*(max_traj_len-traj_len)
                trajectories.append(traj)
                trajectory_lengths.append(traj_len)
        return trajectories, trajectory_lengths



###### For using Julia scripts with .h5 files #######
def pkl2h5(trips, datapath, outputname, swap_lon_lat = True, remove_time = True):
    num_trips = len(trips)
    with h5py.File(f"{datapath}/{outputname}", "w") as f:
        i = 0
        for trip in trips:
            # Remove time axis & change lon, lat position, e.g. should be [-8.580042, 41.19175 ]
            if remove_time:
                trip = [ [x,y] for x,y,t in trip]
            if swap_lon_lat:
                trip = [ [y,x] for x,y in trip]
            
            i = i+1
            trip_length = len(trip)
            f[f"/trips/{i}"] = trip
            f[f"/timestamps/{i}"] = np.arange(trip_length) * 15.0
        f.attrs["num"] = num_trips
    print(f"Completed writing {num_trips} to {datapath}/{outputname}")


###### For using Julia scripts with .h5 files #######
def pkl2h5_wo_time(trips, datapath, outputname, swap_lon_lat = True):
    num_trips = len(trips)
    with h5py.File(f"{datapath}/{outputname}", "w") as f:
        i = 0
        for trip in trips:
            # Remove time axis & change lon, lat position, e.g. should be [-8.580042, 41.19175 ]
            if swap_lon_lat:
                trip = [ [y,x] for x,y in trip]

            i = i+1
            trip_length = len(trip)
            f[f"/trips/{i}"] = trip
            f[f"/timestamps/{i}"] = np.arange(trip_length) * 15.0
        f.attrs["num"] = num_trips
    print(f"Completed writing {num_trips} to {datapath}/{outputname}")