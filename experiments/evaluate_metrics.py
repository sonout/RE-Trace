import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import random
import sys
import os

sys.path.append('../src')

#import trace_plotting
import distance_metrics as dm
from utils import execute_test_loop,train_predict_svm, eval_per_attack


def write_to_csv(result_data, filepath):
    if os.path.exists(filepath):
        result_data.to_csv(filepath, mode='a', header=False)
    else:
        result_data.to_csv(filepath)


city = "hannover"
data_path = f'/home/schestakov/projects/re-identification/data/{city}/db2'
n_jobs = 100
train_split = 0.7
result_path = 'results_metrics'

# Load Data
own_set = pickle.load(open(os.path.join(data_path, "own_set.pkl"), "rb"))
total_set = pickle.load(open(os.path.join(data_path, "total_set.pkl"), "rb"))
total_set_labels = pickle.load(open(os.path.join(data_path, "total_set_labels.pkl"), "rb"))
total_set_description = pickle.load(open(os.path.join(data_path, "total_set_description.pkl"), "rb"))

print(f'own_set {len(own_set)}')
print(f'total_set {len(total_set)}')

# Metrics to evaluate
metrics = { 
    'EDwP': dm.compare_traces_edwp,
    'Frechet': dm.compare_traces_frechet2, 
    'Hausdorf': dm.compare_traces_hausdorf, 
    'EDR': dm.compare_traces_edr,
    }

# For saved data
timestamp = datetime.now().strftime("%m.%d._%H.%M")
filepath = f"{result_path}/metric_results_{timestamp}_hannover2.csv"

from sklearn.metrics import accuracy_score, f1_score
# Loop throug all metrics and evaluate
full_eval_results = {}
for name, dist_func in metrics.items():
    print(f"Evaluating {name}.")
    print("Obtaining distances")
    distances, train_time = execute_test_loop(dist_func, own_set, total_set, n_jobs=n_jobs)
    # store distances
    pickle.dump(distances, open(f"{result_path}/distances_{name}_{timestamp}.pkl", "wb"))

    print("Training predictor")
    # Train a Prediction model and predict
    num_train_samples = int( len(total_set) * train_split )
    predictions = train_predict_svm(distances[:num_train_samples].reshape(-1, 1), distances[num_train_samples:].reshape(-1, 1), total_set_labels[:num_train_samples])

    # Evaluate overall
    accuracy = accuracy_score(total_set_labels[num_train_samples:],predictions)
    f1 = f1_score(total_set_labels[num_train_samples:], predictions)
    print(f"{name} Acc: {accuracy:.3f}   F1: {f1:.3f}")

    # Evaluate for each attack individually
    attacks_res = eval_per_attack(predictions, total_set_description[num_train_samples:], total_set_labels[num_train_samples:])

    full_eval_results[name] = attacks_res
    print("*************")

    # Lets alread append the csv here
    result_data= []
    result_data.append([name, f"{train_time}({n_jobs})", "AVG", accuracy, f1])
    for attack_name, value in attacks_res.items():
        result_data.append([name, f"{train_time}({n_jobs})", attack_name, value[0], value[1]])
    

    df = pd.DataFrame(result_data, columns=['Model', 'time (jobs)', 'Attack', 'Accuracy', 'F1'])
    write_to_csv(df, filepath)

# Convert Results to DF
#result_data= []
#for metric_name, attack_res in full_eval_results.items():
#    for attack_name, value in attack_res.items():
#        result_data.append([metric_name, attack_name, value[0], value[1]])

#df = pd.DataFrame(result_data, columns=['Model', 'Attack', 'Accuracy', 'F1'])

# Safe as CSV

#df.to_csv(f"{result_path}/metric_results_{timestamp}.csv")





