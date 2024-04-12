import numpy as np
from scipy.spatial.distance import directed_hausdorff
from dtw import *
from fastdtw import fastdtw
import similaritymeasures


from frechetdist import frdist
from scipy.spatial.distance import euclidean
#import adapt





######## EUKLIDEAN DISTANCE (ED) #########
def compare_traces_ED(trace_encodings, trace_to_compare):
    """
    Given a list of encodings, compare them to a known encoding and get a euclidean distance
    for each comparison. The distance tells you how similar the encodings are.
    :param trace_encodings: List of encodings to compare
    :param trace_to_compare: A trace encoding to compare against
    :return: A numpy ndarray with the distance for each trace in the same order as the 'traces' array
    """
    if len(trace_encodings) == 0:
        return np.empty((0))
    
    return np.linalg.norm(trace_encodings - trace_to_compare, axis=1)



######## DYNAMIC TIME WARPING (DTW) ########

def compare_traces_dtw(known_traces, trace_to_check):
    dist_list = []
    for t in known_traces:
        distance, path = fastdtw(t, trace_to_check, dist=euclidean)
        dist_list.append(distance)
    return min(dist_list) 




def compare_traces_dtw2(known_traces, trace_to_check):
    dist_list = []
    for t in known_traces:
        dist_list.append( dtw(t, trace_to_check).normalizedDistance)
    return min(dist_list) 

"""
def compare_traces_fastdtw(known_traces, trace_to_check):
    dist_list = []
    for t in known_traces:
        dist_list.append( fastdtw(t, trace_to_check)[0])
    return min(dist_list) 
"""



######## HAUSDORF DISTANCE ########

def compare_traces_hausdorf(known_traces, trace_to_check):
    dist_list = []
    for t in known_traces:
        dist_list.append( directed_hausdorff(t, trace_to_check)[0])
    return min(dist_list) 



######## FRECHET DISTANCE ########

#def compare_traces_frechet(known_traces, trace_to_check):
#    dist_list = []
#    for t in known_traces:
#        dist = adapt.metrics.frechet_distance(t, trace_to_check)
#        dist_list.append(dist)
#    return min(dist_list)


def pad_sequences(a, b):
  # convert the sequences to numpy arrays
  a = np.array(a)
  b = np.array(b)
  # get the lengths of the sequences
  len_a = len(a)
  len_b = len(b)
  # find the maximum length
  maxlen = max(len_a, len_b)
  # create arrays of zeros with the maximum length and the same shape as the sequences
  a_padded = np.zeros((maxlen, a.shape[1]))
  b_padded = np.zeros((maxlen, b.shape[1]))
  # copy the original sequences into the padded arrays
  a_padded[:len_a] = a
  b_padded[:len_b] = b
  # replace the zeros with the last element of the original sequences
  a_padded[len_a:] = a[-1]
  b_padded[len_b:] = b[-1]
  # return the padded sequences as numpy arrays
  return a_padded, b_padded


# Cannot cope with different lengths, therefor padding is needed
def compare_traces_frechet(known_traces, trace_to_check):
    dist_list = []
    for t in known_traces:
        t, trace_to_check = pad_sequences(t, trace_to_check)
        dist_list.append( frdist(t, trace_to_check))
    return min(dist_list) 


def compare_traces_frechet2(known_traces, trace_to_check):
    dist_list = []
    for t in known_traces:
        dist_list.append( similaritymeasures.frechet_dist(t, trace_to_check))
    return min(dist_list) 




######## LONGEST COMMON SUBSEQUENCE (LCSS) ########


def compare_traces_lcss(known_traces, trace_to_check):
    dist_list = []
    for t in known_traces:
        dist_list.append( lcs(t, trace_to_check))
    return max(dist_list) 


def lcs(X, Y):
    X = np.array(X, np.float64)
    Y = np.array(Y, np.float64)

    m, _ = X.shape
    n, _ = Y.shape
    L = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif np.array_equal(X[i - 1], Y[j - 1]):
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    return int(L[m][n])




######## Edit Distance on Real Sequence (EDR) ########

def compare_traces_edr(known_traces, trace_to_check):
    dist_list = []
    for t in known_traces:
        dist_list.append( edr(t, trace_to_check))
    return min(dist_list) 

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def edr(X, Y, r=1):
    X = np.array(X, np.float64)
    Y = np.array(Y, np.float64)

    m, _ = X.shape
    n, _ = Y.shape
    D = np.zeros((m + 1, n + 1))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            #d = euclidean_distance(X[i - 1], Y[j - 1])
            d = np.linalg.norm(X[i - 1] - Y[j - 1]) # Changed as this should be faster
            D[i][j] = min(D[i - 1][j] + r, D[i][j - 1] + r, D[i - 1][j - 1] + d)
    return D[m][n]



######## edit distance with projections (EDwP) ########
def compare_traces_edwp(known_traces, trace_to_check):
    dist_list = []
    for t in known_traces:
        dist_list.append( edwp(t, trace_to_check))
    return min(dist_list)

def edwp(X,Y):
    X = np.array(X, np.float64)
    Y = np.array(Y, np.float64)
    n = len(X)
    m = len(Y)
    D = np.zeros((n+1,m+1))
    for i in range(1,n+1):
        for j in range(1,m+1):
            D[i,j] = np.linalg.norm(X[i-1]-Y[j-1])
    for i in range(2,n+1):
        for j in range(2,m+1):
            D[i,j] += min(D[i-2,j-1],D[i-1,j-2],D[i-2,j-2])
    return D[n,m]

