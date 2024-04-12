import pandas as pd
import numpy as np


def remove_non_significant_bits(trace, spatial_decimals=3):
    rounded_trace = np.zeros_like(trace)
    rounded_trace = np.around(trace, decimals=spatial_decimals)
    return rounded_trace

# Additive Gaussian White Noise
def add_white_noise(trace, spatial_variance=0.0018): # .0005 we had this at 0.001
    noisy_trace = np.zeros_like(trace)
    trace = np.asarray(trace)

    number_of_samples = len(trace[:, 0])
    noisy_trace[:, 0] = trace[:, 0] + np.random.normal(0, spatial_variance, number_of_samples)
    noisy_trace[:, 1] = trace[:, 1] + np.random.normal(0, spatial_variance, number_of_samples)

    return noisy_trace


def _add_noise_with_signal_to_noise_ratio(signal, signal_to_noise_ratio, indices=None):
    # Calculate signal power and convert to dB
    signal_average = np.mean(signal)
    signal_average_db = 10 * np.log10(abs(signal_average))
    # Calculate noise according to [2] then convert to watts
    noise_average_db = signal_average_db - signal_to_noise_ratio
    noise_average = 10 ** (noise_average_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    if indices is None:
        noise = np.random.normal(mean_noise, np.sqrt(noise_average), len(signal))
        return signal + noise
    else:
        noisy_signal = np.zeros_like(signal)
        noisy_signal += signal
        noisy_signal[indices] += np.random.normal(mean_noise, np.sqrt(noise_average), np.sum(indices))
        return noisy_signal

# Additive Signal to Noise Ratio (SNR)
def add_signal_noise(trace, signal_to_noise_ratio=68): # we had this at 70
    noisy_trace = np.zeros_like(trace)
    trace = np.asarray(trace)
    #print(trace.shape)
    # Noise up the original signal
    noisy_trace[:, 0] = _add_noise_with_signal_to_noise_ratio(trace[:, 0], signal_to_noise_ratio)
    noisy_trace[:, 1] = _add_noise_with_signal_to_noise_ratio(trace[:, 1], signal_to_noise_ratio)

    return noisy_trace


# Additive Outliers with SNR (AOSNR)
def add_outliers_with_signal_to_noise_ratio(trace, signal_to_noise_ratio=60, affected_percentage=.4): # was 0.05
    trace = np.asarray(trace)
    number_of_samples = trace.shape[0]
    affected_indices = np.random.choice([False, True],
                                        size=(number_of_samples,),
                                        p=[1 - affected_percentage, affected_percentage])

    noisy_trace = np.zeros_like(trace)

    # Noise up the original signal
    noisy_trace[:, 0] = _add_noise_with_signal_to_noise_ratio(trace[:, 0],
                                                              signal_to_noise_ratio,
                                                              indices=affected_indices)
    noisy_trace[:, 1] = _add_noise_with_signal_to_noise_ratio(trace[:, 1],
                                                              signal_to_noise_ratio,
                                                              indices=affected_indices)
    return noisy_trace


# Replace Random Points (RRP) -> change to downsample
# Downsample
def downsample(trace, removal_percentage=.45): # was 0.3
    trace = np.asarray(trace)
    # Do not downsample if there is less than 5 points anyway, 'couse whats the point? One point is not a trace. 
    if (len(trace)>4):
        number_of_samples = trace.shape[0]
        affected_indices = np.random.choice([True, False],
                                            size=(number_of_samples,),
                                            p=[1-removal_percentage, removal_percentage])
        return trace[affected_indices]
    else:
        return trace                                    


def replace_random_points(trace, removal_percentage=.7): #was 0.3
    trace = np.asarray(trace)
    number_of_samples = trace.shape[0]
    affected_indices = np.random.choice([True, False],
                                        size=(number_of_samples,),
                                        p=[removal_percentage, 1 - removal_percentage])
    noisy_trace = trace.copy()
    for i, replace in enumerate(affected_indices):
        if replace:
            if i == 0:
                noisy_trace[0] = noisy_trace[1]
            else:
                noisy_trace[i] = noisy_trace[i - 1]
    return noisy_trace


# Replace Random Points with Path (RRPP)
def replace_random_points_with_path(trace, removal_percentage=.7): # was 0.3
    trace = np.asarray(trace)
    number_of_samples = trace.shape[0]
    affected_indices = np.random.choice([True, False],
                                        size=(number_of_samples,),
                                        p=[removal_percentage, 1 - removal_percentage])
    affected_indices[0] = False
    affected_indices[-1] = False

    noisy_trace = trace.copy()
    first_index_replace = None
    last_skeleton_point = trace[0]
    for i, replace in enumerate(affected_indices):
        if replace and first_index_replace is None:
            first_index_replace = i
        elif not replace:
            if first_index_replace is not None:
                for replace_index in range(first_index_replace, i):
                    noisy_trace[replace_index] = (replace_index - first_index_replace +1) \
                                                 / (i - first_index_replace + 1) * (
                            trace[i] - last_skeleton_point) + last_skeleton_point

            last_skeleton_point = trace[i]
            first_index_replace = None

    return noisy_trace


# Replace Non-Skeleton Points with Path (RNSPP)
def replace_non_skeleton_points_with_path(trace, epsilon=.005):
    import rdp
    trace = np.asarray(trace)
    noisy_trace = trace.copy()

    mask = rdp.rdp(trace, epsilon, return_mask=True)

    last_skeleton_point = trace[0]
    first_index_replace = None
    for i, keep in enumerate(mask):
        if not keep and first_index_replace is None:
            first_index_replace = i
        elif keep:
            if first_index_replace is not None:
                for replace_index in range(first_index_replace, i):
                    noisy_trace[replace_index] = (replace_index - first_index_replace +1) \
                                                 / (i - first_index_replace + 1) * (
                            trace[i] - last_skeleton_point) + last_skeleton_point

            last_skeleton_point = trace[i]
            first_index_replace = None

    return noisy_trace



# TODO: Add Attacks: Resample along path, Cropping attack


#### Resampling Attack

def resample_along_path(trace):
    trace = np.asarray(trace)

    resampled_trace = []
    # Iterate over trace and resample
    for i in range(trace.shape[0]-1):
        p1 = trace[i]
        p2 = trace[i+1]

        interpol_lon =  p1[0] + (p2[0]-p1[0])/2  if p1[0] < p2[0] else p2[0] + (p1[0]-p2[0])/2
        interpol_lat = p1[1] + (p2[1]-p1[1])/2 if p1[1] > p2[1] else p2[1] + (p1[1]-p2[1])/2

        resampled_trace.append([interpol_lon,interpol_lat])
    return np.asarray(resampled_trace)


### Cropping Attack


def cropping(trace, percentage=0.45): # was 0.33
    crop_points = int(len(trace)*percentage)
    return np.asarray(trace[crop_points:])


