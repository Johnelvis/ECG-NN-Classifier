from preprocessing.filters import median_filter
import numpy as np

"""
Removes the signal baseline wander caused by movements of the patient.
Applies a fast and slow median filter to substract noise signal.
"""
def normalize_signal_baseline(data, sample_rate):
    # Create a 200ms and a 600ms median filter
    th_median = median_filter(data, 200, sample_rate)
    sh_median = median_filter(th_median, 600, sample_rate)
    # Return the baseline removed signal from the original data
    data -= sh_median


"""
Normalizes the ecg signal to zero mean and a standard derivation of one.
"""
def normalize_signal_std(data):
    data -= np.mean(data) / np.std(data)


"""
Normalizes the ecg signal values to scale between zero and one.
"""
def normalize_signal_peaks(data):
    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)
    diff = max_ - min_

    for idx, val in enumerate(data):
        data[idx] = (val - min_) / diff