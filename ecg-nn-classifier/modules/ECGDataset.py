import os
import wfdb
import env
from preprocessing.filters import *
from preprocessing.normalization import *
from utils.multidimensional import *


class ECGDataset(object):
    def __init__(self, db, num, samples=(0, None), channels=None):
        self.db = db
        self.num = num
        self.path = os.path.join(os.getcwd(), env.BASE_DIR, num)
        self.sample_rate = 360

        self.data = wfdb.rdsamp(self.path, sampfrom=samples[0], sampto=samples[1], channels=channels).p_signals
        self.ann = wfdb.rdann(self.path, 'atr', sampfrom=samples[0], sampto=samples[1])

    def normalize(self, metric):
        # Extract ecg data stream into self containing arrays
        arrs = destruct(self.data)

        for arr in arrs:
            if metric == 'mean':
                normalize_signal_peaks(arr)

            elif metric == 'std':
                normalize_signal_std(arr)

            else:
                raise KeyError('Metric not found')

        # Convert back into PhysioBank format
        self.data = construct(arrs)

    def filter(self, freq, **kwargs):
        # Extract ecg data stream into self containing arrays
        arrs = destruct(self.data)

        for arr in arrs:
            if freq == 'low':
                low_pass_filter(arr, kwargs['cutoff'], kwargs['order'])

            elif freq == 'high':
                high_pass_filter(arr, kwargs['cutoff'], kwargs['order'])

            elif freq == 'band':
                band_pass_filter(arr, kwargs['cutoff'], kwargs['order'])

            elif freq == 'median':
                median_filter(arr, kwargs['time'])

            else:
                raise KeyError('Filter not found')

        # Convert back into PhysioBank format
        self.data = construct(arrs)

    def remove_baseline_wander(self, method='median'):
        # Extract ecg data stream into self containing arrays
        arrs = destruct(self.data)

        for arr in arrs:
            if method == 'median':
                normalize_signal_baseline(arr, self.sample_rate)

        # Convert back into PhysioBank format
        self.data = construct(arrs)
