import numpy as np
import matplotlib.pyplot as plt
import env
from modules.ECGDataset import ECGDataset

TARGET_SAMPLE_RATE = 360
TARGET_LABELS = ['N', 'A', 'V', '/']
TARGET_ECG_DATASETS = [
    '100', '104', '108', '113', '117', '122', '201', '207', '212', '217', '222', '231',
    '101', '105', '109', '114', '118', '123', '202', '208', '213', '219', '223', '232',
    '102', '106', '111', '115', '119', '124', '203', '209', '214', '220', '228', '233',
    '103', '107', '112', '116', '121', '200', '205', '210', '215', '221', '230', '234'
]

class AnnotationReader(object):
    def __init__(self):
        X, y = AnnotationReader._create_data_slices()
        X = AnnotationReader._pad_slices(X, sample_rate=TARGET_SAMPLE_RATE)
        X, y = AnnotationReader._filter_by_labels(X, y)

        self.X = X
        self.y = y

    @staticmethod
    def _create_data_slices():
        X, y = [], []

        for set in TARGET_ECG_DATASETS:
            cursor = 0
            ecg = ECGDataset(env.ARRITHMIA_DB_NAME, set, (0, None), channels=[0])

            # Do some data preprocessing before slicing
            ecg.filter('band', cutoff=(env.HIGH_PASS_CUTOFF, env.LOW_PASS_CUTOFF), order=10)
            ecg.remove_baseline_wander()
            ecg.normalize('std')

            samps = ecg.ann.sample
            syms = ecg.ann.symbol

            while cursor < len(samps):
                # Create a "cardiac cycle" between two R-R peaks, choose next neighbors if not exist
                if cursor == 0:
                    cr1 = (samps[cursor] - ((samps[cursor + 2] - samps[cursor + 1]) >> 1))
                    cr2 = (samps[cursor] + samps[cursor + 1]) >> 1
                elif cursor == len(samps) - 1:
                    cr1 = (samps[cursor] - ((samps[cursor] - samps[cursor - 1]) >> 1))
                    cr2 = (samps[cursor] + ((samps[cursor - 1] - samps[cursor - 2]) >> 1))
                else:
                    cr1 = (samps[cursor] - ((samps[cursor] - samps[cursor - 1]) >> 1))
                    cr2 = (samps[cursor] + samps[cursor + 1]) >> 1

                # Prevent out of bounds
                if cr1 < 0: cr1 = 0
                if cr2 >= len(ecg.data): cr2 = len(ecg.data) - 1

                X.append(ecg.data[cr1:cr2])
                y.append(syms[cursor])

                cursor += 1

        return np.array(X), np.array(y)

    # Pad ecg data slices to length of sample_rate
    @staticmethod
    def _pad_slices(data, sample_rate=360):
        # Separate array for padded data slices
        padded = []

        for idx, val in enumerate(data):
            # Amount of samples missing
            missing = sample_rate - len(val)
            # A padded result array
            res = np.zeros((sample_rate, val.shape[-1]))

            for dim in range(0, val.shape[-1]):
                # Pad left and right equally
                lh = rh = missing // 2
                # Get row per data channel
                subarr = val[:, dim]

                # Add missing samples
                if missing > 0:
                    # Compensate missing odd number, make even to get to sample_rate
                    if missing & 0x1 == 1: rh += 1
                    # Pad dimensions
                    res[:, dim] = np.pad(subarr, (lh, rh), mode='constant', constant_values=0.0)

                # Decrease longer slices to sample_rate
                elif missing < 0:
                    # Compensate missing odd number, make even to get to sample_rate
                    if missing & 0x1 == 1: rh += 1
                    # Slice out too many samples
                    res[:, dim] = subarr[-1 * (lh + 1):len(subarr) + rh - 1]

                # Samples are of correct length, forward
                else:
                    res[:, dim] = subarr

            # Overwrite data column
            padded.append(res)

        return np.array(padded)

    @staticmethod
    def _filter_by_labels(X, y):
        idxs = []

        # Get index of values where the label is not in the trainings labels
        for i, lbl in enumerate(y):
            if lbl not in TARGET_LABELS:
                idxs.append(i)

        # Delete rows where other labels have been found
        X_del = np.delete(X, idxs, axis=0)
        y_del = np.delete(y, idxs, axis=0)

        # Remove first and last beat as they might be incomplete
        X_del = np.delete(X_del, [0, -1], axis=0)
        y_del = np.delete(y_del, [0, -1], axis=0)

        return X_del, y_del
