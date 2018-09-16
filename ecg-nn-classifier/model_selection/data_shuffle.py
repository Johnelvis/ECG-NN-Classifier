import random

import numpy as np

import env

# fix random seed for reproducibility
random.seed(env.SEED)


def equally_distributed(X, y, split=0.25):
    # Get all trainings labels
    lbls = np.unique(y)
    # Dataset split categories
    X_train, X_test = [], []
    y_train, y_test = [], []

    for lbl in lbls:
        idx = np.where(y == lbl)[0]
        split_size = int(split * len(idx))

        print('Found %d datasets with label %s' % (len(idx), lbl))

        # Pick some random samples from the data set
        samples = random.sample(range(1, len(idx)), split_size)

        # Assign test samples
        for sample in samples:
            X_test.append(X[idx[sample]])
            y_test.append(y[idx[sample]])

        # Drop out the picked data sets
        idx = np.delete(idx, samples)

        # Assign the rest to the train sets
        for i in idx:
            X_train.append(X[i])
            y_train.append(y[i])

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def randomly(X, y, split=0.25):
    pass
