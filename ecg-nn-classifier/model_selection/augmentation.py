import numpy as np

import env

# fix random seed for reproducibility
np.random.seed(env.SEED)

"""
Generator to yield inputs and their labels in batches.
"""
def shift_signals(x, y, batch_size, dims, max_shift=30):
    while True:
        batch_x = np.zeros((batch_size,) + x.shape[1:dims])
        batch_y = np.zeros((batch_size,) + y.shape[1:])

        for i in range(batch_size):
            # Create random arrays
            idx = np.random.randint(0, len(x) - 1)
            shift = np.random.randint(-max_shift, max_shift)
            nx = np.roll(x[idx], shift)
            batch_x[i] = nx
            batch_y[i] = y[idx]

        yield batch_x, batch_y
