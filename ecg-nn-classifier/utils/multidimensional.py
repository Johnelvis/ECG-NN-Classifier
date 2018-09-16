import numpy as np

"""
Splits ECG data from PhysioBank into multiple 1D-Arrays for each data channel available in the data.
"""
def destruct(arr):
    assert isinstance(arr, np.ndarray)

    # Create arrays for each ecg signal stream
    arrs = [np.zeros((arr.shape[0],), dtype=np.float64) for _ in range(0, arr.shape[-1])]

    # Destruct inner array, assign value from stream into appropriate array
    for o_idx, pair in enumerate(arr):
        for idx, val in enumerate(pair):
            arrs[idx][o_idx] = val

    # Return destructed data
    return arrs


"""
Converts destructed 1D-Arrays back into PhysioBank ecg data array.
"""
def construct(arrs):
    res = np.zeros((arrs[0].shape[0], len(arrs)), dtype=np.float64)

    for pos, arr in enumerate(arrs):
        for idx, val in enumerate(arr):
            res[idx][pos] = val

    return res
