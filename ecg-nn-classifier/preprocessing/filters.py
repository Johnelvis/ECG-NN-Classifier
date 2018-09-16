from scipy.signal import medfilt, butter, lfilter

"""
Low pass filters the ecg signal to remove high frequency noise like electromagnetic interference.
"""
def low_pass_filter(data, cutoff, order, sample_rate=360):
    # Nyquist frequency
    ny = sample_rate >> 1
    cut = cutoff / ny

    # Calculate filter coefficients using butterworth filter
    b, a = butter(order, cut, analog=False, btype="low")

    # Return the filtered signal using linear filter
    return lfilter(b, a, data)


"""
Band pass filters the ecg signal to remove low frequency noise like baseline drift
and high frequency noise like electromagnetic interference.
"""
def band_pass_filter(data, cutoff, order, sample_rate=360):
    # Nyquist frequency
    ny = sample_rate >> 1
    cut = [c / ny for c in cutoff]

    # Calculate filter coefficients using butterworth filter
    b, a = butter(order, cut, analog=False, btype="band")

    # Return the filtered signal using linear filter
    return lfilter(b, a, data)


"""
High pass filters the ecg signal to remove low frequency noise like baseline drift.
"""
def high_pass_filter(data, cutoff, order, sample_rate=360):
    # Nyquist frequency
    ny = sample_rate >> 1
    cut = cutoff / ny

    # Calculate filter coefficients using butterworth filter
    b, a = butter(order, cut, analog=False, btype="high")

    # Return the filtered signal using linear filter
    return lfilter(b, a, data)


"""
Applies a median filter on a dataset where its width is defined in ms.
"""
def median_filter(data, time, sample_rate=360):
    window_size = int(time * sample_rate / 1000)
    # Floor size by 1 if even, as window size must be odd
    if window_size & 0x1 == 0: window_size -= 1
    # Return the median filtered signal
    return medfilt(data, window_size)
