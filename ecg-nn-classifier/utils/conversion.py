"""
Calculates the amount of samples required to represent a given amount of time in ms.
"""
def ms_to_samples(time, sample_rate):
    return time * sample_rate // 1000


"""
Calculates the time in ms which a given amount of samples represent.
"""
def samples_to_ms(samps, sample_rate):
    return samps * 1000 // sample_rate

