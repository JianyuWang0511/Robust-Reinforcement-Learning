import numpy as np

def running_average(data):
    running_avg = np.cumsum(data) / np.arange(1, len(data) + 1)
    return running_avg

