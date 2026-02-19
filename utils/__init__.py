import numpy as np

def get_iqr(data: np.ndarray):
    q75, q25 = np.percentile(data, [75 ,25])
    return q75 - q25

def madn(data: np.ndarray):
    return np.median(np.abs(data - np.median(data))) / 0.675