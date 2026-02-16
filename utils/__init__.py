import numpy as np

def get_iqr(data):
    q75, q25 = np.percentile(data, [75 ,25])
    return q75 - q25
