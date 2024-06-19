import numpy as np

def calculate(list):
    if(len(list) != 9):
        raise ValueError("List must contain nine numbers.")
    else:
        matrix = np.array(list).reshape(3,3)

    calculations = {
        "mean": [matrix.mean(axis = 0).tolist(),matrix.mean(axis = 1).tolist(),np.mean(matrix).tolist()],
        "variance": [matrix.var(axis = 0).tolist(),matrix.var(axis = 1).tolist(),np.var(matrix).tolist()],
        "standard deviation": [matrix.std(axis = 0).tolist(),matrix.std(axis = 1).tolist(),np.std(matrix).tolist()],
        "max": [matrix.max(axis = 0).tolist(),matrix.max(axis = 1).tolist(),np.max(matrix).tolist()],
        "min": [matrix.min(axis = 0).tolist(),matrix.min(axis = 1).tolist(),np.min(matrix).tolist()],
        "sum": [matrix.sum(axis = 0).tolist(),matrix.sum(axis = 1).tolist(),np.sum(matrix).tolist()]
    }

    return calculations