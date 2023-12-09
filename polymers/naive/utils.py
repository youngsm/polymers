import numpy as np
from numba import njit

@njit
def possible_steps(dim=2):
    """returns a list of all possible steps of length 1 in d dimensions
    
    e.g.,
    ```python
    >>> gen_possible_steps(2)
    array([[-1.,  0.],
           [ 0., -1.],
           [ 0.,  1.],
           [ 1.,  0.]])
    ```
    """
    out = np.zeros((2*dim, dim))
    out[:dim, :] = -np.eye(dim, dtype=np.int32)
    out[dim:, :] = np.eye(dim, dtype=np.int32)[::-1]
    return out

def is_valid(*dims):
    return len(dims[0]) == len(set(zip(*dims)))
