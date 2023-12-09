import numpy as np
from numba import njit
from functools import lru_cache

RNG = np.random.default_rng(1)

def set_seed(seed):
    global RNG
    RNG = np.random.default_rng(seed)
    
@njit
def random_integer_log(a, b, rng):
    # Sampling an integer based on the defined probabilities
    p = np.empty(b - a, dtype=np.float32)
    for i in range(a, b):
        p[i] = np.log(1 + 1/(i - a + 1)) / np.log(b - a + 1)
    return np.arange(a,b, dtype=np.int32)[np.searchsorted(np.cumsum(p), rng.random(), side="right")]

def random_transform(dim):
    """Randomly sample an orthogonal matrix from the group G_d of all
    possible transformations of d dimensions.
    
    This is done by starting with the identity matrix and
    randomly swapping rows and columns, then randomly
    flipping the sign of one of the rows.

    Parameters
    ----------
    dim : int, optional
        Dimenion, by default 3
    rng : np.random.RandomGenerator, optional
        Random generator to use, by default np.random.default_rng(1)
 
    Returns
    -------
    np.ndararay
        A random transformation matrix of shape (dim, dim)
    """
    mx = np.eye(dim).astype(int)
    for i in range(dim):
        ival = int((i+1)*RNG.random())
        
        # swap rows
        for ii in range(dim):
            mx[i, ii] = mx[ival, ii]
            mx[ival, ii] = 0
        
        if RNG.random()<0.5:
            mx[ival, i] = -1
        else:
            mx[ival, i] = 1      
    return mx

@lru_cache(maxsize=None)
def Gd(dim):
    # this is a hack because I don't feel like implementing G_d properly
    t = np.unique([random_transform(dim) for _ in range(10000)], axis=0)
    t = [t for t in t if not np.allclose(t, np.eye(dim))]
    return t

def rand_Gd(dim):
    """Randomly sample a matrix from the group G_d of all
    possible transformations of d dimensions, not including the identity."""
    return Gd(dim)[RNG.integers(0, len(Gd(dim)))]