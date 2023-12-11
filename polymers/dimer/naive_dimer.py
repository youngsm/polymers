import numpy as np
from polymers.naive import naive_self_avoiding_walk
from polymers.naive.utils import is_valid
from polymers.random import RNG

__all__ = ['dimer_naive']

def dimer_naive(n, dim=2, init_n=None):
    """
    Generates a SAW of length n by dimerization
    
    Args:
        n (int): the length of the walk
    Returns:
        (x, y) (list, list): SAW of length n
    """
    if init_n is None:
        init_n = n
    
    if n <= 3:
        walk = naive_self_avoiding_walk(n, dim) #base case uses the myopic algorithm
        return walk
    else:
        not_saw = 1
        while not_saw:
            # nt = RNG.integers(n//2, n)
            # nt = n//2
            walk_1 = dimer_naive(n//2,dim,init_n)  #recursive call
            walk_2 = dimer_naive(n-n//2,dim,init_n)  #recursive call

            walk_2 = walk_2 + walk_1[:,-1].reshape(-1,1)  #translates the second walk to the end of the first one
            walk_concat = np.hstack([walk_1, walk_2[:,1:]])  #concatenates the x coordinates
            if is_valid(*walk_concat):   #if walk obtained is SAW, stop
                not_saw = 0
        return walk_concat
