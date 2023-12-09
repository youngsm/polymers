from functools import lru_cache
from typing import Callable
import numpy as np

from polymers.naive import naive_self_avoiding_walk
from polymers.pivot.fast import (attempt_pivot as attempt_fast_pivot,
                                       merge as fast_merge,
                                       is_valid_two as fast_is_valid_two,
                                       shift_to_origin)
from polymers.pivot.slow import (attempt_pivot as attempt_slow_pivot,
                                       merge as slow_merge,
                                       is_valid_two as slow_is_valid_two)

from polymers.random import RNG, rand_Gd

__all__ = ['dimer_pivot']

def dimer_pivot(n, 
                dim=2,
                init_method='rod',
                pivot_method='fast',
                validate_method='fast',
                merge_method='fast',):
    """
    Generates a SAW of length n by dimerization
    
    Args:
        n (int): the length of the walk
    Returns:
        (x, y) (list, list): SAW of length n
    """
    
    kwargs = dict(dim=dim,
                  init_method=init_method,
                  pivot_method=pivot_method,
                  merge_method=merge_method,
                  validate_method=validate_method)
    attempt_pivot, merge, is_valid = get_algos(pivot_method, merge_method, validate_method)
    initializer = get_init_method(init_method)

    if n <= 3:
        return initializer(n, dim, RNG)
    else:
        intersects = 1
        while intersects:
            wl = dimer_pivot(n//2, **kwargs)
            wr = dimer_pivot(n-n//2, **kwargs)

            # pivot wl
            i = RNG.integers(0, len(wl[0]))
            sym = rand_Gd(dim)
            wl_pivoted,pivoted = attempt_pivot(wl, i, sym)
            shift_to_origin(wl_pivoted)

            # pivot wr
            i = RNG.integers(0, len(wr[0]))
            sym = rand_Gd(dim)            
            wr_pivoted,pivoted = attempt_pivot(wr, i, sym)
            shift_to_origin(wr_pivoted)
            
            if is_valid(wl_pivoted, wr_pivoted):
                intersects = 0

        return merge(wl_pivoted, wr_pivoted)
        
def get_algos(pivot_method, merge_method, validate_method) -> (Callable,Callable,Callable):
    _algos = {
    'fast': (attempt_fast_pivot, fast_merge, fast_is_valid_two),
    'normal': (attempt_slow_pivot, slow_merge, slow_is_valid_two)
    }
    return _algos[pivot_method][0], _algos[merge_method][1], _algos[validate_method][2]

def get_init_method(init_method) -> callable:
    _init_methods = {
    'naive': naive_self_avoiding_walk,
    'rod': init_rod
    }
    return _init_methods[init_method]


@lru_cache(maxsize=None)
def rod_combos(n, dim):
    combos = []
    for d in range(dim):
        for pow in range(2):
            curr = np.zeros((dim, n + 1), dtype=np.int32)            
            curr[d,:] = (-1)**pow*np.arange(n + 1, dtype=np.int32)
            combos.append(curr)
    return combos

@lru_cache(maxsize=None)
def init_rod(n, dim, rng):
    """Generates a rod of length n in d dimensions,
    starting at the origin and extending in a random direction.
    """
    return rod_combos(n,dim)[0]
    
