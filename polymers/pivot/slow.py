import numpy as np

from polymers.naive.utils import is_valid
from polymers.random import rand_Gd

def attempt_pivot(walk, nt, sym):
    """Attempt to pivot a walk at site nt
    using numpy linear algebra. This is slow!

    Parameters
    ----------
    walk : np.ndarray (dim, N)
        Walk to pivot
    nt : int
        Site to pivot about

    Returns
    -------
    np.ndarray (dim, N), int
        Pivoted walk, whether the pivot was successful
    """
    pivot_point = walk[:,nt].reshape(-1,1)
    walk_pre, walk_post = walk[:,:nt], walk[:,nt+1:]
    if nt < nt / 2:
        walk_pre = sym @ (walk_pre - pivot_point) + pivot_point
    else:
        walk_post = sym @ (walk_post - pivot_point) + pivot_point
    
    walk_pivoted = np.hstack([walk_pre, pivot_point, walk_post])
    if is_valid(*walk_pivoted):
        return walk_pivoted, 1
    return walk, 0

def merge(walk1, walk2, shift=True):
    """Given two walks, merge them together

    Parameters
    ----------
    walk1 (dim, N1): np.ndarray
        Left side of walk to merge. First site should be at origin.
    walk2 (dim, N2): np.ndarray
        Right side of walk to merge. First site should be at origin.
    shift : bool
        Whether to shift the second walk to the end of the first walk

    Returns
    -------
    np.ndarray (dim, N1+N2-1) 
    """

    if shift:
        walk2 = (walk2 + walk1[:,-1].reshape(-1,1))
    walk_concat = np.hstack([walk1, walk2[:,1:]])
    return walk_concat

def is_valid_two(walk1, walk2, shift=True):
    """Checks if two walks are valid and do not intersect"""
    # tests if two walks are valid don't intersect
    walk2_shifted = walk2
    if shift:
        walk2_shifted = (walk2 + walk1[:,-1].reshape(-1,1))[:,1:]

    return set(zip(*walk1)).isdisjoint(set(zip(*walk2_shifted)))