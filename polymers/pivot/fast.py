import numpy as np
from numba import njit

@njit
def merge(walk1, walk2) -> np.ndarray:
    """Given two walks, merge them together quickly.

    Parameters
    ----------
    walk1 (dim, N1): np.ndarray
        Left side of walk to merge. First site should be at origin.
    walk2 (dim, N2): np.ndarray
        Right side of walk to merge. First site should be at origin.

    Returns
    -------
    np.ndarray (dim, N1+N2-1)
        
    """
    walk_concat = np.zeros((walk1.shape[0], walk1.shape[1] + walk2.shape[1] - 1),
                           dtype=np.int64)
    walk_concat[:,:walk1.shape[1]] = walk1

    # add last pt onto walk2
    for i in range(walk2.shape[0]):
        walk2[i] += walk1[:,-1][i]
    walk_concat[:,walk1.shape[1]-1:] = walk2
    return walk_concat

@njit
def attempt_pivot(walk, nt, sym):
    walk = walk.T
    wt = np.zeros_like(walk)
    intersects = intersect_pivot(nt, 0, walk, wt, sym)
    if intersects == 0:
        copy_walk(wt, walk)
    return walk.T, intersects != 0

@njit
def intersect_pivot(j:int, nmax:int, w:np.ndarray, wt:np.ndarray, sym:np.ndarray) -> int:
    """
    Given a walk w, attempt to pivot at site j using symmetry matrix sym.
    If the pivot is valid, return 0. Otherwise, return the number of intersections.
    
    This uses a dictionary to keep track of sites that have been visited, and checks sites
    starting at the pivot and going outward. Using a hashmap makes this O(N), and checking
    sites outward from the pivot makes this O(N^(1-p)), where p > 0, on average.

    Parameters
    ----------
    j : int
        Index of site to pivot about
    nmax : int
        Maximum number of intersections allowed (should set to zero if inputting a valid walk)
    w : np.ndarray
        Walk to pivot
    wt : np.ndarray
        Copy of w
    sym : np.ndarray
        Symmetry matrix

    Returns
    -------
    int
        Number of intersections
    """
    
    N = len(w)
    nintersect = 0
    site_dict = dict()
    copy_site(w[j], wt[j])
    s = site_to_str(w[j])
    site_dict[s] = j + 1
    if j < N / 2:
        for jj, ii in zip(range(j - 1, -1, -1), range(j + 1, N)):
            rotate_site(sym, w[j], w[jj], wt[jj])
            s = site_to_str(wt[jj])
            if s in site_dict:
                nintersect += 1
            else:
                site_dict[s] = jj + 1

            if nintersect > nmax:
                break

            s = site_to_str(w[ii])
            if s in site_dict:
                nintersect += 1
            else:
                site_dict[s] = ii + 1
            if nintersect > nmax:
                break

        for ii in range(ii+1, N):
            s = site_to_str(w[ii])
            if s in site_dict:
                nintersect += 1
            else:
                site_dict[s] = ii + 1

            if nintersect > nmax:
                break

        if nintersect <= nmax:
            for ii in range(j, N):
                copy_site(w[ii], wt[ii])
    else:
        for jj, ii in zip(range(j + 1, N), range(j - 1, -1, -1)):
            rotate_site(sym, w[j], w[jj], wt[jj])
            s = site_to_str(wt[jj])
            if s in site_dict:
                nintersect += 1
            else:
                site_dict[s] = jj + 1
            if nintersect > nmax:
                break

            s = site_to_str(w[ii])
            if s in site_dict:
                nintersect += 1
            else:
                site_dict[s] = ii + 1
            if nintersect > nmax:
                break
            
        for ii in range(ii-1, -1, -1):
            s = site_to_str(w[ii])
            if s in site_dict:
                nintersect += 1
            else:
                site_dict[s] = ii + 1

            if nintersect > nmax:
                break

        if nintersect <= nmax:
            for ii in range(j + 1):
                copy_site(w[ii], wt[ii])

    return nintersect


@njit
def copy_site(x, y):
    """Copies site x into site y

    Parameters
    ----------
    x : list
        A site in d dimensions to copy from
    y : list
        The site to copy into
    """
    dim = len(x)
    for i in range(dim):
        y[i] = x[i]

@njit
def copy_walk(w, wcopy):
    '''copy w into wcopy'''
    N = len(w)
    for i in range(N):
        copy_site(w[i], wcopy[i])

@njit
def site_to_str(x):
    """Converts a site [0, 0] to a string "0 0"

    Parameters
    ----------
    x : list
        A site in d dimensions

    Returns
    -------
    str
        A string representation of the site to be used in a hash map
    """
    '''NOTE: x must be an array of integers. floats wont work'''
    dim = len(x)
    s = str(x[0])
    for i in range(1, dim):
        s = s + ' ' + str(x[i])
    return s

@njit
def rotate_site(sym, x0, x, y):
    '''Given a symmetry matrix, rotate site x about x0 and store in y

    y = sym @ (x-x0) + x0
    
    Parameters
    ----------
    sym (dim, dim) : rotation matrix
    x0 (dim) : pivot site
    x (dim) : site to be rotated
    y (dim) : rotated site
    '''
    dim = len(sym)
    for i in range(dim):
        y[i] = x0[i]
    for i in range(dim):
        for j in range(dim):
            y[i] += sym[(i, j)] * (x[j] - x0[j])

@njit
def is_valid_two(walk1, walk2):
    # add last pt onto walk2
    walk1 = walk1.T
    walk2 = walk2.copy().T
    for i in range(walk2.shape[1]):
        walk2[:,i] += walk1[-1][i]
    # walk2 = walk2[1:]
    n_w1 = walk1.shape[0]
    n_w2 = walk2.shape[0]

    site_dict = dict()
    
    # start from the end of walk1 and beginning of walk2
    for jj,ii in zip(range(n_w1-1,-1,-1),range(1,n_w2)):
        s = site_to_str(walk1[jj])
        if s in site_dict:
            return False
        else:
            site_dict[s] = jj + 1
            
        s = site_to_str(walk2[ii])
        if s in site_dict:
            return False
        else:
            site_dict[s] = ii + 1
            
    if n_w2-1 > n_w1:
        for ii in range(ii,n_w2):
            s = site_to_str(walk2[ii])
            if s in site_dict:
                return False
            else:
                site_dict[s] = ii + 1
    elif n_w1 > n_w2-1:
        for ii in range(jj-1,-1,-1):
            s = site_to_str(walk1[ii])
            if s in site_dict:
                return False
            else:
                site_dict[s] = ii + 1

    return True
        
        
@njit
def shift_to_origin(walk):
    """Shifts a walk's first site to the origin"""
    ndim = walk.shape[0]
    for d in range(ndim):
        walk[d,:] = walk[d,:] - walk[d,0]