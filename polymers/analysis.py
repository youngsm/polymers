from numba import njit
import numpy as np

@njit
def Re2(w):
    """squared end to end distance"""
    w = w.T.astype(np.float32)
    return np.linalg.norm((w[-1] - w[0]))**2

@njit
def Rg2(w):
    w = w.T.astype(np.float32)
    """squared radius of gyration"""
    out = 0
    N = len(w)
    for i in range(N):
        for j in range(N):
            out += np.linalg.norm((w[i] - w[j]))**2
    out /= N**2
    return out

@njit
def Rm2(w):
    w = w.T.astype(np.float32)
    out = 0
    N = len(w)
    for i in range(N):
        out += np.linalg.norm(w[i]-w[0])**2 + np.linalg.norm(w[i]-w[-1])**2
    return out / (2*N)

def Xe(w):
    w = w.T
    return w[-1]

def X(w):
    w = w.T
    return w.sum(axis=0)

@njit
def X2(w):
    w = w.T
    out = 0
    for i in range(len(w)):
        out += (w[i] * w[i]).sum()
    return out
