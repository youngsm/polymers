import numpy as np

from .utils import possible_steps
from polymers.random import RNG

def naive_self_avoiding_walk(n, dim, ntries=1):
    """
    Generates a self-avoiding walk of length n in d dimensions via rejection sampling
    
    Args:
        n (int): the length of the walk
        d (int): the number of dimensions
        rng (np.random.Generator): a numpy random number generator
    Returns:
        (x, y) (list, list): Self-avoiding walk of length n
    """
    walk = np.array([[0] for _ in range(dim)])
    candidate_steps = possible_steps(dim)      # left, up, right, down in d dimensions
    
    walk_set = set(zip(*walk))
    while True:
        if len(walk[0]) == n+1:                                # if we've reached the desired length, stop
            break

        feasible_steps = []                              # check which possible steps are valid (i.e. don't intersect)
        for step in candidate_steps:
            if tuple(walk[:,-1]+step) not in walk_set:
                feasible_steps.append(step)
            
        if len(feasible_steps) > 0:                      # if there are valid steps, take one at random
            step = feasible_steps[RNG.integers(0,len(feasible_steps))]
            walk = np.hstack([walk, (walk[:,-1]+step).reshape(-1,1)])
            walk_set.add(tuple(walk[:,-1]))
        else:                                            # if there are no valid steps, we're stuck :-( let's try again
            walk = np.array([[0] for _ in range(dim)])
    return walk
