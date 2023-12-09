import polymers.dimer as dm
import numpy as np
from tqdm import tqdm
import os
import sys

def main():
    sys.setrecursionlimit(10**6)

    basename = os.path.join(os.path.dirname(__file__), "../data/dimers/")
    dimer_lengths = np.logspace(2, 5, 20, dtype=int)

    for N in tqdm(dimer_lengths, desc=f"Generating dimers", ncols=80):
        for dim in (2,3):
            fname = basename + f"dimer_d{dim}dimer{str(N).zfill(5)}.npy"
            if os.path.exists(fname):
                continue

            walk = dm.dimer_pivot(N, dim)
            np.save(fname, walk)

if __name__ == "__main__":
    main()