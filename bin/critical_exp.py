from polymers.pivot.fast import attempt_pivot
from polymers.analysis import Re2, Rg2, Rm2
import numpy as np
from tqdm import trange
from polymers.random import rand_Gd, RNG
import time

Rx2 = [Re2, Rg2, Rm2]

def init_log(file, keys):
    """
    Initializes the log file
    """
    with open(file, 'w') as f:
        f.write(','.join(keys) + '\n')
        
def write_log(file, values):
    with open(file, 'a') as f:
        f.write(','.join(map(lambda x: f"{x}", values)) + '\n')

def run_SAW(init_walk, batch_size, batches, log_file):
    """
    Runs a batch of SAWs and returns the squared end-to-end distance
    """
    keys = ['batch', 'acceptance', 'Re2', 'Rg2', 'Rm2', 'time_elapsed']

    init_log(log_file, keys)
    dim = init_walk.shape[0]
    walk = init_walk.copy()
    for i in range(batches):
        acceptance = 0
        tic = time.time()
        for _ in trange(batch_size, desc=f"Batch {i}/{batches}", ncols=80):
            idx = RNG.integers(0, len(walk[0]))
            sym = rand_Gd(dim)
            walk,pivoted = attempt_pivot(walk, idx, sym)
            acceptance += pivoted
        toc = time.time()
        acceptance = acceptance / batch_size
        values = [(i+1)*batch_size, acceptance, Re2(walk), Rg2(walk), Rm2(walk), toc-tic]
        write_log(log_file, values)

def main():
    from glob import glob
    import os
    
    dirname = os.path.dirname(__file__)
    
    def sort_fmt(file):
        dim = file.split('_')[-1].split('dimer')[0].lstrip('d')
        N = file.split('_')[-1].split('dimer')[1].rstrip('.npy')
        return N + ' ' + dim

    files = sorted(glob(dirname+'/../data/dimers/*.npy'), key=sort_fmt)
    print(dirname)
    for file in files:
        walk = np.load(file)
        log_file = os.path.join(dirname, f'../data/dimers/{os.path.basename(file)[:-4]}.csv')
        if os.path.exists(log_file):
            continue
        dim = walk.shape[0]
        N = walk.shape[1]
        print('d =', dim, 'N =', N)
        batch_size = int(10**4)
        batches = 1000
        run_SAW(walk, batch_size, batches, log_file)
        
if __name__ == "__main__":
    main()