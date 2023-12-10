# polymers

### Sam Young, youngsam@stanford.edu

This is a python module that attempts to estimate the critical exponent ($\nu$) of the self-avoiding walk (SAW) using the pivot algorithm. The pivot algorithm is a Markov chain Monte Carlo (MCMC) method that generates a new walk by randomly selecting a pivot point and rotating the walk about that point. The pivot algorithm is a very efficient way to generate SAWs, and is used to generate the initial dimerization of the SAWs in this experiment.


## Installation

Run either of the following commands to install the package.

```bash
pip install git+https://github.com/youngsm/polymers.git
```

or

```bash
git clone https://github.com/youngsm/polymers
cd polymers
pip install .
```

This will install the package and the two scripts found in `bin/` to your path, meaning you can just run them by writing their filename in the terminal (see below).

## Generate initial walks via dimerization

First, generate the initial dimerization. 2- and 3-D walks with lengths $N\in[100,100000]$ are generated, and saved to `polymers/data/dimers`.

```bash
generate_dimers.py
```

## Sample observables after equilibration

Then, run the experiment. The experiment will run the pivot algorithm $10^7$ times, sampling the observables mean-squared end-to-end-distance $\langle R_e^2 \rangle$, mean-squared radius of gyration $\langle R_g^2 \rangle$, and the mean-squared distance of a monomer from its endpoints $\langle R_m^2 \rangle$ for each of the equilibrized states every $10^4$ pivots (or steps). This results in $10^3$ samples for each observable. The results are saved to `.csv` files in `polymers/data/dimers/`.

```bash
critical_exp.py
```

Of course, you'd want to make batch sizes much larger than $10^4$ pivots (and have more than $1000$ samples...), but this is just being run on a laptop. These sizes are hardcoded, but can be changed in `critical_exp.py`.

## Analyse the results

Finally, analyse the results by running `notebooks/experiment_analysis.ipynb`. The critical exponent $\nu$ is estimated by fitting the data to the approximate power law $\langle R_x^2 \rangle \sim D_x N^{2\nu}$. There are higher order corrections I'm not including here (e.g. $\Delta_1$, $\Delta_2$, ...), but this is just a quick and dirty fit.