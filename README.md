# TinyMCMC: Simple MCMC sampling in JAX

TinyMCMC provides a set of Python functions for Markov chain Monte Carlo (MCMC) sampling from log-likelihood landscapes, where the normalizing constant doesn't need to be known explicitly. This library is based on [JAX](https://github.com/google/jax), and is intended to be minimal and extensible for easy experimentation.

TinyMCMC uses automatic differentiation for gradient-based samplers and is CPU and GPU compatible, making it suitable for deep learning applications including with [Flax](https://github.com/google/flax). TinyMCMC includes samplers such as Random Walk Metropolis (RWM), the Metropolis-Adjusted Langevin Algorithm (MALA) and Hamiltonian Monte-Carlo (HMC), as well as support for Replica Exchange/Parallel Tempering. 

## Installation
To use the library, clone it and add it to your python path:
```bash
git clone https://github.com/MatthewQuenneville/tinymcmc.git /path/to/installation/
export PYTHONPATH=$PYTHONPATH:/path/to/installation/tinymcmc/
```
The second line can be added to your `~/.bashrc` file to permanently add it to your python path. The libraries listed in `requirements.txt` should be installed as well. The examples may have additional required libraries.

## Quickstart

Below is a minimal example to perform Hamiltonian Monte Carlo sampling from a log-likelihood, with Replica Exchange/Parallel Tempering. Samples are intialized from a single point, and relaxed towards the desired log-likelihood. For best performance, a function should be constructed for the MCMC step, such that it can be accelerated with just-in-time compilation with `jax.jit`. In this case, the sampling function contains an HMC step for the base distribution, an HMC step for the high-temperature replica (here, at 20 times the base temperature), and an exchange step between the two replicas.
```python
import tinymcmc
import jax

def minus_log_likelihood(x):
    return (20*x**2*(x**2-1))

n_steps = 50
epsilon = 0.25
L = 3
T_ratio = 20.

@jax.jit
def mcmc_and_exchange_step(key, x1, x2):
    minus_log_likelihood_high_temp = lambda x: minus_log_likelihood(x)/T_ratio
    key1, key2, key_exchange = jax.random.split(key, 3)
    x1 = tinymcmc.step_hmc(key1, minus_log_likelihood, x1, epsilon, L)
    x2 = tinymcmc.step_hmc(key2, minus_log_likelihood_high_temp, x2, epsilon, L)
    return tinymcmc.step_exchange(key_exchange, minus_log_likelihood, x1,
                                  minus_log_likelihood_high_temp, x2)
```
Having defined the sampling step, we can initialize samples (10000 chains for each replica in this case), perform the MCMC steps, and plot a histogram of the results:
```python
samples_lowT = jax.numpy.ones(10000)/jax.numpy.sqrt(2)
samples_highT = jax.numpy.ones(10000)/jax.numpy.sqrt(2)

key = jax.random.key(1)
for i in range(n_steps):
    key, key_temp = jax.random.split(key, 2)
    samples_lowT, samples_highT = mcmc_and_exchange_step(key_temp, samples_lowT, samples_highT)

import matplotlib.pyplot as plt

plt.hist(samples_lowT, bins=jax.numpy.linspace(-1.2,1.2,25))
plt.xlabel('Value')
plt.ylabel('Counts')
plt.show()
```
A more complete version of this example is provided in `examples/example_replica_exchange.py`.
