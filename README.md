# TinyMCMC: Simple MCMC sampling in JAX

TinyMCMC provides a set of functions for Markov chain Monte Carlo (MCMC) sampling from log-likelihood landscapes, where the normalizing constant doesn't need to be known explicitly. This library is based on [JAX](https://github.com/google/jax), and is intended to be minimal and extensible for easy experimentation.

## Quickstart

Below is a minimal example to perform Hamiltonian Monte Carlo sampling from a log-likelihood. Samples are intialized from a uniform distribution, and relaxed towards the desired log-likelihood. For best performance, a function should be constructed for the MCMC step, such that it can be accelerated with just-in-time compilation with `jax.jit`.
```python
import tinymcmc
import jax
import matplotlib.pyplot as plt

def minus_log_likelihood(x):
    return 0.5*x**2

n_steps = 10
epsilon = 0.5
L = 3

@jax.jit
def mcmc_step(key, x):
    return tinymcmc.step_hmc(key, minus_log_likelihood, x, epsilon, L)

key = jax.random.key(1)
key, key_temp = jax.random.split(key, 2)
init_samples = jax.random.uniform(key, (1000,), minval=-1, maxval=1)
samples = init_samples

for i in range(n_steps):
    key, key_temp = jax.random.split(key, 2)
    samples = mcmc_step(key_temp, samples)

plt.hist(init_samples, alpha=0.5, bins=jax.numpy.linspace(-4,4,25), label='Initial Samples')
plt.hist(samples, alpha=0.5, bins=jax.numpy.linspace(-4,4,25), label='Final Samples')
plt.xlabel('Value')
plt.ylabel('Counts')
plt.legend()
plt.show()
```
