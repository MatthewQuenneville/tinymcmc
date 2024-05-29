# JAXmcmc: Simple MCMC sampling in JAX

JAXmcmc provides a set of functions for MCMC sampling from log-likelihood landscapes, where the normalizing constant doesn't need to be known explicitly. 

## Quickstart

Below is a minimal example to perform Hamiltonian Monte Carlo sampling from a log-likelihood. For best performance, a function should be constructed for the MCMC step, such that it can be accelerated with just-in-time compilation with `jax.jit`.
```python
import jaxmcmc
import jax
import matplotlib.pyplot as plt

def minus_log_likelihood(x):
    return 0.5*x**2

@jax.jit
def mcmc_step(key, x):
    return jaxmcmc.step_hmc(key, minus_log_likelihood, x, 0.5, 3)

key = jax.random.key(1)
key, key_temp = jax.random.split(key, 2)
init_samples = jax.random.uniform(key, (1000,), minval=-1, maxval=1)
samples = init_samples

for i in range(10):
    key, key_temp = jax.random.split(key, 2)
    samples = mcmc_step(key_temp, samples)

plt.hist(init_samples, alpha=0.5, bins=jax.numpy.linspace(-4,4,25), label='Initial Samples')
plt.hist(samples, alpha=0.5, bins=jax.numpy.linspace(-4,4,25), label='Final Samples')
plt.xlabel('Value')
plt.ylabel('Counts')
plt.legend()
plt.show()
```
