import tinymcmc
import jax
import matplotlib.pyplot as plt
from functools import partial

def minus_log_likelihood(x, temp):
    return (20*x**2*(x**2-1))/temp

n_steps = 250
epsilon = 0.25
L = 3

@jax.jit
def mcmc_step(key, x):
    return tinymcmc.step_hmc(key, E_dists[0], x, epsilon, L)

@jax.jit
def mcmc_and_tempering_step(key, x, indices):
    for dist, ind in zip(E_dists, indices):
        key, key_temp = jax.random.split(key, 2)
        x = x.at[ind].set(tinymcmc.step_hmc(key_temp, dist, x[ind], epsilon, L))
    indices = tinymcmc.step_tempering(key, E_dists, x, indices)
    return x, indices

samples_per_replica = 500
n_replicas = 5
temp_factor = 2

# Set up replicas

E_dists = tuple(partial(minus_log_likelihood, temp=temp_factor**i) for i in range(n_replicas))
samples_baseline = jax.numpy.ones(samples_per_replica)/jax.numpy.sqrt(2)
samples_tempering = jax.numpy.ones(n_replicas*samples_per_replica)/jax.numpy.sqrt(2)
index_tempering = jax.numpy.arange(n_replicas*samples_per_replica).reshape((n_replicas, samples_per_replica))

mean_values_baseline = [jax.numpy.mean(samples_baseline)]
mean_values_tempering = [jax.numpy.mean(samples_tempering[index_tempering[0]])]

# Perform sampling

key = jax.random.key(1)
for i in range(n_steps):
    key, key_exchange, key_mc, key_tempering = jax.random.split(key, 4)
    samples_baseline = mcmc_step(key_mc, samples_baseline)
    samples_tempering, index_tempering = mcmc_and_tempering_step(key_tempering, samples_tempering, index_tempering)
    mean_values_baseline.append(jax.numpy.mean(samples_baseline))
    mean_values_tempering.append(jax.numpy.mean(samples_tempering[index_tempering[0]]))

# Plot results

plt.figure(figsize=(8,4))
plt.hist(samples_baseline, alpha=0.5, bins=jax.numpy.linspace(-2,2,51), 
         label='Final Samples\n(without RE)', density=True)
plt.hist(samples_tempering[index_tempering[0]], alpha=0.5, bins=jax.numpy.linspace(-2,2,51), 
             label=f'Final Samples\n(with RE)', density=True)
plt.axvline(1/2**0.5, color='k', linestyle='--', label='Initial Value')

dense_x = jax.numpy.linspace(-2,2,201)
dx = jax.numpy.mean(dense_x[1:]-dense_x[:-1])
prob = jax.numpy.exp(-E_dists[0](dense_x))
prob = prob/jax.numpy.sum(prob*dx)

plt.plot(dense_x, prob, 'k', label='True Distribution')
plt.xlim(-2,2)
plt.xlabel('Value')
plt.ylabel('Counts')
plt.legend()
plt.show()

plt.plot(mean_values_baseline, label='Sample Mean\n(without RE)')
plt.plot(mean_values_tempering, label='Sample Mean\n(with RE)')

plt.axhline(0,color='k', label='True Mean')
plt.xlabel('Step Number')
plt.ylabel('Sample Mean')
plt.legend()
plt.show()