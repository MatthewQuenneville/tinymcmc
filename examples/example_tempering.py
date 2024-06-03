import tinymcmc
import jax
import matplotlib.pyplot as plt
from functools import partial

def minus_log_likelihood(x, temp):
    return (20*x**2*(x**2-1))/temp

n_steps = 300
epsilon = 0.25
L = 3

@jax.jit
def mcmc_step(key, x):
    return tinymcmc.step_hmc(key, E_dists[0], x, epsilon, L)

@partial(jax.jit, static_argnames=('parity'))
def mcmc_and_tempering_step_parity(key, x, indices, parity):
    for i, i_count in enumerate(index_counts):
        key, key_temp = jax.random.split(key, 2)
        inds = jax.numpy.nonzero(indices==i, size=i_count)
        x = x.at[inds].set(tinymcmc.step_hmc(key_temp, E_dists[i], x[inds], epsilon, L))
    indices = tinymcmc.step_tempering(key, E_dists, x, indices, index_counts, parity)
    return x, indices

def mcmc_and_tempering_step(key, x, indices):
    key, key_parity = jax.random.split(key, 2)
    parity = int(jax.random.randint(key_parity, (1,), 0, 2)[0])
    return mcmc_and_tempering_step_parity(key, x, indices, parity)

n_samples = 15000
n_replicas = 3
temp_factor = 5
assert n_samples%n_replicas==0

E_dists = tuple(partial(minus_log_likelihood, temp=temp_factor**i) for i in range(n_replicas))
samples_baseline = jax.numpy.ones(n_samples)/jax.numpy.sqrt(2)
samples_tempering = jax.numpy.ones(n_samples)/jax.numpy.sqrt(2)
index_counts = (n_samples//n_replicas,)*n_replicas
index_tempering = jax.numpy.concatenate(
    [i*jax.numpy.ones(n_samples//n_replicas, dtype=int) for i in range(n_replicas)]
    )

mean_values_baseline = [jax.numpy.mean(samples_baseline)]
mean_values_tempering = [jax.numpy.mean(samples_tempering[index_tempering==0])]

key = jax.random.key(0)
for i in range(n_steps):
    key, key_exchange, key_mc, key_tempering = jax.random.split(key, 4)
    samples_baseline = mcmc_step(key_mc, samples_baseline)
    samples_tempering, index_tempering = mcmc_and_tempering_step(key_tempering, samples_tempering, index_tempering)
    mean_values_baseline.append(jax.numpy.mean(samples_baseline))
    mean_values_tempering.append(jax.numpy.mean(samples_tempering[index_tempering==0]))

plt.figure(figsize=(8,4))
plt.hist(samples_baseline, alpha=0.5, bins=jax.numpy.linspace(-2,2,51), 
         label='Final Samples\n(without RE)', density=True)
plt.hist(samples_tempering[index_tempering==0], alpha=0.5, bins=jax.numpy.linspace(-2,2,51), 
         label='Final Samples\n(Low Temp)', density=True)
plt.hist(samples_tempering[index_tempering==1], alpha=0.5, bins=jax.numpy.linspace(-2,2,51), 
         label='Final Samples\n(Mid Temp)', density=True)
plt.hist(samples_tempering[index_tempering==2], alpha=0.5, bins=jax.numpy.linspace(-2,2,51), 
         label='Final Samples\n(High Temp)', density=True)
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
plt.plot(mean_values_tempering, label='Sample Mean\n(Tempering)')

plt.axhline(0,color='k', label='True Mean')
plt.xlabel('Step Number')
plt.ylabel('Sample Mean')
plt.legend()
plt.show()


