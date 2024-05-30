import tinymcmc
import jax
import matplotlib.pyplot as plt

def minus_log_likelihood(x):
    return (20*x**2*(x**2-1))

def minus_log_likelihood_high_temp(x):
    return minus_log_likelihood(x)/20

n_steps = 300
epsilon = 0.25
L = 3

@jax.jit
def mcmc_step(key, x):
    return tinymcmc.step_hmc(key, minus_log_likelihood, x, epsilon, L)

@jax.jit
def mcmc_and_exchange_step(key, x1, x2):
    key1, key2, key_exchange = jax.random.split(key, 3)
    x1 = tinymcmc.step_hmc(key1, minus_log_likelihood, x1, epsilon, L)
    x2 = tinymcmc.step_hmc(key2, minus_log_likelihood_high_temp, x2, epsilon, L)
    return tinymcmc.step_exchange(key_exchange, minus_log_likelihood, x1,
                                  minus_log_likelihood_high_temp, x2)

init_samples = jax.numpy.ones(10000)/jax.numpy.sqrt(2)
samples_lowT = init_samples
samples_highT = init_samples
samples_baseline = init_samples

mean_values_baseline = [jax.numpy.mean(samples_baseline)]
mean_values_lowT = [jax.numpy.mean(samples_lowT)]
mean_values_highT = [jax.numpy.mean(samples_highT)]

key = jax.random.key(1)
for i in range(n_steps):
    key, key_exchange, key_mc = jax.random.split(key, 3)
    samples_baseline = mcmc_step(key_mc, samples_baseline)
    samples_lowT, samples_highT = mcmc_and_exchange_step(key_exchange, samples_lowT, samples_highT)
    mean_values_lowT.append(jax.numpy.mean(samples_lowT))
    mean_values_highT.append(jax.numpy.mean(samples_highT))
    mean_values_baseline.append(jax.numpy.mean(samples_baseline))

plt.figure(figsize=(8,4))
plt.hist(samples_baseline, alpha=0.5, bins=jax.numpy.linspace(-2,2,51), 
         label='Final Samples\n(without RE)', density=True)
plt.hist(samples_lowT, alpha=0.5, bins=jax.numpy.linspace(-2,2,51), 
         label='Final Samples\n(with RE)', density=True)
plt.hist(samples_highT, alpha=0.5, bins=jax.numpy.linspace(-2,2,51), 
         label='Final Samples\n(Replica)', density=True)
plt.axvline(1/2**0.5, color='k', linestyle='--', label='Initial Value')

dense_x = jax.numpy.linspace(-2,2,201)
dx = jax.numpy.mean(dense_x[1:]-dense_x[:-1])
prob = jax.numpy.exp(-minus_log_likelihood(dense_x))
prob = prob/jax.numpy.sum(prob*dx)

plt.plot(dense_x, prob, 'k', label='True Distribution')
plt.xlim(-2,2)
plt.xlabel('Value')
plt.ylabel('Counts')
plt.legend()
plt.show()

plt.plot(mean_values_baseline, label='Sample Mean\n(without RE)')
plt.plot(mean_values_lowT, label='Sample Mean\n(with RE)')
plt.plot(mean_values_highT, label='Sample Mean\n(Replica)')
plt.axhline(0,color='k', label='True Mean')
plt.xlabel('Step Number')
plt.ylabel('Sample Mean')
plt.legend()
plt.show()


