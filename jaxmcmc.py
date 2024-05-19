import jax.numpy as jnp
import jax.random as jrandom
import jax
import sys
import matplotlib.pyplot as plt
from functools import partial

key = jrandom.key(1)

class normal_mixture:
    def __init__(self, locs, scales, weights=None):
        self.n_mix = len(locs)
        assert self.n_mix == len(scales)

        self.locs = jnp.array(locs)
        self.scales = jnp.array(scales)

        if weights is not None:
            assert self.n_mix == len(scales)
            self.weights = jnp.array(weights)
        else:
            self.weights = jnp.ones(self.n_mix)/self.n_mix

    def sample(self, key, shape=()):
        key_normal, key_choice = jrandom.split(key, 2)
        samples = jrandom.normal(key_normal, shape = shape)
        classes = jrandom.choice(key_choice, self.n_mix, shape = shape, p = self.weights)

        return samples*self.scales[classes]+self.locs[classes]
    
    def E(self, x):
        E = (x[...,jnp.newaxis]-self.locs)**2/2/self.scales**2
        b = self.weights/jnp.sqrt(2*jnp.pi)/self.scales
        E_mix = -jax.scipy.special.logsumexp(-E, b=b, axis=-1)
        return E_mix

def random_walk_metropolis(key, E_dist, samples, proposal_std, n_steps, adaptive=False):
    proposal_mean = 0.
    for i in range(n_steps):
        damping = 1/(i+1)
        key, key_temp = jrandom.split(key)
        new_samples = step_random_walk_metropolis(key, E_dist, samples, proposal_std, proposal_mean)
        del key_temp
        if adaptive:
            delta = new_samples-samples
            proposal_std = jnp.sqrt(proposal_std**2+damping*(jnp.mean((delta-proposal_mean)**2)-proposal_std**2))
            proposal_mean = proposal_mean+damping*jnp.mean(delta-proposal_mean)
            print(proposal_mean, proposal_std)
        samples = new_samples
    return samples

@partial(jax.jit, static_argnames=('E_dist'))
def step_random_walk_metropolis(key, E_dist, samples, proposal_std):
    key_step, key_accept = jrandom.split(key, 2)
    proposal = samples+jrandom.normal(key_step, shape=(len(samples),))*proposal_std
    alpha = jnp.exp(E_dist(samples)-E_dist(proposal))
    alpha = jnp.minimum(alpha,1.)
    accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha
    samples = jnp.where(accept, proposal, samples)
    return samples

def random_walk_metropolis(key, E_dist, samples, proposal_std, n_steps, return_mean = False):
    if return_mean:
        mean = [jnp.mean(samples)]
    for i in range(n_steps):
        key, key_temp = jrandom.split(key, 2)
        samples = step_random_walk_metropolis(key_temp, E_dist, samples, proposal_std)
        del key_temp

        if return_mean:
            mean.append(jnp.mean(samples))
    if return_mean:
        return samples, jnp.array(mean)
    else:
        return samples

@partial(jax.jit, static_argnames=('E_dist'))
def step_mala(key, E_dist, samples, proposal_std):
    E_grad = jax.vmap(jax.grad(E_dist))
    key_step, key_accept = jrandom.split(key, 2)
    proposal = samples
    proposal = proposal - 0.5*proposal_std**2*E_grad(samples)
    proposal = proposal + jrandom.normal(key_step, shape=(len(samples),))*proposal_std

    def E_q(xp, x):
        return (1/2/proposal_std**2*(xp-x+0.5*proposal_std**2*E_grad(x))**2)

    log_alpha = E_dist(samples)-E_dist(proposal) - (E_q(samples, proposal) - E_q(proposal, samples))
    alpha = jnp.exp(jnp.minimum(log_alpha,0.))
    accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha
    samples = jnp.where(accept, proposal, samples)
    return samples

def mala(key, E_dist, samples, proposal_std, n_steps, return_mean = False):
    if return_mean:
        mean = [jnp.mean(samples)]
    for i in range(n_steps):
        key, key_temp = jrandom.split(key, 2)
        samples = step_mala(key_temp, E_dist, samples, proposal_std)
        del key_temp

        if return_mean:
            mean.append(jnp.mean(samples))
    if return_mean:
        return samples, jnp.array(mean)
    else:
        return samples
    
#@partial(jax.jit, static_argnames=('E_dist'))
def step_hmc(key, E_dist, samples, proposal_std, n_intervals = 1, M=1.):
    epsilon = proposal_std/n_intervals
    key_step, key_accept = jrandom.split(key, 2)
    initial_momentum = jrandom.normal(key_step, shape = (len(samples),))*jnp.sqrt(M)
    E_grad = jax.vmap(jax.grad(E_dist))
    
    # Leapfrog integration
    proposal = samples
    momentum = initial_momentum
    x = [proposal]
    p = [momentum]
    for i in range(n_intervals):
        momentum = momentum - 0.5*epsilon*E_grad(proposal)
        proposal = proposal + epsilon*momentum/M
        momentum = momentum - 0.5*epsilon*E_grad(proposal)
        x.append(proposal)
        p.append(momentum)
    for i in range(100):
        plt.plot([j[i] for j in x], [j[i] for j in p])
    plt.show()
    log_alpha = (E_dist(samples)+0.5*initial_momentum**2/M) \
        - (E_dist(proposal)+0.5*momentum**2/M)
    alpha = jnp.exp(jnp.minimum(log_alpha,0.))
    accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha
    samples = jnp.where(accept, proposal, samples)
    return samples

def hmc(key, E_dist, samples, proposal_std, n_steps, return_mean = False, M=1.):
    if return_mean:
        mean = [jnp.mean(samples)]
    for i in range(n_steps):
        key, key_temp = jrandom.split(key, 2)
        samples = step_hmc(key_temp, E_dist, samples, proposal_std, n_intervals = 10, M=M)
        del key_temp

        if return_mean:
            mean.append(jnp.mean(samples))
    if return_mean:
        return samples, jnp.array(mean)
    else:
        return samples

n_points = 10000
model0 = normal_mixture((-1,1), (0.1,0.1), (0.8, 0.2))
model1 = normal_mixture((-1,1), (0.1,0.1), (0.2, 0.8))

key, key_temp = jrandom.split(key,2)
samples = model0.sample(key_temp, (n_points,))
del key_temp

M = 0.1
key, key_temp = jrandom.split(key,2)
hmc_samples, hmc_mean = hmc(key_temp, model0.E, samples, 0.4*jnp.sqrt(M), 1, return_mean=True, M=M)
del key_temp

sys.exit()
key, key_temp = jrandom.split(key,2)
mala_samples, mala_mean = mala(key_temp, model1.E, samples, 0.39, 50000, return_mean=True)
del key_temp

key, key_temp = jrandom.split(key,2)
rwm_samples, rwm_mean = random_walk_metropolis(key_temp, model1.E, samples, 2.0, 50000, return_mean=True)
del key_temp

#print(jnp.argmax(rwm_mean>0.6-1.2/jnp.sqrt(n_points)))
#print(0.6-1.2/jnp.sqrt(n_points))
plt.plot(range(1,50002),rwm_mean)
plt.plot(range(1,50002),mala_mean)
plt.xscale('log')
plt.axhline(0.6)
plt.axhline(-0.6)
plt.show()
sys.exit()
bins = jnp.linspace(-1.5,1.5,50)
dense = jnp.linspace(-1.5,1.5,1000)

plt.hist(samples, bins = bins, density=True, alpha=0.5)
plt.hist(rwm_samples, bins = bins, density=True, alpha=0.5)

plt.plot(dense, jnp.exp(-model0.E(dense)))
plt.plot(dense, jnp.exp(-model1.E(dense)))
plt.show()
sys.exit()