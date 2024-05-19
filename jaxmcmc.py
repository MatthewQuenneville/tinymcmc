import jax.numpy as jnp
import jax.random as jrandom
import jax
import sys
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

key = jrandom.key(1)

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
    
@partial(jax.jit, static_argnames=('E_dist', 'L'))
def step_hmc(key, E_dist, samples, epsilon, L = 1, M=1.):
    key_step, key_accept = jrandom.split(key, 2)
    initial_momentum = jrandom.normal(key_step, shape = (len(samples),))*jnp.sqrt(M)
    E_grad = jax.vmap(jax.grad(E_dist))
    
    # Leapfrog integration
    proposal = samples
    momentum = initial_momentum
    #x = [proposal]
    #p = [momentum]
    for i in range(L):
        momentum = momentum - 0.5*epsilon*E_grad(proposal)
        proposal = proposal + epsilon*momentum/M
        momentum = momentum - 0.5*epsilon*E_grad(proposal)
    #    x.append(proposal)
    #    p.append(momentum)
    #for i in range(100):
    #    plt.plot([j[i] for j in x], [j[i] for j in p])
    #plt.show()
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
        samples = step_hmc(key_temp, E_dist, samples, proposal_std/10, L = 100, M=M)
        del key_temp

        if return_mean:
            mean.append(jnp.mean(samples))
    if return_mean:
        return samples, jnp.array(mean)
    else:
        return samples

if __name__=="__main__":
    n_points = 10000
    def logp(x):
        return -x**2/2-0.5*jnp.log(2*jnp.pi)

    key, key_temp = jrandom.split(key, 2)
    samples = jrandom.normal(key=key_temp, shape=(n_points,))
    del key_temp

    sample_variance = jnp.var(samples)

    @jax.jit
    def corr(init_samples, samples):
        return jnp.mean((init_samples-jnp.mean(init_samples))*(samples-jnp.mean(samples)))

    # HMC

    hmc_samples = samples
    hmc_corr = [corr(samples, hmc_samples)]

    while hmc_corr[-1]/sample_variance>1/jnp.sqrt(n_points):
        key, key_temp = jrandom.split(key, 2)
        hmc_samples = step_hmc(key_temp, lambda x: -logp(x), hmc_samples, epsilon = jnp.pi/2, L=1)
        del key_temp
        hmc_corr.append(corr(samples, hmc_samples))

    # MALA

    mala_samples = samples
    mala_corr = [corr(samples, mala_samples)]

    while mala_corr[-1]/sample_variance>1/jnp.sqrt(n_points):
        key, key_temp = jrandom.split(key, 2)
        mala_samples = step_mala(key_temp, lambda x: -logp(x), mala_samples, 2.15)
        del key_temp
        mala_corr.append(corr(samples, mala_samples))

    # RWM

    rwm_samples = samples
    rwm_corr = [corr(samples, rwm_samples)]

    while rwm_corr[-1]/sample_variance>1/jnp.sqrt(n_points):
        key, key_temp = jrandom.split(key, 2)
        rwm_samples = step_random_walk_metropolis(key_temp, lambda x: -logp(x), rwm_samples, 2.0)
        del key_temp
        rwm_corr.append(corr(samples, rwm_samples))
    

    plt.plot(rwm_corr,label='RWM')
    plt.plot(mala_corr,label='MALA')
    plt.plot(hmc_corr,label='HMC')
    plt.legend()
    plt.ylim(1/jnp.sqrt(n_points), 1)
    plt.yscale('log')
    plt.show()

    bins = jnp.linspace(-5,5,101)
    plt.hist(samples, bins=bins, density=True, alpha=0.5,label='Initial Samples')
    plt.hist(rwm_samples, bins=bins, density=True, alpha=0.5,label='RWM')
    plt.hist(mala_samples, bins=bins, density=True, alpha=0.5,label='MALA')
    plt.hist(hmc_samples, bins=bins, density=True, alpha=0.5,label='HMC')
    plt.plot(bins, jnp.exp(logp(bins)))
    plt.legend()
    plt.show()