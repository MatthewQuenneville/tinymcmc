import jax.numpy as jnp
import jax.random as jrandom
import jax
import matplotlib.pyplot as plt
from functools import partial

@partial(jax.jit, static_argnames=('E_dist'))
def step_rwm(key, E_dist, samples, proposal_std):
    key_step, key_accept = jrandom.split(key, 2)
    proposal = samples+jrandom.normal(key_step, shape=samples.shape)*proposal_std
    alpha = jnp.exp(E_dist(samples)-E_dist(proposal))
    alpha = jnp.minimum(alpha,1.)
    accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha
    samples = jnp.where(accept, proposal, samples)
    return samples

@partial(jax.jit, static_argnames=('E_dist'))
def step_mala(key, E_dist, samples, proposal_std):
    E_grad = jax.vmap(jax.grad(E_dist))
    key_step, key_accept = jrandom.split(key, 2)
    proposal = samples
    proposal = proposal - 0.5*proposal_std**2*E_grad(samples)
    proposal = proposal + jrandom.normal(key_step, shape=samples.shape)*proposal_std

    def E_q(xp, x):
        return 1/2/proposal_std**2*jnp.einsum('i...->i',(xp-x+0.5*proposal_std**2*E_grad(x))**2)

    log_alpha = E_dist(samples) - E_dist(proposal) - (E_q(samples, proposal) - E_q(proposal, samples))
    alpha = jnp.exp(jnp.minimum(log_alpha,0.))
    accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha
    samples = jnp.where(accept, proposal, samples)
    return samples
    
@partial(jax.jit, static_argnames=('E_dist', 'L'))
def step_hmc(key, E_dist, samples, length_scale, L = 10, M=1.):
    epsilon = length_scale/L
    key_step, key_accept = jrandom.split(key, 2)
    initial_momentum = jrandom.normal(key_step, shape = samples.shape)*jnp.sqrt(M)
    E_grad = jax.vmap(jax.grad(E_dist))
    
    # Leapfrog integration
    proposal = samples
    momentum = initial_momentum
    for i in range(L):
        momentum = momentum - 0.5*epsilon*E_grad(proposal)
        proposal = proposal + epsilon*momentum/M
        momentum = momentum - 0.5*epsilon*E_grad(proposal)
    log_alpha = (E_dist(samples)+0.5*initial_momentum**2/M) \
        - (E_dist(proposal)+0.5*momentum**2/M)
    alpha = jnp.exp(jnp.minimum(log_alpha,0.))
    accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha
    samples = jnp.where(accept, proposal, samples)
    return samples

def evolve_chains(key, E_dist, samples, length_scale, n_steps, sampler, **kwargs):
    samplers = {
        'rwm': step_rwm,
        'mala': step_mala,
        'hmc': step_hmc
        }
    step_fn = samplers[sampler]

    for i in range(n_steps):
        key, key_temp = jrandom.split(key, 2)
        samples = step_fn(key_temp, E_dist, samples, length_scale, **kwargs)
        del key_temp

    return samples

if __name__=="__main__":

    key = jrandom.key(1)

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
        hmc_samples = step_hmc(key_temp, lambda x: -logp(x), hmc_samples, jnp.pi/2, L=1)
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
        rwm_samples = step_rwm(key_temp, lambda x: -logp(x), rwm_samples, 2.0)
        del key_temp
        rwm_corr.append(corr(samples, rwm_samples))
    

    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    axes[0].plot(rwm_corr,label='RWM')
    axes[0].plot(mala_corr,label='MALA')
    axes[0].plot(hmc_corr,label='HMC')
    axes[0].legend()
    axes[0].set_ylim(1/jnp.sqrt(n_points), 1)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Autocorrelation')

    bins = jnp.linspace(-5,5,101)
    axes[1].hist(rwm_samples, bins=bins, density=True, alpha=0.25,label='RWM')
    axes[1].hist(mala_samples, bins=bins, density=True, alpha=0.25,label='MALA')
    axes[1].hist(hmc_samples, bins=bins, density=True, alpha=0.25,label='HMC')
    axes[1].hist(samples, bins=bins, density=True, alpha=0.25,label='Initial Samples')
    axes[1].plot(bins, jnp.exp(logp(bins)))
    axes[1].legend()
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('Normalized Counts')

    plt.show()