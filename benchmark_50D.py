import jax.numpy as jnp
import jax.random as jrandom
import jax
import matplotlib.pyplot as plt
import jaxmcmc

if __name__=="__main__":

    key = jrandom.key(1)

    n_points = 1000000
    d=50

    @jax.jit
    def logp(x):
        return -jnp.sum(x**2, axis=-1)/2-0.5*jnp.log(2*jnp.pi)

    key, key_temp = jrandom.split(key, 2)
    samples = jrandom.normal(key=key_temp, shape=(n_points,d))
    del key_temp

    sample_variance = jnp.var(samples)

    @jax.jit
    def corr(init_samples, samples):
        return jnp.mean((init_samples-jnp.mean(init_samples))*(samples-jnp.mean(samples)))

    # HMC

    hmc_length_scale = 0.6
    L = 3

    @jax.jit
    def step_func(key, samples):
        return jaxmcmc.step_hmc(key, lambda x: -logp(x), samples, hmc_length_scale, L = L)
    
    hmc_samples = samples
    hmc_corr = [corr(samples, hmc_samples)/sample_variance]

    while hmc_corr[-1]/sample_variance>1/jnp.sqrt(n_points):
        key, key_temp = jrandom.split(key, 2)
        hmc_samples = step_func(key_temp, hmc_samples)
        del key_temp
        hmc_corr.append(corr(samples, hmc_samples)/sample_variance)

    # MALA

    mala_length_scale = 0.8

    @jax.jit
    def step_func(key, samples):
        return jaxmcmc.step_mala(key, lambda x: -logp(x), samples, mala_length_scale)

    mala_samples = samples
    mala_corr = [corr(samples, mala_samples)/sample_variance]

    while mala_corr[-1]/sample_variance>1/jnp.sqrt(n_points):
        key, key_temp = jrandom.split(key, 2)
        mala_samples = step_func(key_temp, mala_samples)
        del key_temp
        mala_corr.append(corr(samples, mala_samples)/sample_variance)

    # RWM

    rwm_length_scale = 0.3

    @jax.jit
    def step_func(key, samples):
        return jaxmcmc.step_rwm(key, lambda x: -logp(x), samples, rwm_length_scale)

    rwm_samples = samples
    rwm_corr = [corr(samples, rwm_samples)/sample_variance]

    while rwm_corr[-1]/sample_variance>1/jnp.sqrt(n_points):
        key, key_temp = jrandom.split(key, 2)
        rwm_samples = step_func(key_temp, rwm_samples)
        del key_temp
        rwm_corr.append(corr(samples, rwm_samples)/sample_variance)  


    plt.plot(1+jnp.arange(len(rwm_corr)), rwm_corr,label='RWM')
    plt.plot(1+jnp.arange(len(mala_corr)), mala_corr,label='MALA')
    plt.plot(1+L*jnp.arange(len(hmc_corr)), hmc_corr,label='HMC')
    plt.legend()
    plt.ylim(1/jnp.sqrt(n_points), 1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Steps')
    plt.ylabel('Autocorrelation')
    plt.show()