import jax.numpy as jnp
import jax.random as jrandom
import jax
import matplotlib.pyplot as plt
from functools import partial

@partial(jax.jit, static_argnames=('E_dist'))
def step_rwm(key, E_dist, samples, proposal_std, bounds=(-jnp.inf, jnp.inf)):
    key_step, key_accept = jrandom.split(key, 2)
    proposal = samples+jrandom.normal(key_step, shape=samples.shape)*proposal_std

    # Metropolis-Hastings acceptance/Rejection
    alpha = jnp.exp(E_dist(samples)-E_dist(proposal))
    alpha = jnp.minimum(alpha,1.)
    accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha

    # Enforce bound constraints
    out_of_bounds = jnp.logical_or(proposal>bounds[1],
                                   proposal<bounds[0])
    if len(samples.shape)>1:
        out_of_bounds = jnp.any(out_of_bounds, axis=1)
    accept = jnp.where(out_of_bounds, False, accept)
    if len(samples.shape)>1:
        accept = jnp.repeat(accept[:,jnp.newaxis],samples.shape[1],axis=1)
    samples = jnp.where(accept, proposal, samples)
    return samples

@partial(jax.jit, static_argnames=('E_dist'))
def step_mala(key, E_dist, samples, proposal_std, bounds=(-jnp.inf, jnp.inf)):
    
    E_grad = jax.vmap(jax.grad(E_dist))

    key_step, key_accept = jrandom.split(key, 2)
    proposal = samples
    proposal = proposal - 0.5*proposal_std**2*E_grad(samples)
    proposal = proposal + jrandom.normal(key_step, shape=samples.shape)*proposal_std

    def E_q(xp, x):
        return 1/2/proposal_std**2*jnp.einsum('i...->i',(xp-x+0.5*proposal_std**2*E_grad(x))**2)

    # Metropolis-Hastings acceptance/Rejection
    log_alpha = E_dist(samples) - E_dist(proposal) - (E_q(samples, proposal) - E_q(proposal, samples))
    alpha = jnp.exp(jnp.minimum(log_alpha,0.))
    accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha

    # Enforce bound constraints
    out_of_bounds = jnp.logical_or(proposal>bounds[1],
                                   proposal<bounds[0])
    if len(samples.shape)>1:
        out_of_bounds = jnp.any(out_of_bounds, axis=1)
    accept = jnp.where(out_of_bounds, False, accept)
    if len(samples.shape)>1:
        accept = jnp.repeat(accept[:,jnp.newaxis],samples.shape[1],axis=1)
    samples = jnp.where(accept, proposal, samples)
    return samples
    
@partial(jax.jit, static_argnames=('E_dist', 'L'))
def step_hmc(key, E_dist, samples, length_scale, L = 10, M=1., bounds=(-jnp.inf, jnp.inf), perturb_step = 0.2):
    epsilon = length_scale/L
    key_step, key_perturb, key_accept = jrandom.split(key, 3)
    try:
        initial_momentum = jrandom.multivariate_normal(key_step, 
                                                       mean=jnp.zeros(samples.shape[0]),
                                                       cov = M)
    except ValueError:
        initial_momentum = jrandom.normal(key_step, shape = samples.shape)*jnp.sqrt(M)
    
    E_grad = jax.vmap(jax.grad(E_dist))

    proposal = samples
    momentum = initial_momentum
    for i in range(L):
        step_size = epsilon*(1+perturb_step*jax.random.uniform(key_perturb))
        momentum = momentum - 0.5*epsilon*E_grad(proposal)
        proposal = proposal + epsilon*momentum/M
        momentum = momentum - 0.5*epsilon*E_grad(proposal)

    # Metropolis-Hastings acceptance/rejection
    log_alpha = (E_dist(samples)+0.5*jnp.einsum('i...->i',initial_momentum**2)/M) \
        - (E_dist(proposal)+0.5*jnp.einsum('i...->i',momentum**2)/M)
    alpha = jnp.exp(jnp.minimum(log_alpha,0.))
    accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha

    # Enforce bound constraints
    out_of_bounds = jnp.logical_or(proposal>bounds[1],
                                   proposal<bounds[0])
    if len(samples.shape)>1:
        out_of_bounds = jnp.any(out_of_bounds, axis=1)
    accept = jnp.where(out_of_bounds, False, accept)
    if len(samples.shape)>1:
        accept = jnp.repeat(accept[:,jnp.newaxis],samples.shape[1],axis=1)

    samples = jnp.where(accept, proposal, samples)

    return samples