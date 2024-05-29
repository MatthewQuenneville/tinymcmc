import jax.numpy as jnp
import jax.random as jrandom
import jax
from functools import partial

@partial(jax.jit, static_argnames=('E_dist', 'metropolize'))
def step_rwm(key, E_dist, samples, proposal_std, 
             bounds = (-jnp.inf, jnp.inf), metropolize = True):
    """Random Walk Metropolis sampling step

    Computes the next set of samples after a Random Walk Metropolis step

    Parameters
    ----------
    key : jax.prng.PRNGKeyArray
        Random key
    E_dist : function
        Negative log-likelihood function
    samples : array_like
        Array of samples, with the first axis indexing the samples. Any
        additional axes should be mapped by E_dist to a scalar.
    proposal_std : float
        Standard deviation of the proposal distribution
    bounds : {tuple of floats, tuple of array_likes}, optional
        Boundary constraints to be enforced on samples. Must be either
        a tuple of two floats, or a tuple of two 1-dimensional 
        array_likes with the same size as the first axis of samples.
        Defaults to (-infinity, infinity).
    metropolize: boolean, optional
        Controls whether to apply a Metropolis-Hastings correction.
        If False, the returned samples are just evolved by random walk.
        Defaults to True.

    Returns
    -------
    out : array_like
        Array of samples after a single step of Random Walk Metropolis

    """
    key_step, key_accept = jrandom.split(key, 2)
    proposal = samples+jrandom.normal(key_step, shape=samples.shape)*proposal_std

    # Metropolis-Hastings acceptance/Rejection
    if metropolize:
        alpha = jnp.exp(E_dist(samples)-E_dist(proposal))
        alpha = jnp.minimum(alpha,1.)
        accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha
    else:
        accept = jnp.ones(samples.shape[0], dtype=bool)

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

@partial(jax.jit, static_argnames=('E_dist', 'metropolize'))
def step_mala(key, E_dist, samples, proposal_std, 
              bounds = (-jnp.inf, jnp.inf), metropolize = True):
    """Metropolis-Adjusted Langevin Algorithm sampling step

    Computes the next set of samples after a Metropolis-Adjusted Langevin step

    Parameters
    ----------
    key : jax.prng.PRNGKeyArray
        Random key
    E_dist : function
        Negative log-likelihood function
    samples : array_like
        Array of samples, with the first axis indexing the samples. Any
        additional axes should be mapped by E_dist to a scalar.
    proposal_std : float
        Standard deviation of the random part of the proposal distribution
    bounds : {tuple of floats, tuple of array_likes}, optional
        Boundary constraints to be enforced on samples. Must be either
        a tuple of two floats, or a tuple of two 1-dimensional 
        array_likes with the same size as the first axis of samples.
        Defaults to (-infinity, infinity).
    metropolize: boolean, optional
        Controls whether to apply a Metropolis-Hastings correction.
        Defaults to True.

    Returns
    -------
    out : array_like
        Array of samples after a single step of RMetropolis-Adjusted Langevin

    """
    E_grad = jax.vmap(jax.grad(E_dist))

    key_step, key_accept = jrandom.split(key, 2)
    proposal = samples
    proposal = proposal - 0.5*proposal_std**2*E_grad(samples)
    proposal = proposal + jrandom.normal(key_step, shape=samples.shape)*proposal_std

    def E_q(xp, x):
        return 1/2/proposal_std**2*jnp.einsum('i...->i',(xp-x+0.5*proposal_std**2*E_grad(x))**2)

    # Metropolis-Hastings acceptance/Rejection
    if metropolize:
        log_alpha = E_dist(samples) - E_dist(proposal) - (E_q(samples, proposal) - E_q(proposal, samples))
        alpha = jnp.exp(jnp.minimum(log_alpha,0.))
        accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha
    else:
        accept = jnp.ones(samples.shape[0], dtype=bool)

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
    
@partial(jax.jit, static_argnames=('E_dist', 'L', 'metropolize'))
def step_hmc(key, E_dist, samples, epsilon, L, M=1., 
             bounds=(-jnp.inf, jnp.inf), perturb_step = 0.2, metropolize = True):
    """Hamiltonian Monte Carlo sampling step

    Computes the next set of samples after a Hamiltonian Monte Carlo step

    Parameters
    ----------
    key : jax.prng.PRNGKeyArray
        Random key
    E_dist : function
        Negative log-likelihood function
    samples : array_like
        Array of samples, with the first axis indexing the samples. Any
        additional axes should be mapped by E_dist to a scalar.
    epsilon : float
        Single time-step length scale parameter
    L : int
        Number of time-steps taken for a complete HMC step
    M : {float, array_like}, optional
        Mass parameter value, or diagonal of mass matrix. Currently,
        only diagonal mass matrices are allowed, meaning M is either a 
        float, or a vector of shape samples.shape[1:].
        Defaults to 1.
    bounds : {tuple of floats, tuple of array_likes}, optional
        Boundary constraints to be enforced on samples. Must be either
        a tuple of two floats, or a tuple of two 1-dimensional 
        array_likes with the same size as the first axis of samples.
        Defaults to (-infinity, infinity).
    perturb_step: float, optional
        Perturb step-size to avoid non-ergodicity. Step-size is randomly
        chosen in [epsilon*(1-perturb_step), epsilon*(1+perturb_step)], 
        meaning perturb_step must be in the range [0,1).
        Defaults to 0.2.
    metropolize: boolean, optional
        Controls whether to apply a Metropolis-Hastings correction.
        Defaults to True.

    Returns
    -------
    out : array_like
        Array of samples after a single step of Hamiltonian Monte Carlo

    """
    key_step, key_perturb, key_accept = jrandom.split(key, 3)
    try:
        initial_momentum = jrandom.multivariate_normal(key_step, 
                                                       mean=jnp.zeros(samples.shape[0]),
                                                       cov = M)
    except ValueError:
        initial_momentum = jrandom.normal(key_step, shape = samples.shape)*jnp.sqrt(M)
    
    E_grad = jax.vmap(jax.grad(E_dist))

    assert perturb_step>=0 and perturb_step<1
    proposal = samples
    momentum = initial_momentum
    for i in range(L):
        step_size = epsilon*(1+perturb_step*jax.random.uniform(key_perturb))
        momentum = momentum - 0.5*step_size*E_grad(proposal)
        proposal = proposal + step_size*momentum/M
        momentum = momentum - 0.5*step_size*E_grad(proposal)

    # Metropolis-Hastings acceptance/rejection
    if metropolize:
        log_alpha = (E_dist(samples)+0.5*jnp.einsum('i...,...->i',initial_momentum**2,1/M)) \
            - (E_dist(proposal)+0.5*jnp.einsum('i...,...->i',momentum**2,1/M))
        alpha = jnp.exp(jnp.minimum(log_alpha,0.))
        accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha
    else:
        accept = jnp.ones(samples.shape[0], dtype=bool)

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