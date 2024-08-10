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
    for i in samples.shape[1:]:
        out_of_bounds = jnp.any(out_of_bounds, axis=-1)
    
    accept = jnp.where(out_of_bounds, False, accept)
    for i in samples.shape[1:]:
        accept = jnp.repeat(accept[...,jnp.newaxis], i, axis=-1)
    samples = jnp.where(accept, proposal, samples)
    return samples

@partial(jax.jit, static_argnames=('E_dist', 'metropolize'))
def step_mala(key, E_dist, samples, proposal_std, 
              bounds = (-jnp.inf, jnp.inf), metropolize = True,
              noise_scale = 1.0):
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
    metropolize : boolean, optional
        Controls whether to apply a Metropolis-Hastings correction.
        Defaults to True.
    noise_scale : float, optional
        Scaling factor for the gaussian noise term. A value of 1
        is needed for Metropolis-Adjusted Langevin, while a value of 0
        neglects the gaussian noise term. A value of 1 is needed to guarantee
        convergence to the correct stationary distribution. 
        Defaults to 1. 

    Returns
    -------
    out : array_like
        Array of samples after a single step of RMetropolis-Adjusted Langevin

    """
    E_grad = jax.vmap(jax.grad(E_dist))

    key_step, key_accept = jrandom.split(key, 2)

    proposal = samples
    proposal = proposal - 0.5*proposal_std**2*E_grad(samples)
    proposal = proposal + noise_scale * \
        jrandom.normal(key_step, shape=samples.shape)*proposal_std

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
    for i in samples.shape[1:]:
        out_of_bounds = jnp.any(out_of_bounds, axis=-1)
    
    accept = jnp.where(out_of_bounds, False, accept)
    for i in samples.shape[1:]:
        accept = jnp.repeat(accept[...,jnp.newaxis], i, axis=-1)
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
    for i in samples.shape[1:]:
        out_of_bounds = jnp.any(out_of_bounds, axis=-1)

    accept = jnp.where(out_of_bounds, False, accept)

    for i in samples.shape[1:]:
        accept = jnp.repeat(accept[...,jnp.newaxis], i, axis=-1)

    samples = jnp.where(accept, proposal, samples)

    return samples

@partial(jax.jit, static_argnames=('E_dist1', 'E_dist2', 'metropolize'))
def step_exchange(key, E_dist1, samples1, E_dist2, samples2, metropolize=True):
    """Replica Exchange (Parallel Tempering) sampling step for a pair of replicas

    Computes the next sets of samples after a pair Replica Exchange step

    Parameters
    ----------
    key : jax.prng.PRNGKeyArray
        Random key
    E_dist1 : function
        Negative log-likelihood function for first replica
    samples1 : array_like
        Array of samples for the first replica, with the first axis 
        indexing the samples. Any additional axes should be mapped by 
        E_dist1 to a scalar. The shape of samples1 must match the shape 
        of samples2, with the possible exception of the first axis length.
    E_dist2 : function
        Negative log-likelihood function for second replica
    samples2 : array_like
        Array of samples for the second replica, with the first axis 
        indexing the samples. Any additional axes should be mapped by 
        E_dist1 to a scalar. The shape of samples2 must match the shape 
        of samples1, with the possible exception of the first axis length.
    metropolize: boolean, optional
        Controls whether to apply a Metropolis-Hastings correction.
        If False, all proposed exchanges are accepted.
        Defaults to True.

    Returns
    -------
    out1 : array_like
        Array of samples for the first replica after the exchange step
    out2 : array_like
        Array of samples for the second replica after the exchange step

    Notes
    -----
    The step_tempering function below should be favoured for applications 
    where the data has many dimensions, and performance is important.

    If samples1 and samples2 contain different numbers of samples (ie. their
    first axes have different lengths), then a single exchange is attempted for
    each sample in the smaller set, and the remaining samples in the larger set
    are left unchanged.

    """
    n_swap = min(samples1.shape[0], samples2.shape[0])
    key1, key2, key_accept = jrandom.split(key, 3)

    proposal_index1 = jrandom.choice(key1, samples1.shape[0], shape=(n_swap,), replace=False)
    proposal_index2 = jrandom.choice(key2, samples2.shape[0], shape=(n_swap,), replace=False)

    if metropolize:
        initial_E = E_dist1(samples1[proposal_index1]) + E_dist2(samples2[proposal_index2])
        swapped_E = E_dist2(samples1[proposal_index1]) + E_dist1(samples2[proposal_index2])
        alpha = jnp.exp(jnp.minimum(initial_E-swapped_E, 0.))
        accept = jrandom.uniform(key_accept, minval=0, maxval=1, shape = alpha.shape)<alpha
    else:
        accept = jnp.ones(n_swap, dtype=bool)

    for i in samples1.shape[1:]:
        accept = jnp.repeat(accept[...,jnp.newaxis], i, axis=-1)

    swapped_values1 = jnp.where(accept, samples2[proposal_index2], samples1[proposal_index1])
    swapped_values2 = jnp.where(accept, samples1[proposal_index1], samples2[proposal_index2])

    samples1 = samples1.at[proposal_index1].set(swapped_values1)
    samples2 = samples2.at[proposal_index2].set(swapped_values2)
    
    return samples1, samples2

@partial(jax.jit, static_argnames=('E_dists'))
def step_tempering(key, E_dists, samples, replica_index):
    """Replica Exchange (Parallel Tempering) sampling step

    Computes the next sets of samples after a Replica Exchange step

    Parameters
    ----------
    key : jax.prng.PRNGKeyArray
        Random key
    E_dists : tuple of functions
        Negative log-likelihood functions for indexed replicas
    samples : array_like
        Array of samples for all replicas, with the first axis 
        indexing the samples. Any additional axes should be mapped by 
        the functions in E_dists to scalars. Each replica must contain
        an equal number of samples.
    replica_index : array_like of ints
        2D array of replica indices for each sample in samples. The first axis
        indexes the different replicas, while the second indexes the samples
        within each replica. Indices should be integers ranging from 0 to 
        len(samples)-1.

    Returns
    -------
    out : array_like of ints
        Array of replica indices after the parallel tempering step has been 
        performed. See replica_index for further description.

    Notes
    -----
    Each replica must have the same number of samples. 
    """

    n_dists, samples_per_dist = replica_index.shape

    key, key_permute, key_parity = jrandom.split(key, 3)
    replica_index = jrandom.permutation(
        key_permute, 
        replica_index,
        axis=1, independent=True)
    if n_dists>2:
        parity = jrandom.randint(key_parity, (samples_per_dist,),0,2)

    E = jnp.zeros_like(replica_index, dtype=float)
    E_shift_up = jnp.nan*jnp.ones_like(replica_index, dtype=float)
    E_shift_down = jnp.nan*jnp.ones_like(replica_index, dtype=float)
    for i in range(n_dists):
        E = E.at[i].set(E_dists[i](samples[replica_index[i]]))
        if i<n_dists-1:
            E_shift_up = E_shift_up.at[i].set(E_dists[i+1](samples[replica_index[i]]))
        if i>0:
            E_shift_down = E_shift_down.at[i].set(E_dists[i-1](samples[replica_index[i]]))

    # Even parity swaps
    E_init = E[:-1:2]+E[1::2]
    E_final = E_shift_up[:-1:2]+E_shift_down[1::2]
    alpha_even = jnp.exp(jnp.minimum(E_init-E_final, 0.))

    if n_dists>2:
        # Odd parity swaps
        E_init = E[1:-1:2]+E[2::2]
        E_final = E_shift_up[1:-1:2]+E_shift_down[2::2]
        alpha_odd = jnp.exp(jnp.minimum(E_init-E_final, 0.))


    alpha = jnp.zeros((n_dists-1,samples_per_dist))
    alpha = alpha.at[::2].set(alpha_even)
    if n_dists>2:
        alpha = alpha.at[1::2].set(alpha_odd)
        alpha = jnp.where(jnp.indices(alpha.shape)[0]%2==parity,alpha,0.)

    key, key_accept = jrandom.split(key, 2)

    accept = jrandom.uniform(key_accept, alpha.shape,minval=0,maxval=1)<alpha
    shift_dir = jnp.pad(accept.astype(int),((0,1),(0,0)))-jnp.pad(accept.astype(int),((1,0),(0,0)))
    replica_index = jnp.select((shift_dir==-1,shift_dir==0,shift_dir==1),
                               (replica_index.take(jnp.arange(-1,n_dists-1),axis=0),
                                replica_index.take(jnp.arange(0,n_dists),axis=0),
                                replica_index.take(jnp.arange(1,n_dists+1),axis=0)))
    return replica_index
