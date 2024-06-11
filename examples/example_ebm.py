import jax.random as jrandom
import jax.numpy as jnp
import jax
from flax import linen as nn
import optax
import matplotlib.pyplot as plt
import numpy as np
import tinymcmc
from functools import partial
import tqdm

# Generate dataset
n_data = 10000
key = jrandom.key(0)

key, key_normal, key_choice = jrandom.split(key, 3)
data = jrandom.normal(key, shape=(n_data,2)) \
    + jrandom.choice(key_choice, 
                     jnp.array([[-5.,-5.],[-5.,5.],[5., 5.],[5.,-5.]]),
                     shape=(n_data,),
                     p=jnp.array([0.1, 0.2, 0.3, 0.4]))

# Initialize model
n_hidden = 5
hidden_size = 25

class MLP(nn.Module):
    hidden_size: int
    n_hidden: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.n_hidden):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.swish(x)
        x = nn.Dense(1,use_bias=False)(x)
        return x

model = MLP(hidden_size=hidden_size, n_hidden=n_hidden)
key, key_params = jrandom.split(key,2)
params = model.init(key_params, jnp.ones([1, 2]))

# Initialize samples
bounds = (-15,15)
batch_size = 100

key, key_sample = jrandom.split(key,2)
samples = jrandom.uniform(key_sample, shape=(batch_size, 2), minval=bounds[0], maxval=bounds[1])

# Define training and sampling update
@jax.jit
def get_train_grads(params, data_batch, sample_batch):
    def loss_fn(p):
        return jnp.mean(model.apply(p, data_batch)) \
            - jnp.mean(model.apply(p, sample_batch))
    grad_loss = jax.grad(loss_fn)
    grads = grad_loss(params)
    return grads

@partial(jax.jit, static_argnames=('n_steps'))
def sampling_step(params, samples, key, step_size=0.1, n_steps=10):
    for i in range(n_steps):
        key, key_step = jrandom.split(key, 2)
        samples = tinymcmc.step_mala(
            key_step, 
            lambda x: model.apply(params, x)[0],
            samples,
            step_size, bounds=bounds, metropolize=False)
    return samples

# Initialize and train optimizer
learning_rate = 0.00005
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

for i in tqdm.tqdm(range(5000)):
    key, key_sample, key_batch, key_tempering = jrandom.split(key, 4)
    samples = sampling_step(params, samples, key_sample)
    data_batch = data[jrandom.choice(key_batch, n_data, shape=(batch_size,), replace=False)]
    grads = get_train_grads(params, data_batch, samples)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

# Plot output

def plot(key):
    xx,yy = np.meshgrid(np.linspace(*bounds,101),np.linspace(*bounds,101))
    zz = model.apply(params, np.array([xx,yy]).reshape(2,-1).T).reshape(xx.shape)
    key, key_batch = jrandom.split(key, 2)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,4))
    axes[0].plot(data[jrandom.choice(key_batch, n_data, shape=(batch_size,), replace=False),0], 
                 data[jrandom.choice(key_batch, n_data, shape=(batch_size,), replace=False),1],'k.',alpha=1.)
    axes[0].set_title('Data (random batch)')
    axes[0].set_aspect('equal')

    axes[1].imshow(zz, origin='lower', extent=bounds+bounds, cmap='Reds')
    axes[1].plot(samples[:,0], samples[:,1],'k.',alpha=1.)
    axes[1].set_title('Model')
    axes[1].set_aspect('equal')
    plt.show()

key, key_temp = jrandom.split(key)
plot(key_temp)


