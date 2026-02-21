import jax.numpy as jnp
import jax.random as jr
import jax
import time 
from tqdm import tqdm

# @jax.jit
def make_mask(key, sparsity=0.5):
    # use the ER mask function
    return jr.bernoulli(key, p=sparsity, shape=(1024,1024))

m = 1024
keys = jr.split(jr.PRNGKey(42), m)
print(keys.dtype)
start = time.time()
# for i in tqdm(range(1000)):
masks = jax.vmap(make_mask)(keys)
end = time.time()

print(end - start)
