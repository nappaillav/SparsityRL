import jax
import jax.numpy as jnp
from functools import partial

# -------------------------------------------------
# Linear layer
# -------------------------------------------------
def linear(params, x):
    # params["W"]: [in_features, out_dim]
    # x: [batch, in_features]
    return x @ params["W"]


# -------------------------------------------------
# Loss (example: MSE)
# -------------------------------------------------
def loss_fn(params, x, y):
    preds = linear(params, x)
    return jnp.mean((preds - y) ** 2)


# -------------------------------------------------
# Adam state initialization
# -------------------------------------------------
def init_adam_state(params):
    m = jax.tree_map(jnp.zeros_like, params)
    v = jax.tree_map(jnp.zeros_like, params)
    t = 0
    return {"m": m, "v": v, "t": t}


# -------------------------------------------------
# Custom Adam update
# -------------------------------------------------
@partial(jax.jit, static_argnums=())
def adam_update(params, grads, state,
                lr=1e-3,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8):

    t = state["t"] + 1

    # m_t and v_t
    m = jax.tree_map(
        lambda m, g: beta1 * m + (1 - beta1) * g,
        state["m"], grads
    )

    v = jax.tree_map(
        lambda v, g: beta2 * v + (1 - beta2) * (g ** 2),
        state["v"], grads
    )

    # Bias correction
    m_hat = jax.tree_map(lambda m: m / (1 - beta1 ** t), m)
    v_hat = jax.tree_map(lambda v: v / (1 - beta2 ** t), v)

    # Parameter update
    new_params = jax.tree_map(
        lambda p, m_h, v_h: p - lr * m_h / (jnp.sqrt(v_h) + eps),
        params, m_hat, v_hat
    )

    new_state = {"m": m, "v": v, "t": t}

    return new_params, new_state

# Dimensions
in_features = 4
out_dim = 2
batch_size = 8

# Random init
key = jax.random.PRNGKey(0)
W = jax.random.normal(key, (in_features, out_dim)) * 0.01
params = {"W": W}

state = init_adam_state(params)

# Dummy data
x = jax.random.normal(key, (batch_size, in_features))
true_W = jnp.ones((in_features, out_dim))
y = x @ true_W  # linear ground truth

# One training step
grads = jax.grad(loss_fn)(params, x, y)
params, state = adam_update(params, grads, state)