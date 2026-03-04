import optax
import jax.numpy as jnp
import jax
from typing import Any, NamedTuple
from flax.core import unfreeze

class MaskedAdamState(NamedTuple):
  """State for the Masked Adam optimizer."""
  count: jnp.ndarray
  mu: optax.Updates
  nu: optax.Updates

def masked_adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8):
    """Custom Adam optimizer that handles masking by freezing state for masked weights."""
    
    def init_fn(params):
        return MaskedAdamState(
            count=jnp.zeros([], jnp.int32),
            mu=jax.tree_util.tree_map(jnp.zeros_like, params),
            nu=jax.tree_util.tree_map(jnp.zeros_like, params)
        )

    def update_fn(updates, state, params=None, mask=None):
        # Ensure we are working with plain dicts to avoid structure mismatch errors
        # when mixing dict and FrozenDict in tree_map
        # takes the params and updates with the learning rate
        updates = unfreeze(updates) if hasattr(updates, 'unfreeze') else updates
        mask = unfreeze(mask) if hasattr(mask, 'unfreeze') else mask

        if mask is None:
            # Default to all ones if no mask provided
            mask = jax.tree_util.tree_map(jnp.ones_like, updates)

        count = state.count + 1
        
        # 1. Update Mu (momentum)
        new_mu = jax.tree_util.tree_map(
            lambda m, g, mk: jnp.where(mk > 0, b1 * m + (1 - b1) * g, m),
            state.mu, updates, mask
        )
        
        # 2. Update Nu (variance)
        new_nu = jax.tree_util.tree_map(
            lambda v, g, mk: jnp.where(mk > 0, b2 * v + (1 - b2) * (g**2), v),
            state.nu, updates, mask
        )
        
        # 3. Compute bias-corrected updates
        def compute_update(m, v, mk):
            m_hat = m / (1 - b1**count)
            v_hat = v / (1 - b2**count)
            u = -learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
            return jnp.where(mk > 0, u, 0.0)

        new_updates = jax.tree_util.tree_map(
            compute_update, new_mu, new_nu, mask
        )

        return new_updates, MaskedAdamState(count=count, mu=new_mu, nu=new_nu)

    return optax.GradientTransformation(init_fn, update_fn)
