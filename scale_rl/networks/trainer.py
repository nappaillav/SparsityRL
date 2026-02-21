from functools import partial
from typing import Any, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.core import freeze, unfreeze
from scale_rl.networks.utils import tree_norm
from scale_rl.agents.sparse import get_sparsities_erdos_renyi, get_var_shape_dict, create_mask, generate_masks_jax, sample_mask_from_avg
PRNGKey = jnp.ndarray


@flax.struct.dataclass
class Trainer:
    network_def: nn.Module = flax.struct.field(pytree_node=False)
    params: flax.core.FrozenDict[str, Any]
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    update_step: int = 0
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None
    sparse: bool = flax.struct.field(default=False, pytree_node=False)
    #network_mask: Optional[flax.core.FrozenDict[str, Any]] = None  # Add this line
    network_mask: Optional[flax.core.FrozenDict[str, Any]] = flax.struct.field(default=None, pytree_node=False)
    sparsities: Optional[flax.core.FrozenDict[str, Any]] = flax.struct.field(default=None, pytree_node=False)
    """
    dataclass decorator makes custom class to be passed safely to Jax.
    https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html

    Trainer class wraps network & optimizer to easily optimize the network under the hood.

    args:
        network_def:
        params: network parameters.
        tx: optimizer (e.g., optax.Adam).
        opt_state: current state of the optimizer (e.g., beta_1 in Adam).
        update_step: number of update step so far.
    """

    @classmethod
    def create(
        cls,
        network_def: nn.Module,
        network_inputs: flax.core.FrozenDict[str, jnp.ndarray],
        tx: Optional[optax.GradientTransformation] = None,
        dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None,
        sparse = False,
        network_mask = None,
        sparsities = None,
    ) -> "Trainer":
        variables = network_def.init(**network_inputs)
        params = variables.pop("params")
        # if sparse:
        #     params_unfrozen = unfreeze(params)
        #     mask_unfrozen = unfreeze(network_mask)
        #     # set weight to zero
        #     masked_params_unfrozen = jax.tree_util.tree_map(lambda p, m: p * m, params_unfrozen, mask_unfrozen)
        #     params = freeze(masked_params_unfrozen)
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        network = cls(
            network_def=network_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            dynamic_scale=dynamic_scale,
            sparse=sparse,
            network_mask=network_mask,
            sparsities=sparsities
        )

        return network

    def __call__(self, *args, **kwargs):
        return self.network_def.apply({"params": self.params}, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.network_def.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn, rnd_seeds=None, rng=None) -> Tuple[Any, "Trainer"]:
        if self.dynamic_scale:
            grad_fn = self.dynamic_scale.value_and_grad(loss_fn, has_aux=True)
            dynamic_scale, is_fin, (_, info), grads = grad_fn(self.params)
        else:
            grad_fn = jax.grad(loss_fn, has_aux=True)
            grads, info = grad_fn(self.params)
            dynamic_scale = None
            is_fin = True
        if self.sparse and self.sparsities is not None and rnd_seeds is not None:
            # Generate masks per sample
            batched_gen_masks = jax.vmap(generate_masks_jax, in_axes=(0, None, None)) # Define it outside 
            masks = batched_gen_masks(rnd_seeds, self.params, self.sparsities)
            # Average masks across batch dimension
            avg_mask = jax.tree.map(lambda x: jnp.mean(x, axis=0), masks) 
            
            # If rng is provided, sample a binary mask from the averaged probabilities
            if rng is not None:
                final_mask = sample_mask_from_avg(avg_mask, rng)
            else:
                final_mask = avg_mask
                
            # Mask gradients
            grads = jax.tree.map(lambda g, m: g * m, grads, final_mask)
            grad_norm = tree_norm(grads)
            info["grad_norm"] = grad_norm
        elif self.sparse and self.network_mask is not None:
            mask_grad = jax.tree.map(lambda u, m: u * m, grads, self.network_mask)
            grad_norm = tree_norm(mask_grad)
            info["grad_norm"] = grad_norm
        else:
            grad_norm = tree_norm(grads)
            info["grad_norm"] = grad_norm
        # TODO adjust sparse gnorm
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        
        # Mask updates again if we already masked grads, 
        # but the user might want it. Let's keep it consistent with their previous logic for now if network_mask exists.
        if self.sparse and self.network_mask is not None:
            updates = jax.tree.map(lambda u, m: u * m, updates, self.network_mask) #  TODO: Task 1 
        new_params = optax.apply_updates(self.params, updates)

        network = self.replace(
            params=jax.tree_util.tree_map(
                partial(jnp.where, is_fin), new_params, self.params
            ),
            opt_state=jax.tree_util.tree_map(
                partial(jnp.where, is_fin), new_opt_state, self.opt_state
            ),
            update_step=self.update_step + 1,
            dynamic_scale=dynamic_scale,
        )

        return network, info
