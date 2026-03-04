import jax
import jax.numpy as jnp
import flax.linen as nn
from scale_rl.agents.sparse import get_sparsities_erdos_renyi, get_var_shape_dict, generate_masks_jax, percentile_mask_from_avg
import flax

def test_percentile_er_sparsity():
    print("Verifying ER compliance for Percentile Masking...")
    
    class LargeNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(512)(x)
            x = nn.relu(x)
            x = nn.Dense(256)(x)
            x = nn.relu(x)
            x = nn.Dense(10)(x)
            return x

    net = LargeNet()
    rng = jax.random.PRNGKey(42)
    params = net.init(rng, jnp.ones((1, 100)))['params']
    
    target_sparsity = 0.8
    sparsities = get_sparsities_erdos_renyi(get_var_shape_dict(params), default_sparsity=target_sparsity)
    
    # Generate a batch of masks and average them
    batch_size = 100
    seeds = jax.random.split(rng, batch_size)
    batched_gen_masks = jax.vmap(generate_masks_jax, in_axes=(0, None, None))
    masks = batched_gen_masks(seeds, params, sparsities)
    avg_mask = jax.tree.map(lambda x: jnp.mean(x, axis=0), masks)
    
    # Apply percentile masking
    final_mask = percentile_mask_from_avg(avg_mask, sparsities)
    
    # Compute global sparsity
    flat_params, _ = jax.tree_util.tree_flatten(params)
    flat_masks, _ = jax.tree_util.tree_flatten(final_mask)
    
    total_params = sum(p.size for p in flat_params)
    total_nonzero = sum(jnp.sum(m) for m in flat_masks)
    
    empirical_sparsity = 1.0 - (total_nonzero / total_params)
    
    print(f"Target Sparsity: {target_sparsity}")
    print(f"Empirical Sparsity: {empirical_sparsity:.4f}")
    
    # Layer-wise check
    flat_sparsities, _ = jax.tree_util.tree_flatten(sparsities)
    for i, (m, s) in enumerate(zip(flat_masks, flat_sparsities)):
        if s is not None:
            layer_sparsity = 1.0 - (jnp.sum(m) / m.size)
            print(f"Layer {i}: Target={s:.4f}, Empirical={layer_sparsity:.4f}")
            assert jnp.isclose(layer_sparsity, s, atol=1e-3), f"Layer {i} sparsity mismatch!"

    assert jnp.isclose(empirical_sparsity, target_sparsity, atol=0.01), "Global sparsity mismatch!"
    print("ER verification passed!")

if __name__ == "__main__":
    test_percentile_er_sparsity()
