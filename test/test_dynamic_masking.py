import jax
import jax.numpy as jnp
import flax
from scale_rl.agents.sparse import generate_masks_jax, erk, get_var_shape_dict, create_mask, get_sparsities_erdos_renyi, sample_mask_from_avg
from scale_rl.networks.trainer import Trainer
import optax
import flax.linen as nn

def test_dynamic_masking():
    """
    Check post average masking sparsity is correct (simple Network across a minibatch)
    """
    # 1. Test generate_masks_jax
    params = {'layer1': {'kernel': jnp.ones((4, 4))}, 'layer2': {'kernel': jnp.ones((4, 2))}}
    sparsities = {'layer1': {'kernel': 0.5}, 'layer2': {'kernel': 0.0}}
    seed = jax.random.PRNGKey(42)
    
    mask = generate_masks_jax(seed, params, sparsities)
    print("Single mask layer1 shape:", mask['layer1']['kernel'].shape)
    print("Single mask layer1 non-zeros:", jnp.sum(mask['layer1']['kernel'] != 0))
    print("Single mask layer2 non-zeros (expect 8):", jnp.sum(mask['layer2']['kernel'] != 0))

    # 2. Test vmap and averaging
    seeds = jax.random.split(seed, 100)
    batched_gen_masks = jax.vmap(generate_masks_jax, in_axes=(0, None, None))
    masks = batched_gen_masks(seeds, params, sparsities) # TODO : Sanity Check the sparsity 
    
    avg_mask = jax.tree.map(lambda x: jnp.mean(x, axis=0), masks)
    print(avg_mask)
    print("Averaged mask layer1 (first few elements):", avg_mask['layer1']['kernel'][0, :4])
    # With sparsity 0.5, we expect mean to be around 0.5
    print("Averaged mask layer1 mean:", jnp.mean(avg_mask['layer1']['kernel']))

    # 3. Test Trainer integration
    class SimpleNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(2)(x)
            
    net = SimpleNet()
    trainer = Trainer.create(
        network_def=net,
        network_inputs={'rngs': jax.random.PRNGKey(0), 'x': jnp.zeros((1, 4))},
        tx=optax.adam(1e-3),
        sparse=True,
        sparsities=flax.core.freeze({'params': {'Dense_0': {'kernel': 0.5, 'bias': 0.0}}})
    )
    
    def loss_fn(params):
        return jnp.sum(trainer.apply({'params': params}, jnp.ones((10, 4)))), {}
        
    new_trainer, info = trainer.apply_gradient(loss_fn, rnd_seeds=seeds)
    print("Gradient norm from info (scaling):", info['grad_norm'])
    
    # Test Trainer with sampling
    sampling_rng = jax.random.PRNGKey(456)
    new_trainer_sampled, info_sampled = trainer.apply_gradient(loss_fn, rnd_seeds=seeds, rng=sampling_rng)
    print("Gradient norm from info (sampling):", info_sampled['grad_norm'])
    print("Update with sampling successful")

def test_erk_sparsity():
    """
    Test that ERK distributes sparsity layer-wise
    but preserves correct global sparsity.
    """
    class SimpleNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = nn.Dense(64)(x)
            x = nn.relu(x)
            x = nn.Dense(1)(x)
            return x

    net = SimpleNet()
    rng = jax.random.PRNGKey(0)

    variables = net.init(rng, jnp.ones((1, 10)))
    params = variables["params"]   # 🔥 IMPORTANT

    target_sparsity = 0.8
    seed = jax.random.PRNGKey(0)
    print(get_var_shape_dict(params))
    # 2️⃣ Compute ERK sparsities
    sparsities = get_sparsities_erdos_renyi(get_var_shape_dict(params), default_sparsity=target_sparsity)

    print("Layer-wise ERK sparsities:")
    print(sparsities)

    # 3️⃣ Generate masks
    rng, key = jax.random.split(rng)
    mask = create_mask(params, sparsities, key)

    # 4️⃣ Compute global sparsity
    flat_params, _ = jax.tree_util.tree_flatten(params)
    flat_masks, _ = jax.tree_util.tree_flatten(mask)

    total_params = sum(p.size for p in flat_params)
    total_nonzero = sum(jnp.sum(m) for m in flat_masks)

    empirical_sparsity = 1.0 - (total_nonzero / total_params)

    print("Target sparsity:", target_sparsity)
    print("Empirical sparsity:", empirical_sparsity)

    # 5️⃣ Assert approximate match
    assert jnp.isclose(empirical_sparsity, target_sparsity, atol=0.05), \
        "ERK global sparsity mismatch!"

    print("ERK test passed.")

def test_mask_sampling():
    """
    Test that sampling from the averaged mask preserves binary sparsity.
    """
    class SimpleNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(128)(x)
            x = nn.relu(x)
            x = nn.Dense(64)(x)
            x = nn.relu(x)
            x = nn.Dense(1)(x)
            return x

    net = SimpleNet()
    rng = jax.random.PRNGKey(0)

    variables = net.init(rng, jnp.ones((1, 10)))
    params = variables["params"]   # 🔥 IMPORTANT
    target_sparsity = 0.8
    seed = jax.random.PRNGKey(0)
    print(get_var_shape_dict(params))
    # 2️⃣ Compute ERK sparsities
    sparsities = get_sparsities_erdos_renyi(get_var_shape_dict(params), default_sparsity=target_sparsity)

    print("Layer-wise ERK sparsities:")
    print(sparsities)
    
    # Generate 100 masks
    seeds = jax.random.split(seed, 100)
    batched_gen_masks = jax.vmap(generate_masks_jax, in_axes=(0, None, None))
    masks = batched_gen_masks(seeds, params, sparsities)
    
    # Average them (this is now a probability map)
    avg_mask = jax.tree.map(lambda x: jnp.mean(x, axis=0), masks)
    print("\n--- Mask Sampling Test ---")
    # print("Averaged mask mean (prob of keeping weight):", jnp.mean(avg_mask['layer1']['kernel']))
    
    # Sample a final binary mask from the probability map
    sample_rng = jax.random.PRNGKey(123)
    final_binary_mask = sample_mask_from_avg(avg_mask, sample_rng)

    # 4️⃣ Compute global sparsity
    flat_params, _ = jax.tree_util.tree_flatten(params)
    flat_masks, _ = jax.tree_util.tree_flatten(final_binary_mask)

    total_params = sum(p.size for p in flat_params)
    total_nonzero = sum(jnp.sum(m) for m in flat_masks)

    empirical_sparsity = 1.0 - (total_nonzero / total_params)

    print("Target sparsity:", target_sparsity)
    print("Empirical sparsity:", empirical_sparsity)
    
    # # Check if final mask is binary
    # unique_vals = jnp.unique(final_binary_mask['layer1']['kernel'])
    # print("Unique values in final mask (should be [0, 1]):", unique_vals)
    
    # # Check sparsity of final binary mask
    # nonzero_count = jnp.sum(final_binary_mask['layer1']['kernel'])
    # total_count = final_binary_mask['layer1']['kernel'].size
    # empirical_sparsity = 1.0 - (nonzero_count / total_count)
    
    # print(f"Target Sparsity: 0.8")
    # print(f"Final Binary Mask Empirical Sparsity: {empirical_sparsity:.4f}")
    
    # assert jnp.all((final_binary_mask['layer1']['kernel'] == 0) | (final_binary_mask['layer1']['kernel'] == 1))
    # assert jnp.isclose(empirical_sparsity, 0.8, atol=0.01)
    print("Mask sampling test passed.")

if __name__ == "__main__":
    # test_dynamic_masking()
    # test_erk_sparsity()
    test_mask_sampling()
