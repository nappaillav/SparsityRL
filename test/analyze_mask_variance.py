import jax
import jax.numpy as jnp
import flax.linen as nn
from scale_rl.agents.sparse import get_sparsities_erdos_renyi, get_var_shape_dict, generate_masks_jax, percentile_mask_from_avg, sample_mask_from_avg
import flax

def analyze_variance(masking_type="percentile"):
    print(f"\n--- Variance Analysis ({masking_type}) ---")
    
    class SimpleNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(100)(nn.Dense(100)(x))

    net = SimpleNet()
    rng = jax.random.PRNGKey(0)
    params = net.init(rng, jnp.ones((1, 100)))['params']
    target_sparsity = 0.8 # Keep 10%
    sparsities = get_sparsities_erdos_renyi(get_var_shape_dict(params), default_sparsity=target_sparsity)
    
    batch_sizes = [1, 4, 16, 64, 256]
    num_trials = 20
    
    for bs in batch_sizes:
        masks = []
        for i in range(num_trials):
            trial_rng = jax.random.PRNGKey(i)
            seeds = jax.random.split(trial_rng, bs)
            batched_gen_masks = jax.vmap(generate_masks_jax, in_axes=(0, None, None))
            sample_masks = batched_gen_masks(seeds, params, sparsities)
            avg_mask = jax.tree.map(lambda x: jnp.mean(x, axis=0), sample_masks)
            
            if masking_type == "percentile":
                mask = percentile_mask_from_avg(avg_mask, sparsities)
            else: # sample
                mask = sample_mask_from_avg(avg_mask, jax.random.PRNGKey(i + 1000))
            
            masks.append(mask)
            
        # Analyze masks for the first layer (kernel)
        # Find the first layer that has a kernel
        kernel_path = None
        for path, val in flax.traverse_util.flatten_dict(params).items():
            if path[-1] == 'kernel':
                kernel_path = path
                break
        
        layer_masks = jnp.stack([m[kernel_path[0]][kernel_path[1]] for m in masks]) # [num_trials, ...]
        
        # 1. Calculate average overlap between any two masks
        overlaps = []
        for i in range(num_trials):
            for j in range(i + 1, num_trials):
                overlap = jnp.sum(layer_masks[i] * layer_masks[j]) / jnp.sum(layer_masks[i])
                overlaps.append(overlap)
        avg_overlap = jnp.mean(jnp.array(overlaps)) if overlaps else 1.0
        
        # 2. Total fraction of weights that were ever active
        ever_active = jnp.any(layer_masks > 0, axis=0)
        fraction_ever_active = jnp.mean(ever_active)
        
        # 3. Average variance of weights
        # weight_variance = jnp.var(layer_masks, axis=0)
        # avg_weight_variance = jnp.mean(weight_variance)
        
        print(f"Batch Size {bs:3d}: Overlap={avg_overlap:.4f}, Ever Active={fraction_ever_active:.4f}")

if __name__ == "__main__":
    analyze_variance("percentile")
    analyze_variance("sample")
