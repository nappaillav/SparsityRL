import jax
import jax.numpy as jnp
from scale_rl.networks.optimizer import masked_adam
from scale_rl.networks.trainer import Trainer
import flax.linen as nn
import optax
import flax

def test_masked_adam_behavior():
    print("Starting Masked Adam behavior verification...")
    # 1. Setup a simple network
    class SimpleNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(2, use_bias=False)(x)
    
    net = SimpleNet()
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 2))
    params = net.init(key, x)['params']
    
    # 2. Setup Trainer with masked_adam
    tx = masked_adam(1e-1)
    trainer = Trainer.create(
        network_def=net,
        network_inputs={'rngs': key, 'x': x},
        tx=tx,
        sparse=True
    )
    
    # Initial state
    initial_mu = trainer.opt_state.mu['Dense_0']['kernel']
    print(f"Initial mu all zero: {jnp.all(initial_mu == 0)}")
    
    # 3. Create a mask where first row is 1, second row is 0
    # Shape of kernel for Dense(2) with input(2) is (2, 2)
    mask = {'Dense_0': {'kernel': jnp.array([[1.0, 1.0], [0.0, 0.0]])}}
    trainer = trainer.replace(network_mask=flax.core.freeze(mask))
    
    # 4. Apply gradient
    def loss_fn(p):
        return jnp.sum(net.apply({'params': p}, x)), {}
    
    new_trainer, info = trainer.apply_gradient(loss_fn)
    
    # 5. Check mu (momentum)
    # The first row of mu should be updated (non-zero)
    # The second row of mu should remain zero
    new_mu = new_trainer.opt_state.mu['Dense_0']['kernel']
    print(f"New mu row 0 updated: {jnp.any(new_mu[0] != 0)}")
    print(f"New mu row 1 still zero: {jnp.all(new_mu[1] == 0)}")
    
    assert jnp.any(new_mu[0] != 0), "Row 0 of mu should be updated"
    assert jnp.all(new_mu[1] == 0), "Row 1 of mu should remain zero"
    
    # 6. Check params
    # First row of params should change, second should not
    initial_params = trainer.params['Dense_0']['kernel']
    new_params = new_trainer.params['Dense_0']['kernel']
    print(f"Params row 0 changed: {jnp.any(new_params[0] != initial_params[0])}")
    print(f"Params row 1 unchanged: {jnp.all(new_params[1] == initial_params[1])}")
    
    assert jnp.any(new_params[0] != initial_params[0]), "Row 0 of params should change"
    assert jnp.all(new_params[1] == initial_params[1]), "Row 1 of params should remain unchanged"

    print("\nVerification passed successfully!")

if __name__ == "__main__":
    test_masked_adam_behavior()
