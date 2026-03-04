import itertools

seeds = [0, 1]
sparsities = [0.9]  # shared
actor_num_blocks = [1, 2]
actor_hidden_dims = [128, 512]
masking_type = ["sample", "percentile"]

envs = ["humanoid-run", "humanoid-walk", "dog-run", "dog-walk"]
envs = ["humanoid-run", "dog-run"]

commands = []

for (
    seed,
    sparsity,
    a_blocks,
    a_dim,
    env,
    m_type,
) in itertools.product(
    seeds,
    sparsities,
    actor_num_blocks,
    actor_hidden_dims,
    envs,
    masking_type,
):

    # Enforce structural ratios
    c_blocks = 2 * a_blocks
    c_dim = 4 * a_dim

    cmd = (
        f"python run.py --config_name base_sac "
        f"--overrides seed={seed} "
        f"--overrides actor_sparsity={sparsity} "
        f"--overrides critic_sparsity={sparsity} "
        f"--overrides actor_num_blocks={a_blocks} "
        f"--overrides actor_hidden_dim={a_dim} "
        f"--overrides critic_num_blocks={c_blocks} "
        f"--overrides critic_hidden_dim={c_dim} "
        f"--overrides env_name={env} "
        f"--overrides masking_type={m_type}"
    )

    commands.append(cmd)

# ---- Split into two experiment sets ----
mid = len(commands) // 2
set1 = commands[:mid]
set2 = commands[mid:]

with open("hyperparameter_set1.sh", "w") as f:
    f.write("\n".join(set1))

with open("hyperparameter_set2.sh", "w") as f:
    f.write("\n".join(set2))

print(f"Total experiments: {len(commands)}")
print(f"Set1: {len(set1)}")
print(f"Set2: {len(set2)}")