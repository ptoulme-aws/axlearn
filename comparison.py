import jax.numpy as jnp

c2 = jnp.load("./updated_compare/updated_states_cpu_step1.npy",allow_pickle=True)

n2 = jnp.load("updated_states_neuron_step1.npy",allow_pickle=True)

print(c2-n2)
print((c2-n2).max())

c2_ = jnp.load("updated_states_cpu_step3.npy",allow_pickle=True)

n2_ = jnp.load("updated_states_neuron_step3.npy",allow_pickle=True)

print(c2_)
print(n2_)
print(c2_-n2_)
print((c2_-n2_).max())