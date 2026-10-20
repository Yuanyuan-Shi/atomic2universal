"""Correct actor -> rollout weight sync for RLinf's embodied FSDP actor (patched in by sitecustomize.py).

RLinf's EmbodiedFSDPActor.get_rollout_state_dict hands the weight syncer the actor's *sharded* state dict
(get_model_state_dict(full_state_dict=False)); the syncer rebuilds each tensor with DTensor.full_tensor(). With
FSDP (sharding_strategy shard_grad_op) over 8 GPUs this returns 17 GR00T DiT tensors of shape (3072, 1536)
(transformer_blocks.*.norm1.linear.weight, proj_out_1.weight) with their values in the wrong order, so every
rollout worker ran a corrupted DiT while the actor trained the correct one: PPO log-ratios of -7 to -25 before
any update, approx_kl ~8 (Phase C diagnosis, docs/environment.md). 2 GPUs happened to be unaffected.
The fix returns the full state dict instead (full_state_dict=True: gathered by FSDP, the path that writes the
correct full_weights.pt), so the syncer receives plain, correctly laid-out tensors; training is unchanged.
"""

from types import ModuleType


def patch(module: ModuleType) -> None:
    actor_class = module.EmbodiedFSDPActor
    if not callable(getattr(actor_class, "get_rollout_state_dict", None)):
        raise RuntimeError("[a2u weight sync fix] EmbodiedFSDPActor.get_rollout_state_dict not found; RLinf changed")

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=True)

    actor_class.get_rollout_state_dict = get_rollout_state_dict
