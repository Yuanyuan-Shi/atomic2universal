"""RLinf's batch divisibility rules for embodied PPO with actor, env and rollout colocated on every GPU.

The same assertions RLinf makes at startup or in its first update (docs/rl_recon.md Q7:
rlinf/config.py:866-892 and 1281-1287, workers/actor/fsdp_actor_worker.py:1688-1724), printed with
the numbers so a bad hardware file fails before anything starts. Batch sizes count chunk samples;
rollout_size_per_actor_rank = envs/world * (max_steps_per_rollout_epoch/num_action_chunks) * epochs.
build_config.py runs them on every config; run this file to try other sizes, e.g.
    python expert_training/batch_rules.py --envs 128 --epochs 4 --global-batch 512 --micro-batch 16 --gpus 8
"""

import argparse
import sys


def check(envs: int, epochs: int, global_batch: int, micro_batch: int, world: int, steps: int, chunk: int,
          stages: int = 1, group: int = 1) -> list[str]:
    """Print every rule with its numbers and return the violated ones (empty: all hold)."""
    rollout_size = envs // world * (steps // chunk) * epochs  # chunk samples per actor rank per iteration
    per_rank, per_stage = global_batch // world, envs // world // stages
    rules = [
        (f"max_steps_per_rollout_epoch % num_action_chunks = {steps} % {chunk}", steps % chunk),
        (f"total_num_envs % world_size = {envs} % {world}", envs % world),
        (f"total_num_envs // world_size // pipeline_stage_num % group_size = {per_stage} % {group} (> 0)",
         per_stage % group if per_stage > 0 else -1),
        (f"global_batch % (micro_batch * world_size) = {global_batch} % ({micro_batch} * {world})",
         global_batch % (micro_batch * world)),
        (f"rollout_size_per_actor_rank % (global_batch / world_size) = ({envs}/{world} envs * {steps // chunk} chunks"
         f" * {epochs} epochs = {rollout_size}) % ({global_batch}/{world} = {per_rank})",
         rollout_size % per_rank if per_rank else -1),
    ]
    print("RLinf batch divisibility check:")
    for text, remainder in rules:
        print(f"  {'OK  ' if remainder == 0 else 'FAIL'} {text} = {remainder}")
    print(f"  -> {envs * epochs} rollouts/iteration, {rollout_size // max(per_rank, 1)} optimizer steps/iteration, "
          f"gradient accumulation {global_batch // max(micro_batch * world, 1)}")
    return [text for text, remainder in rules if remainder != 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("envs", "epochs", "global-batch", "micro-batch", "gpus"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--steps", type=int, default=752, help="max_steps_per_rollout_epoch (default: 752 = 47 x 16)")
    parser.add_argument("--chunk", type=int, default=16, help="num_action_chunks (default: 16)")
    args = parser.parse_args()
    if any(value < 1 for value in vars(args).values()):
        parser.error("all values must be positive")
    failures = check(args.envs, args.epochs, args.global_batch, args.micro_batch, args.gpus, args.steps, args.chunk)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
