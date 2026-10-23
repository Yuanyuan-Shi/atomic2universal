# 100-episode comparison (pnp_cabinet_drawer, split target, seeds 0–99, no video)

| Model | Task | Checkpoint | Successes | SR | Wilson 95% CI |
|---|---|---|---|---|---|
| baseline | PickPlaceCounterToCabinet | `checkpoints/sft_joint/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000` | 67/100 | 67% | [57%, 75%] |
| baseline | PickPlaceDrawerToCounter | `checkpoints/sft_joint/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000` | 59/100 | 59% | [49%, 68%] |
| experts | PickPlaceCounterToCabinet | `checkpoints/experts/PickPlaceCounterToCabinet/step_95` | 78/100 | 78% | [69%, 85%] |
| experts | PickPlaceDrawerToCounter | `checkpoints/experts/PickPlaceDrawerToCounter/step_100` | 69/100 | 69% | [59%, 77%] |

## Expert gain over baseline (Newcombe hybrid-score 95% CI)

| Task | Expert − baseline | 95% CI | CI excludes 0 |
|---|---|---|---|
| PickPlaceCounterToCabinet | +11 pp | [-1, +23] pp | no |
| PickPlaceDrawerToCounter | +10 pp | [-3, +23] pp | no |

The models share seeds 0–99, but the policy samples noise on every query, so the runs are treated as independent samples; with positively correlated outcomes these CIs are conservative.
