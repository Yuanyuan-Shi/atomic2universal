"""Fast CPU checks of `distillation.gr00t.wrapper` on a tiny randomly initialized GR00T action head (the fork's own
classes, small sizes): batched, length-grouped `flow_solve` equals the fork's `get_action` per sample with the same
noise (and treating left padding as valid tokens does not, so grouping matters); continuous t reaches the DiT as
fp32-discretized buckets; the `process_backbone_output` memo gives the same outputs and gradients as recomputing
(and is actually used) and is recomputed after a backward through it, an optimizer step, a parameter edit, a
grad-mode switch or a new or modified cond; the module cannot leave eval mode. The GPU integration tests on the
real checkpoints are separate (Phase C)."""

import contextlib
import copy
from unittest import mock

import pytest
import torch
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig
from torch import Tensor
from transformers.feature_extraction_utils import BatchFeature

from distillation.flow_utils import flow_solve
from distillation.gr00t.wrapper import GR00TVelocity

BLOCK = dict(attention_head_dim=4, num_attention_heads=4, num_layers=2, dropout=0.2, final_dropout=True,
             positional_embeddings=None)
TINY_HEAD = dict(action_dim=4, action_horizon=3, add_pos_embed=True, backbone_embedding_dim=16, hidden_size=12,
                 input_embedding_dim=16, max_state_dim=5, max_num_embodiments=2, max_seq_len=8,
                 num_inference_timesteps=4, num_target_vision_tokens=2, num_timestep_buckets=1000, use_vlln=True,
                 vl_self_attention_cfg=BLOCK,
                 diffusion_model_cfg=dict(BLOCK, cross_attention_dim=16, interleave_self_attention=True,
                                          norm_type="ada_norm", output_dim=12))
LENGTHS = [5, 7, 5, 6]  # valid backbone lengths of the batch: three groups


def tiny_model(seed: int = 0) -> GR00TVelocity:
    """GR00TVelocity around a tiny fp32 head whose parameters are N(0, 0.3²) (non-trivial outputs)."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        head = FlowmatchingActionHead(FlowmatchingActionHeadConfig(**TINY_HEAD))
        for p in head.parameters():
            p.data.normal_(0.0, 0.3)
    return GR00TVelocity(head)


def tiny_cond(lengths: list[int] = LENGTHS, seed: int = 1) -> dict[str, Tensor]:
    """Left-padded cond with random (non-zero) padding features, as `BackboneFeatures` lays it out."""
    g = torch.Generator().manual_seed(seed)
    batch_size, t_max = len(lengths), max(lengths)
    mask = torch.arange(t_max)[None, :] >= t_max - torch.tensor(lengths)[:, None]
    return {"backbone_features": torch.randn(batch_size, t_max, 16, generator=g), "backbone_attention_mask": mask,
            "state": torch.randn(batch_size, 1, 5, generator=g), "embodiment_id": torch.arange(batch_size) % 2}


def noise(batch_size: int, seed: int = 2) -> Tensor:
    """Standard normal fp32 actions [B, 3, 4]."""
    return torch.randn(batch_size, 3, 4, generator=torch.Generator().manual_seed(seed))


def get_action_reference(model: GR00TVelocity, cond: dict[str, Tensor], b: int, z: Tensor) -> Tensor:
    """The fork's own sampler on sample b alone (unpadded, B = 1), with its noise replaced by z (recon §1)."""
    length = int(cond["backbone_attention_mask"][b].sum())
    backbone = BatchFeature({"backbone_features": cond["backbone_features"][b:b + 1, -length:]})
    action_input = BatchFeature({"state": cond["state"][b:b + 1], "embodiment_id": cond["embodiment_id"][b:b + 1]})
    with torch.autocast("cpu", dtype=torch.bfloat16), \
            mock.patch("torch.randn", lambda *, size, dtype, device: z[None].to(device, dtype)):
        return model.action_head.get_action(backbone, action_input)["action_pred"][0]


def count_vl_attention_calls(model: GR00TVelocity) -> list[int]:
    """A one-element counter incremented by every vl_self_attention forward (i.e. process_backbone_output run)."""
    calls = [0]
    model.action_head.vl_self_attention.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    return calls


def test_grouped_flow_solve_equals_get_action_per_sample() -> None:
    """Batched 4-step flow_solve == the fork's get_action on each sample alone; all-valid mask (padding as tokens)
    changes the shorter samples (negative control)."""
    model, cond, z = tiny_model(), tiny_cond(), noise(len(LENGTHS))
    a = flow_solve(model, z, cond, 4, requires_grad=False)
    for b in range(len(LENGTHS)):
        assert torch.equal(a[b], get_action_reference(model, cond, b, z[b]))
    leaky = dict(cond, backbone_attention_mask=torch.ones_like(cond["backbone_attention_mask"]))
    short = [b for b, n in enumerate(LENGTHS) if n < max(LENGTHS)]
    assert (flow_solve(model, z, leaky, 4, requires_grad=False)[short] - a[short]).abs().amax() > 1e-3


def test_right_padded_mask_raises() -> None:
    """A mask whose valid tokens are not the last T fails loudly."""
    cond = tiny_cond()
    cond["backbone_attention_mask"] = cond["backbone_attention_mask"].flip(1)
    with pytest.raises(ValueError, match="left padded"):
        tiny_model().velocity(noise(len(LENGTHS)), torch.zeros(len(LENGTHS)), cond)


def test_time_discretization_in_fp32() -> None:
    """t → (t·1000).long() in fp32, clamped to 999: bf16 t = 0.75 gives 750 (not 752), t = 1 gives 999."""
    model, seen = tiny_model(), []
    model.action_head.model.register_forward_pre_hook(lambda _, args, kwargs: seen.append(kwargs["timestep"]),
                                                      with_kwargs=True)
    t = torch.tensor([0.0, 0.25, 0.75, 1.0])
    for t_in in (t, t.to(torch.bfloat16)):
        model.velocity(noise(4), t_in, tiny_cond([5, 5, 5, 5]))
        assert seen[-1].tolist() == [0, 250, 750, 999] and seen[-1].dtype == torch.long


def memo_run(model: GR00TVelocity, cond: dict[str, Tensor]) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """distill_step's student pattern: a 4-step solve with gradients, then one more velocity call on the same cond,
    one backward through both; returns the sample, the velocity and the parameter gradients."""
    dtype = cond["backbone_features"].dtype
    a = flow_solve(model, noise(len(LENGTHS)).to(dtype), cond, 4, requires_grad=True)
    v = model.velocity(a.detach(), torch.full((len(LENGTHS),), 0.3, dtype=dtype), cond)
    (a.square().sum() + v.square().sum()).backward()
    return a, v, {name: p.grad for name, p in model.named_parameters()}


def memo_and_plain(dtype: torch.dtype) -> tuple[GR00TVelocity, GR00TVelocity, list[int], list[int]]:
    """Two identical tiny models in `dtype`, one memoizing and one recomputing, with their call counters."""
    memo_model = tiny_model().to(dtype)
    plain = copy.deepcopy(memo_model)
    plain.memoize = False
    return memo_model, plain, count_vl_attention_calls(memo_model), count_vl_attention_calls(plain)


def test_memo_matches_recomputation_outputs_and_grads() -> None:
    """Memoized vs recomputing model: under bf16 autocast (as used) the outputs are bitwise equal and the memo runs
    process_backbone_output once per group (3) instead of once per call and group (15); in float64 without autocast
    (no bf16 rounding of one summed vs four separate backward passes) the gradients agree to rtol 1e-9."""
    memo_model, plain, memo_calls, plain_calls = memo_and_plain(torch.float32)
    (a_m, v_m, _), (a_p, v_p, _) = memo_run(memo_model, tiny_cond()), memo_run(plain, tiny_cond())
    assert torch.equal(a_m, a_p) and torch.equal(v_m, v_p)
    assert memo_calls[0] == 3 and plain_calls[0] == 15
    memo_model, plain, _, _ = memo_and_plain(torch.float64)
    cond = {key: value.double() if value.is_floating_point() else value for key, value in tiny_cond().items()}
    with mock.patch("torch.autocast", lambda **_: contextlib.nullcontext()):
        (_, _, g_m), (_, _, g_p) = memo_run(memo_model, cond), memo_run(plain, cond)
    for name, grad in g_p.items():  # atol: key biases have an exactly zero gradient (softmax shift invariance)
        torch.testing.assert_close(g_m[name], grad, rtol=1e-9, atol=1e-12, msg=lambda m: f"{name}: {m}")


def test_memo_recomputes_after_backward_step_grad_mode_or_cond_change() -> None:
    """A hit adds no vl_self_attention call (3 groups → 3 calls per recomputation). Recomputed: after a backward
    through the memo (its graph is freed, so a second backward works), a grad-mode switch, a fused AdamW step
    (which does not bump p._version), an in-place parameter edit, a new feature tensor and an in-place feature
    edit; after the step the result equals a fresh model's."""
    model, cond, x, t = tiny_model(), tiny_cond(), noise(len(LENGTHS)), torch.full((len(LENGTHS),), 0.5)
    calls = count_vl_attention_calls(model)
    v = model.velocity(x, t, cond)
    model.velocity(x, t, cond)
    assert calls[0] == 3
    assert copy.deepcopy(model)._memo is None  # a memo holding graph tensors does not block copies (fake = copy)
    v.square().sum().backward()
    model.velocity(x, t, cond).square().sum().backward()
    assert calls[0] == 6
    with torch.no_grad():
        model.velocity(x, t, cond)
        model.velocity(x, t, cond)
        assert calls[0] == 9
        versions = [p._version for p in model.parameters()]
        torch.optim.AdamW(model.parameters(), lr=1e-2, fused=True).step()
        assert versions == [p._version for p in model.parameters()]
        fresh = tiny_model()
        fresh.load_state_dict(model.state_dict())
        assert torch.equal(model.velocity(x, t, cond), fresh.velocity(x, t, cond)) and calls[0] == 12
        model.action_head.vlln.weight.mul_(1.5)
        model.velocity(x, t, cond)
        assert calls[0] == 15
        copied = dict(cond, backbone_features=cond["backbone_features"].clone())
        model.velocity(x, t, copied)
        model.velocity(x, t, copied)
        assert calls[0] == 18
        copied["backbone_features"].mul_(2.0)
        model.velocity(x, t, copied)
        assert calls[0] == 21


def test_eval_mode_is_enforced() -> None:
    """.train() keeps every submodule in eval mode (dropout off, repeated calls agree); a submodule switched to
    training mode behind the wrapper's back makes velocity raise."""
    model, cond, x, t = tiny_model(), tiny_cond(), noise(len(LENGTHS)), torch.full((len(LENGTHS),), 0.5)
    model.memoize = False
    model.train()
    assert not any(module.training for module in model.modules())
    assert torch.equal(model.velocity(x, t, cond), model.velocity(x, t, cond))
    model.action_head.model.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        model.velocity(x, t, cond)

