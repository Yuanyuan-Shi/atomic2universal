"""GR00T N1.5's action head as a distillation `VelocityModel` (native convention: t = 0 noise, v = action − noise).
`GR00TVelocity.velocity` is one Euler-step body of the fork's sampler `FlowmatchingActionHead.get_action`
(FMH:352-408; FMH = third_party/Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py), built from the
head's own submodules in get_action's order. Batching is the only addition: the fork ignores attention masks (recon
§10.1), so the batch runs per group of equal valid backbone length, each group unpadded (padding is on the left), and
the groups are scattered back in batch order. Continuous t is discretized in fp32 (in bf16, 0.75·1000 becomes 752)
and clamped to the last bucket (t = 1 never occurs in training, recon §1). Everything runs under bf16 autocast as in
the eval (POL:190, POL = third_party/Isaac-GR00T/gr00t/model/policy.py); teachers hold bf16 weights, student and fake
fp32 master weights. Dropout (p = 0.2) is always off: the module never leaves eval mode. `process_backbone_output`
(vlln + vl_self_attention, ≈ 0.35 of the FLOPs, recon §2) is memoized across calls with the same cond; see
`_vl_groups` for when a memoized result is reused."""

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from accelerate import init_empty_weights
from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig
from safetensors import safe_open
from torch import Tensor, nn
from torch.optim.optimizer import register_optimizer_step_post_hook

INDEX_FILE = "model.safetensors.index.json"
_optimizer_steps = 0  # steps of any torch optimizer so far (fused AdamW does not bump p._version)


def _count_optimizer_step(*_: object) -> None:
    """Global post-hook of every torch optimizer step: invalidates all memos."""
    global _optimizer_steps
    _optimizer_steps += 1


register_optimizer_step_post_hook(_count_optimizer_step)


def checkpoint_tensors(
    checkpoint: Path, prefix: str, device: str | torch.device = "cpu"
) -> Iterator[tuple[str, Tensor]]:
    """Yield (name without `prefix`, tensor on `device`) for each tensor of a sharded checkpoint whose name starts
    with `prefix`, one tensor at a time through `safe_open` (never a whole checkpoint in host RAM)."""
    weight_map = json.loads((Path(checkpoint) / INDEX_FILE).read_text())["weight_map"]
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(Path(checkpoint) / shard), framework="pt", device=str(device)) as f:
            for name in f.keys():
                if name.startswith(prefix):
                    yield name.removeprefix(prefix), f.get_tensor(name)


def _valid_length_groups(mask: Tensor) -> list[tuple[Tensor, int]]:
    """(sample indices, valid length T) per distinct T of a left-padded attention mask [B, T_max]; raises unless
    each sample's valid tokens are exactly its last T (the fork's tokenizer pads on the left,
    gr00t/model/transforms.py:47-52)."""
    lengths = mask.sum(dim=1)
    groups = []
    for length in lengths.unique().tolist():
        idx = (lengths == length).nonzero().squeeze(1)
        if length == 0 or not mask[idx, -length:].all():
            raise ValueError(f"backbone_attention_mask of samples {idx.tolist()} is not left padded "
                             f"with {length} valid tokens")
        groups.append((idx, length))
    return groups


class GR00TVelocity(nn.Module):
    """One GR00T action head (no backbone) behind `distillation.interfaces.VelocityModel`. Its only parameters are
    the head's, under `action_head.*`, so student and fake parameter names match for `ida_update`. Set
    `memoize = False` on an instance to recompute `process_backbone_output` on every call."""

    memoize: bool = True

    def __init__(self, action_head: FlowmatchingActionHead) -> None:
        """Wrap `action_head` and put it in eval mode for good."""
        super().__init__()
        self.action_head = action_head
        self._memo: tuple[Tensor, Tensor, tuple, list[tuple[Tensor, Tensor]]] | None = None
        self.eval()

    @classmethod
    def from_checkpoint(
        cls, path: Path, dtype: torch.dtype, device: str | torch.device, trainable: bool
    ) -> "GR00TVelocity":
        """Build the head from the checkpoint's `action_head_cfg` (as N15:82-83, N15 = gr00t/model/gr00t_n1.py)
        with empty weights, then load only its `action_head.*` tensors cast to `dtype` on `device` (strict);
        parameters require grad iff `trainable` (teachers: bf16, frozen; student and fake: fp32, trainable)."""
        config = json.loads((Path(path) / "config.json").read_text())["action_head_cfg"]
        with init_empty_weights():
            head = FlowmatchingActionHead(FlowmatchingActionHeadConfig(**config))
        state = {name: t.to(dtype) for name, t in checkpoint_tensors(path, "action_head.", device)}
        head.load_state_dict(state, strict=True, assign=True)
        return cls(head.requires_grad_(trainable))

    def train(self, mode: bool = True) -> "GR00TVelocity":
        """Stay in eval mode whatever `mode` is: GR00T samples with dropout off (POL:241, recon §10.4)."""
        return super().train(False)

    def __getstate__(self) -> dict:
        """Deep copies and pickles start with an empty memo (it may hold graph-carrying tensors)."""
        return {**self.__dict__, "_memo": None}

    def velocity(self, actions: Tensor, t: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
        """Native velocity [B, H, D] at `actions` and continuous time t [B] in [0, 1], in the dtype of `actions`.
        `cond` holds `backbone_features` [B, T, 2048] (left padded), `backbone_attention_mask` [B, T],
        `state` [B, 1, 64] and `embodiment_id` [B]; any other key (e.g. `action_mask`) is ignored."""
        self._check_inputs(actions, t)
        buckets = self.action_head.num_timestep_buckets
        t_disc = (t.float() * buckets).long().clamp(max=buckets - 1)  # FMH:379 in fp32; t = 1 → last bucket
        v = torch.empty_like(actions)
        with torch.autocast(device_type=actions.device.type, dtype=torch.bfloat16):  # POL:190
            for idx, vl_embs in self._vl_groups(cond):
                v_group = self._step_velocity(actions[idx], t_disc[idx], vl_embs, cond["state"][idx],
                                              cond["embodiment_id"][idx])
                v[idx] = v_group.to(v.dtype)
        return v

    def _check_inputs(self, actions: Tensor, t: Tensor) -> None:
        """Raise on wrong action or time shapes, or if any submodule was switched to training mode (dropout on)."""
        shape = (self.action_head.action_horizon, self.action_head.action_dim)
        if actions.dim() != 3 or tuple(actions.shape[1:]) != shape:
            raise ValueError(f"actions must have shape [B, {shape[0]}, {shape[1]}], got {tuple(actions.shape)}")
        if t.shape != actions.shape[:1]:
            raise ValueError(f"t must have shape [{actions.shape[0]}], got {tuple(t.shape)}")
        if any(module.training for module in self.modules()):
            raise RuntimeError("a GR00TVelocity submodule is in training mode (dropout on); it must stay in eval mode")

    def _vl_groups(self, cond: Mapping[str, Tensor]) -> list[tuple[Tensor, Tensor]]:
        """(sample indices, vl_embs) per valid-length group: `process_backbone_output` (FMH:266-271, called at 355)
        on a fresh dict holding the group's unpadded features (it writes into the dict it gets, FMH:270). The result
        is reused only while the cond's feature and mask tensors are the same objects (`is`) at the same version,
        no optimizer has stepped, the grad mode is the same, every vlln / vl_self_attention parameter keeps its
        storage, version and requires_grad (so IDA or a state-dict load recompute), and no backward
        pass has gone through it (that frees its graph). Until then calls share one graph, as within a
        `distill_step` (one backward)."""
        features, mask = cond["backbone_features"], cond["backbone_attention_mask"]
        key = self._memo_key(features, mask)
        memo = self._memo
        if self.memoize and memo is not None and memo[0] is features and memo[1] is mask and memo[2] == key:
            return memo[3]
        head = self.action_head
        groups = [(idx, head.process_backbone_output({"backbone_features": features[idx, -length:]})
                   ["backbone_features"]) for idx, length in _valid_length_groups(mask)]
        self._memo = (features, mask, key, groups) if self.memoize else None
        for _, vl_embs in groups:
            if self.memoize and vl_embs.requires_grad:
                vl_embs.register_hook(self._forget_memo)
        return groups

    def _forget_memo(self, grad: Tensor) -> None:
        """Gradient hook on memoized results: backward reached them and frees their graph, so drop the memo."""
        self._memo = None

    def _memo_key(self, features: Tensor, mask: Tensor) -> tuple:
        """Everything besides tensor identity that a memoized `process_backbone_output` result depends on."""
        head = self.action_head
        params = [*head.vlln.parameters(), *head.vl_self_attention.parameters()]
        return (features._version, mask._version, _optimizer_steps, torch.is_grad_enabled(),
                tuple((p.data_ptr(), p._version, p.requires_grad) for p in params))

    def _step_velocity(
        self, a: Tensor, t_disc: Tensor, vl_embs: Tensor, state: Tensor, embodiment_id: Tensor
    ) -> Tensor:
        """get_action's loop body for one unpadded group: state encoder (FMH:362), action encoder (385), position
        embedding (387-390), state ⊕ future tokens ⊕ action tokens (393-394), DiT (397-401), decoder and the last
        H tokens (402-404)."""
        head = self.action_head
        state_features = head.state_encoder(state, embodiment_id)
        action_features = head.action_encoder(a, t_disc, embodiment_id)
        if head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=a.device)
            action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
        future_tokens = head.future_tokens.weight.unsqueeze(0).expand(a.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        model_output = head.model(hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=t_disc)
        return head.action_decoder(model_output, embodiment_id)[:, -head.action_horizon:]

