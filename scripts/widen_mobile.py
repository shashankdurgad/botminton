"""Widen a target-stage rsl_rl checkpoint for the mobile stage's base actions and observations.

The mobile stage appends 3 base actions and 10 observations (the 3 base actions, base x/y, cos/sin yaw,
base velocity). New input columns and new action rows get zero weights, so the widened policy plays
exactly like the stationary one and keeps the base still until training finds a use for it. The
normalizer is seeded with rough scales for the new inputs and keeps its sample count: lowering it lets the
first training batches (which start mid-episode, often with the racket not yet gripped) rewrite the input
statistics within a few iterations, and the policy collapses.

Usage:
    python scripts/widen_mobile.py <target model.pt> <out.pt>
"""

import argparse

import torch

from badminton_rl.tasks.badminton_env_cfg import BadmintonMobileEnvCfg, BadmintonTargetEnvCfg

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
parser.add_argument("--base_std", type=float, default=0.3, help="initial exploration std of the base actions")
parser.add_argument("--normalizer_count", type=float, default=None,
                    help="optionally cap the normalizer sample count (default: keep it; see module docstring)")
parser.add_argument("--max_std", type=float, default=0.8,
                    help="clip the existing action stds into the mobile runner's std_range so they stay learnable")
args = parser.parse_args()

# rough means / variances: base actions, base x, base y, cos yaw, sin yaw, 0.3 * (vx, vy, wz); later mobile
# observations are appended with scripts/widen_obs.py
mean = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
var = torch.tensor([0.3, 0.3, 0.3, 2.0, 2.0, 0.05, 0.05, 0.3, 0.3, 0.2])
old_obs = BadmintonTargetEnvCfg().observation_space
new_obs = old_obs + len(mean)
old_act, new_act = BadmintonTargetEnvCfg().action_space, BadmintonMobileEnvCfg().action_space

ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
for net in ("actor_state_dict", "critic_state_dict"):
    sd = ckpt[net]
    w = sd["mlp.0.weight"]
    assert w.shape[1] == old_obs, w.shape
    sd["mlp.0.weight"] = torch.cat((w, torch.zeros(w.shape[0], new_obs - old_obs, dtype=w.dtype)), dim=1)
    sd["obs_normalizer._mean"] = torch.cat((sd["obs_normalizer._mean"], mean.view(1, -1)), dim=1)
    sd["obs_normalizer._var"] = torch.cat((sd["obs_normalizer._var"], var.view(1, -1)), dim=1)
    sd["obs_normalizer._std"] = torch.cat((sd["obs_normalizer._std"], var.sqrt().view(1, -1)), dim=1)
    if args.normalizer_count is not None:
        sd["obs_normalizer.count"] = torch.tensor(min(float(sd["obs_normalizer.count"]), args.normalizer_count))

actor = ckpt["actor_state_dict"]
extra = new_act - old_act
w, b = actor["mlp.6.weight"], actor["mlp.6.bias"]
actor["mlp.6.weight"] = torch.cat((w, torch.zeros(extra, w.shape[1], dtype=w.dtype)), dim=0)
actor["mlp.6.bias"] = torch.cat((b, torch.zeros(extra, dtype=b.dtype)))
std = actor["distribution.std_param"].clamp(max=args.max_std)
actor["distribution.std_param"] = torch.cat((std, torch.full((extra,), args.base_std, dtype=std.dtype)))

# Keep Adam's moments. Without them every weight takes a full learning-rate step in its gradient's sign on
# the first noisy batches, which knocks the zero-initialized base outputs to full speed within an iteration.
# Widened entries get a zero first moment and their layer's mean second moment, so they step like the rest.
# Optimizer state index -> (state_dict, key, dim the tensor was widened along, entries added) in parameter order:
# actor std, actor mlp.0-6 (weight, bias), critic mlp.0-6.
opt = ckpt.get("optimizer_state_dict")
if opt is not None:
    widened = {0: (0, extra), 1: (1, new_obs - old_obs), 7: (0, extra), 8: (0, extra), 9: (1, new_obs - old_obs)}
    for idx, (dim, n_new) in widened.items():
        st = opt["state"][idx]
        pad_shape = list(st["exp_avg"].shape)
        pad_shape[dim] = n_new
        st["exp_avg"] = torch.cat((st["exp_avg"], torch.zeros(pad_shape)), dim=dim)
        st["exp_avg_sq"] = torch.cat((st["exp_avg_sq"], torch.full(pad_shape, st["exp_avg_sq"].mean().item())), dim=dim)
    expected = [actor["distribution.std_param"]] + [actor[f"mlp.{i}.{p}"] for i in (0, 2, 4, 6) for p in ("weight", "bias")]
    expected += [ckpt["critic_state_dict"][f"mlp.{i}.{p}"] for i in (0, 2, 4, 6) for p in ("weight", "bias")]
    for idx, param in enumerate(expected):
        assert opt["state"][idx]["exp_avg"].shape == param.shape, (idx, opt["state"][idx]["exp_avg"].shape, param.shape)
torch.save(ckpt, args.dst)
print(f"widened {args.src} -> {args.dst}: observations {old_obs} -> {new_obs}, actions {old_act} -> {new_act}")
