"""Widen a hit-stage rsl_rl checkpoint for the target stage's extra observations.

The target stage appends the landing target (x, y) to the 99-dim hit observation. New input columns
get zero weights (the network initially computes exactly what it did before) and the observation
normalizer is seeded with the uniform target distribution so the new inputs arrive standardized.

Usage:
    python scripts/widen_checkpoint.py <hit model.pt> <out.pt>
"""

import argparse

import torch

from badminton_rl.tasks.badminton_env_cfg import BadmintonTargetEnvCfg

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
args = parser.parse_args()

cfg = BadmintonTargetEnvCfg()
ranges = (cfg.target_x_range, cfg.target_y_range)
mean = torch.tensor([(lo + hi) / 2 for lo, hi in ranges])
var = torch.tensor([(hi - lo) ** 2 / 12 for lo, hi in ranges])
extra = len(ranges)

ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
for net in ("actor_state_dict", "critic_state_dict"):
    sd = ckpt[net]
    w = sd["mlp.0.weight"]
    sd["mlp.0.weight"] = torch.cat((w, torch.zeros(w.shape[0], extra, dtype=w.dtype)), dim=1)
    sd["obs_normalizer._mean"] = torch.cat((sd["obs_normalizer._mean"], mean.view(1, -1)), dim=1)
    sd["obs_normalizer._var"] = torch.cat((sd["obs_normalizer._var"], var.view(1, -1)), dim=1)
    sd["obs_normalizer._std"] = torch.cat((sd["obs_normalizer._std"], var.sqrt().view(1, -1)), dim=1)
    assert sd["mlp.0.weight"].shape[1] == cfg.observation_space
# Adam moments no longer match the first layer's shape; rsl_rl still expects the key, so keep the
# parameter groups and drop the per-parameter state
opt = ckpt.get("optimizer_state_dict")
if opt is not None:
    ckpt["optimizer_state_dict"] = {"state": {}, "param_groups": opt["param_groups"]}
torch.save(ckpt, args.dst)
print(f"widened {args.src} -> {args.dst}: observations 99 -> {cfg.observation_space}")
