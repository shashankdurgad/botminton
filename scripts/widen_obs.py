"""Append observation columns to an rsl_rl checkpoint, keeping the policy's behaviour and Adam state.

New inputs get zero first-layer weights in the actor and critic, so the widened networks act exactly like
the old ones until training finds a use for them. The normalizer is seeded with the given variance for the new
inputs and keeps its sample count (see ``widen_mobile.py`` for why), and Adam's moments for the two first-layer
weights are padded like the rest of their layer.

Usage:
    python scripts/widen_obs.py <model.pt> <out.pt> --new 2 --var 2.0
"""

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
parser.add_argument("--new", type=int, required=True, help="observation columns appended")
parser.add_argument("--var", type=float, default=1.0, help="normalizer variance prior of the new inputs")
args = parser.parse_args()

ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
n = args.new
for net in ("actor_state_dict", "critic_state_dict"):
    sd = ckpt[net]
    w = sd["mlp.0.weight"]
    sd["mlp.0.weight"] = torch.cat((w, torch.zeros(w.shape[0], n, dtype=w.dtype)), dim=1)
    sd["obs_normalizer._mean"] = torch.cat((sd["obs_normalizer._mean"], torch.zeros(1, n)), dim=1)
    sd["obs_normalizer._var"] = torch.cat((sd["obs_normalizer._var"], torch.full((1, n), args.var)), dim=1)
    sd["obs_normalizer._std"] = torch.cat((sd["obs_normalizer._std"], torch.full((1, n), args.var**0.5)), dim=1)
old_obs = w.shape[1]

# optimizer state order: actor std, actor mlp.0-6 (weight, bias), critic mlp.0-6; the first-layer weights are 1, 9
opt = ckpt.get("optimizer_state_dict")
if opt is not None:
    for idx, net in ((1, "actor_state_dict"), (9, "critic_state_dict")):
        st = opt["state"][idx]
        pad = (st["exp_avg"].shape[0], n)
        st["exp_avg"] = torch.cat((st["exp_avg"], torch.zeros(pad)), dim=1)
        st["exp_avg_sq"] = torch.cat((st["exp_avg_sq"], torch.full(pad, st["exp_avg_sq"].mean().item())), dim=1)
        assert st["exp_avg"].shape == ckpt[net]["mlp.0.weight"].shape, idx
torch.save(ckpt, args.dst)
print(f"widened {args.src} -> {args.dst}: observations {old_obs} -> {old_obs + n}")
