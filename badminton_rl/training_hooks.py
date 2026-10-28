"""Optional patches to rsl_rl training, switched on by environment variables.

``BADMINTON_CRITIC_WARMUP=<n>``: keep the actor fixed for the first ``n`` PPO updates while the critic trains.
A warm-started actor meeting a new reward (the mobile stage) otherwise follows advantages from a critic fitted to
the old task's returns, and its skills degrade before the critic catches up. Each update runs as usual, then the
actor's parameters are restored, so the gradient all-reduce across ranks is unchanged.
"""

import os

import torch


def install() -> None:
    warmup = int(os.environ.get("BADMINTON_CRITIC_WARMUP", "0"))
    if warmup <= 0:
        return
    from rsl_rl.algorithms import PPO

    original_update = PPO.update
    if getattr(original_update, "_critic_warmup", False):
        return

    def update(self) -> dict[str, float]:
        self._warmup_updates = getattr(self, "_warmup_updates", 0) + 1
        if self._warmup_updates > warmup:
            return original_update(self)
        saved = [p.detach().clone() for p in self.actor.parameters()]
        losses = original_update(self)
        with torch.no_grad():
            for p, s in zip(self.actor.parameters(), saved):
                p.copy_(s)
        if self._warmup_updates in (1, warmup):
            print(f"[badminton] critic warm-up: actor held fixed, update {self._warmup_updates}/{warmup}", flush=True)
        return losses

    update._critic_warmup = True
    PPO.update = update
