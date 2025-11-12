# =============================================================================
# vqvae/codebook_manager.py
# -----------------------------------------------------------------------------
# Manages VQ-VAE codebook health by tracking per-code usage statistics,
# maintaining a reservoir of recent encoder latents, and reseeding "dead" codes.
#
# A code i is considered dead when:
#   (usage_ema[i] / usage_ema.sum()) < dead_eps   AND   age[i] > dead_patience
#
# Dead codes are reinitialized with random samples drawn from the latent
# reservoir, ensuring that replacements lie on the current data manifold.
#
# Key components:
#   - usage_ema      – EMA of per-code assignment counts (smoothed frequency)
#   - age            – Steps since last assignment for each code
#   - reservoir      – Rolling buffer of recent latent vectors for reseeding
#   - dead_eps       – Minimum normalized usage fraction to stay alive
#   - dead_patience  – Max tolerated consecutive unused steps
#
# Works with both straight-through and EMA quantizers, and supports DDP.
# The perplexity reported is a scalar entropy-based measure of code usage
# diversity, reduced across ranks when distributed.
# =============================================================================

import torch
import torch.distributed as dist
from torch import nn


class CodebookManager(nn.Module):
    """
    Tracks codebook usage, maintains a rolling reservoir of encoder latents,
    and reinitializes 'dead' codebook entries using samples from the reservoir.
    Works with both EMA and straight-through quantizers.

    Usage (inside training step):
        manager.push_reservoir(z_e)
        manager.observe_assignments(indices)
        manager.maybe_reseed(quantizer.embeddings)
        pplx = manager.stats()['perplexity']  # scalar
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        reservoir_size: int = 65536,
        ema_decay: float = 0.995,
        dead_eps: float = 1e-4,
        dead_patience: int = 10000,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.device = torch.device(device)

        # Exponential moving average of code usage
        self.register_buffer("usage_ema", torch.zeros(num_codes, device=self.device))
        # Step counter since last usage
        self.register_buffer("age", torch.zeros(num_codes, dtype=torch.long, device=self.device))

        self.ema_decay = ema_decay
        self.dead_eps = dead_eps
        self.dead_patience = dead_patience

        # Simple rolling buffer (reservoir)
        self.reservoir_size = reservoir_size
        self.register_buffer("reservoir", torch.zeros(reservoir_size, code_dim, device=self.device))
        self.register_buffer("reservoir_head", torch.zeros(1, dtype=torch.long, device=self.device))
        self.register_buffer("reservoir_filled", torch.zeros(1, dtype=torch.bool, device=self.device))

    @torch.no_grad()
    def push_reservoir(self, z_e: torch.Tensor):
        """
        Add a random subsample of encoder latents to the reservoir.
        """
        if z_e.numel() == 0:
            return
        B = z_e.shape[0]
        push = z_e.detach()
        push = push.reshape(-1, self.code_dim)
        # subsample ~5%
        if push.shape[0] > self.reservoir_size // 10:
            idx = torch.randperm(push.shape[0], device=self.device)[: self.reservoir_size // 10]
            push = push[idx]

        n = push.shape[0]
        start = self.reservoir_head.item()
        end = start + n

        if end < self.reservoir_size:
            self.reservoir[start:end] = push
        else:
            first = self.reservoir_size - start
            self.reservoir[start:] = push[:first]
            self.reservoir[: end % self.reservoir_size] = push[first:]

        new_head = end % self.reservoir_size
        self.reservoir_head.fill_(new_head)
        if end >= self.reservoir_size:
            self.reservoir_filled.fill_(True)

    @torch.no_grad()
    def observe_assignments(self, indices: torch.LongTensor):
        """
        Update usage stats based on quantizer indices from current batch.
        """
        if indices.numel() == 0:
            return
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_codes).float()

        # Distributed reduction
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        # EMA update
        self.usage_ema.mul_(self.ema_decay).add_(counts * (1 - self.ema_decay))

        # Update age counters
        hit_mask = counts > 0
        self.age[hit_mask] = 0
        self.age[~hit_mask] += 1

    @torch.no_grad()
    def maybe_reseed(self, embeddings: torch.Tensor):
        """
        Replace embeddings of dead codes using samples from the reservoir.
        """
        if not self.reservoir_filled and self.reservoir_head.item() < self.code_dim * 4:
            # Not enough reservoir data yet
            return

        usage_sum = self.usage_ema.sum()
        if usage_sum == 0:
            return
        probs = self.usage_ema / usage_sum
        dead_mask = (probs < self.dead_eps) & (self.age > self.dead_patience)
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return

        # Sample replacements
        n_avail = self.reservoir_size if self.reservoir_filled else self.reservoir_head.item()
        if n_avail == 0:
            return
        idx = torch.randint(0, n_avail, (n_dead,), device=self.device)
        new_vecs = self.reservoir[idx]

        embeddings[dead_mask] = new_vecs
        # reset stats
        self.usage_ema[dead_mask] = self.usage_ema.mean()
        self.age[dead_mask] = 0

        # sync reseeds across ranks
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(embeddings, src=0)

    @torch.no_grad()
    def stats(self) -> dict:
        """
        Return scalar perplexity and raw stats.
        """
        usage_sum = self.usage_ema.sum()
        if usage_sum == 0:
            pplx = torch.tensor(1.0, device=self.device)
        else:
            p = self.usage_ema / usage_sum
            entropy = -(p * (p.clamp_min(1e-12)).log()).sum()
            pplx = torch.exp(entropy)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(pplx, op=dist.ReduceOp.SUM)
            pplx = pplx / dist.get_world_size()

        return {
            "perplexity": float(pplx.item()),  # scalar for logging
            "usage_ema": self.usage_ema.clone().detach(),
            "age": self.age.clone().detach(),
        }

    def snapshot(self) -> dict:
        """
        Save manager state for checkpointing.
        """
        return {
            "usage_ema": self.usage_ema.cpu(),
            "age": self.age.cpu(),
            "reservoir": self.reservoir.cpu(),
            "reservoir_head": self.reservoir_head.cpu(),
            "reservoir_filled": self.reservoir_filled.cpu(),
        }

    def load_snapshot(self, state: dict):
        """
        Restore manager state from checkpoint.
        """
        self.usage_ema.copy_(state["usage_ema"].to(self.device))
        self.age.copy_(state["age"].to(self.device))
        self.reservoir.copy_(state["reservoir"].to(self.device))
        self.reservoir_head.copy_(state["reservoir_head"].to(self.device))
        self.reservoir_filled.copy_(state["reservoir_filled"].to(self.device))
