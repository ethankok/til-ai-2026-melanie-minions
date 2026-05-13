"""Policy network for AE.

CNN over each viewcone, MLP over scalars, concat, action head. Built for
fast CPU inference (<10 ms/call) and configurable for N-frame stacking.

When `n_frames > 1` the input channel dim scales linearly (25*N) and the
scalar dim scales the same way. The first conv layer absorbs the bigger
channel count without changing downstream tensor shapes.
"""

from __future__ import annotations

import torch
from torch import nn

from encoder import SCALAR_DIM


VIEW_CHANNELS = 25
AGENT_VIEW_HW = (7, 5)
BASE_VIEW_HW = (7, 7)
ACTION_DIM = 6


class PolicyNetwork(nn.Module):
    """Frame-stacked CNN policy. Outputs raw logits."""

    def __init__(self, n_frames: int = 4, action_dim: int = ACTION_DIM):
        super().__init__()
        if n_frames < 1:
            raise ValueError("n_frames must be ≥ 1")
        self.n_frames = n_frames

        in_ch = VIEW_CHANNELS * n_frames
        scalar_dim = SCALAR_DIM * n_frames

        # Bigger conv stacks than the single-frame baseline because each input
        # carries 4x the channels worth of game-state history.
        self.agent_conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.base_conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        agent_flat = 32 * AGENT_VIEW_HW[0] * AGENT_VIEW_HW[1]   # 1120
        base_flat = 16 * BASE_VIEW_HW[0] * BASE_VIEW_HW[1]      # 784
        self.head = nn.Sequential(
            nn.Linear(agent_flat + base_flat + scalar_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, agent_view: torch.Tensor, base_view: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        a = self.agent_conv(agent_view).flatten(start_dim=1)
        b = self.base_conv(base_view).flatten(start_dim=1)
        return self.head(torch.cat([a, b, scalars], dim=-1))

    def select_action(
        self,
        agent_view: torch.Tensor,
        base_view: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        greedy: bool = True,
    ) -> int:
        """Single-observation inference; pass un-batched tensors."""

        with torch.no_grad():
            logits = self.forward(
                agent_view.unsqueeze(0),
                base_view.unsqueeze(0),
                scalars.unsqueeze(0),
            ).squeeze(0)
            if action_mask is not None:
                logits = logits + torch.log(action_mask.clamp(min=1e-9))
            if greedy:
                return int(logits.argmax().item())
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
