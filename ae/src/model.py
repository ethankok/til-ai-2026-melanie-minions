"""Policy network for AE.

Small CNN over each viewcone, MLP over scalars, concat, action head.
Built to stay under ~200k params so CPU inference latency in the AE
container stays under ~5 ms per call.
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
    """CNN-over-viewcones policy. Outputs raw logits."""

    def __init__(self, action_dim: int = ACTION_DIM):
        super().__init__()
        self.agent_conv = nn.Sequential(
            nn.Conv2d(VIEW_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.base_conv = nn.Sequential(
            nn.Conv2d(VIEW_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        agent_flat = 16 * AGENT_VIEW_HW[0] * AGENT_VIEW_HW[1]   # 560
        base_flat = 8 * BASE_VIEW_HW[0] * BASE_VIEW_HW[1]       # 392
        self.head = nn.Sequential(
            nn.Linear(agent_flat + base_flat + SCALAR_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
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
                # log(0)→-inf masks illegal actions cleanly under argmax/softmax.
                logits = logits + torch.log(action_mask.clamp(min=1e-9))
            if greedy:
                return int(logits.argmax().item())
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
