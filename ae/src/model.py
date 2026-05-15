"""Policy network for AE.

CNN over each viewcone, MLP over scalars, concat, action head. Built for
fast CPU inference (<10 ms/call) and configurable for N-frame stacking.

When `n_frames > 1` the input channel dim scales linearly (25*N) and the
scalar dim scales the same way. The first conv layer absorbs the bigger
channel count without changing downstream tensor shapes.

When `use_belief=True`, an additional CNN branch ingests a (K, 16, 16)
belief tensor (the heuristic's persistent world-state memory) and its
features are concatenated into the head. This is the key architectural
lever for closing the local→cloud gap: the egocentric viewcones alone
have *no* memory of where the agent has been, while the belief map
gives the policy strictly more world state than any number of stacked
viewcones could.
"""

from __future__ import annotations

import torch
from torch import nn

from encoder import BELIEF_CHANNELS, SCALAR_DIM


VIEW_CHANNELS = 25
AGENT_VIEW_HW = (7, 5)
BASE_VIEW_HW = (7, 7)
BELIEF_HW = (16, 16)
ACTION_DIM = 6


class PolicyNetwork(nn.Module):
    """Frame-stacked CNN policy. Outputs raw logits.

    Parameters
    ----------
    n_frames
        Number of viewcone frames stacked along the channel dim.
    action_dim
        Output action count (default 6 for AE).
    use_belief
        If True, expects an additional (B, BELIEF_CHANNELS, 16, 16)
        belief-map tensor in ``forward``. Adds a small conv branch
        (~50 K params) and a chunk to the head's input dim.
    """

    def __init__(self, n_frames: int = 4, action_dim: int = ACTION_DIM,
                 use_belief: bool = False):
        super().__init__()
        if n_frames < 1:
            raise ValueError("n_frames must be ≥ 1")
        self.n_frames = n_frames
        self.use_belief = bool(use_belief)

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
        head_in = agent_flat + base_flat + scalar_dim

        if self.use_belief:
            # Belief branch: stride-aware conv + pool to keep the head's
            # input dimension manageable on CPU. Output is 8 ch × 8 × 8 = 512
            # features — small enough to roughly double rather than 10x the
            # head input, but large enough to encode the full 16×16 grid.
            self.belief_conv = nn.Sequential(
                nn.Conv2d(BELIEF_CHANNELS, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AvgPool2d(2),                                # 16→8
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            belief_flat = 8 * (BELIEF_HW[0] // 2) * (BELIEF_HW[1] // 2)  # 512
            head_in += belief_flat

        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(
        self,
        agent_view: torch.Tensor,
        base_view: torch.Tensor,
        scalars: torch.Tensor,
        belief_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = self.agent_conv(agent_view).flatten(start_dim=1)
        b = self.base_conv(base_view).flatten(start_dim=1)
        parts = [a, b, scalars]
        if self.use_belief:
            if belief_map is None:
                raise ValueError(
                    "PolicyNetwork was constructed with use_belief=True but "
                    "forward() was called without belief_map."
                )
            c = self.belief_conv(belief_map).flatten(start_dim=1)
            parts.append(c)
        return self.head(torch.cat(parts, dim=-1))

    def select_action(
        self,
        agent_view: torch.Tensor,
        base_view: torch.Tensor,
        scalars: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        greedy: bool = True,
        belief_map: torch.Tensor | None = None,
    ) -> int:
        """Single-observation inference; pass un-batched tensors."""

        with torch.no_grad():
            belief_b = belief_map.unsqueeze(0) if belief_map is not None else None
            logits = self.forward(
                agent_view.unsqueeze(0),
                base_view.unsqueeze(0),
                scalars.unsqueeze(0),
                belief_map=belief_b,
            ).squeeze(0)
            if action_mask is not None:
                logits = logits + torch.log(action_mask.clamp(min=1e-9))
            if greedy:
                return int(logits.argmax().item())
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
