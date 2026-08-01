import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM), Perez et al. 2018.

    Paper Sec. "Constraint-Conditioned Pairing Generator", Eq. (5):
    H^enc = Transformer(FiLM_pre(H^(0); c~)), H = FiLM_post(H^enc; c~).
    This module implements one FiLM_pre / FiLM_post instance: it maps the
    normalized rule-profile vector c~ to (gamma, beta) and modulates the
    flight embeddings so that the same flight is represented differently
    under different airline rule profiles.
    """

    def __init__(self, constraint_dim: int, hidden_dim: int, use_skip: bool = False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(constraint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        nn.init.zeros_(self.mlp[2].weight)
        if use_skip:
            # use_skip=True: out = gamma*x + beta + x
            # Identity init: gamma=0, beta=0 -> out = x
            nn.init.zeros_(self.mlp[2].bias)
        else:
            # use_skip=False: out = gamma*x + beta
            # Identity init: gamma=1, beta=0 -> out = x
            nn.init.ones_(self.mlp[2].bias[:hidden_dim])   # gamma = 1
            nn.init.zeros_(self.mlp[2].bias[hidden_dim:])  # beta = 0

        self.use_skip = use_skip

    def forward(self, flight_vecs: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flight_vecs: (N, hidden_dim)
            constraint:  (constraint_dim,) -- normalized rule-profile vector c~, values in [0, 1]
        Returns:
            (N, hidden_dim)
        """
        params = self.mlp(constraint)
        gamma, beta = params.chunk(2, dim=-1)
        out = gamma.unsqueeze(0) * flight_vecs + beta.unsqueeze(0)
        if self.use_skip:
            out = out + flight_vecs
        return out
