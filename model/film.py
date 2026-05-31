import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation

    constraint 벡터 → gamma, beta 생성 → flight 벡터를 변조.
    constraint가 달라지면 같은 flight도 다르게 해석된다.
    """

    def __init__(self, constraint_dim: int, hidden_dim: int, use_skip: bool = False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(constraint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        # gamma 초기값 = 1 (identity), beta 초기값 = 0 → 학습 초기 출력 보존
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.ones_(self.mlp[2].bias[:hidden_dim])   # gamma = 1
        nn.init.zeros_(self.mlp[2].bias[hidden_dim:])  # beta = 0

        self.use_skip = use_skip

    def forward(self, flight_vecs: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flight_vecs: (N, hidden_dim)
            constraint:  (constraint_dim,) — 정규화된 값 [0, 1]
        Returns:
            (N, hidden_dim)
        """
        params = self.mlp(constraint)
        gamma, beta = params.chunk(2, dim=-1)
        out = gamma.unsqueeze(0) * flight_vecs + beta.unsqueeze(0)
        if self.use_skip:
            out = out + flight_vecs
        return out
