# FiLM 기법 (Feature-wise Linear Modulation)
# constraint가 달라지면 같은 flight도 다르게 해석됨
# constraint 벡터 → gamma, beta 생성 → flight 벡터를 변조

import torch
import torch.nn as nn


class FiLM(nn.Module):

    def __init__(self, constraint_dim: int, hidden_dim: int, use_skip: bool = False):
        super().__init__()
        self.mlp = nn.Sequential( # 변조 파라미터 생성
            nn.Linear(constraint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # gamma + beta 두 세트 한번에 생성하기
        )
        self.use_skip = use_skip  # True면 원본 flight_vecs를 출력에 더함 (residual)

    def forward(self, flight_vecs: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flight_vecs: (N, hidden_dim) — flight embedding 벡터들
            constraint: (constraint_dim,) — 항공사 규정 벡터
        Returns:
            (N, hidden_dim) — constraint로 변조된 flight 벡터들
        """
        params = self.mlp(constraint)                    # (hidden_dim * 2,)
        gamma, beta = params.chunk(2, dim=-1)            # (hidden_dim,), (hidden_dim,) 로 반등분 -> 앞부분은 곱하기용(gamma), 뒷부분은 더하기용(beta)으로 할당
        out = gamma.unsqueeze(0) * flight_vecs + beta.unsqueeze(0)  # (N, hidden_dim)
        if self.use_skip:
            out = out + flight_vecs                      # skip: FiLM이 원본 embedding을 완전히 덮어쓰지 않도록
        return out
