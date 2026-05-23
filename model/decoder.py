# pointer attention 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerDecoder(nn.Module):
    """Pointer Network Decoder — 매 step 호출 (decode)

    현재 state를 query로, encoded flights를 key로 → dot-product attention → 확률
    """

    def __init__(self, d_model: int = 128, airport_emb_dim: int = 32, skip_state_mlp: bool = False):
        """
        d_model: attention 공간 차원 (encoder d_model과 일치해야 함)
        airport_emb_dim: state_vec 내 공항 embedding 차원 (encoder airport_emb_dim과 일치해야 함)
        skip_state_mlp: ablation용 — True면 MLP(Linear→ReLU→Linear) 대신 linear projection만 사용
        """
        super().__init__()

        # state_vec(38,) = airport_emb(airport_emb_dim) + 6개 스칼라
        # 6개: time_of_day, day_norm, duty_time/max, legs/max, duty_period/max, is_resting
        state_input_dim = airport_emb_dim + 6
        self.state_mlp = nn.Sequential(
            nn.Linear(state_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.skip_state_mlp = skip_state_mlp
        if skip_state_mlp:
            self.skip_proj = nn.Linear(state_input_dim, d_model, bias=False)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)

        # END_DUTY, END_PAIRING 각각 학습 가능한 벡터
        self.end_duty_token    = nn.Parameter(torch.randn(d_model))
        self.end_pairing_token = nn.Parameter(torch.randn(d_model))

        self.d_model = d_model

    def forward(
        self,
        encoded_flights: torch.Tensor,
        state_vec: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoded_flights: (N, d_model)
            state_vec: (airport_emb_dim+6,)
            mask: (N+2,) — [...flights, END_DUTY, END_PAIRING]
        Returns:
            probs: (N+2,)
        """
        q = self.state_mlp(state_vec)
        if self.skip_state_mlp:
            q = q + self.skip_proj(state_vec)
        q = self.W_q(q)

        keys = torch.cat([
            encoded_flights,
            self.end_duty_token.unsqueeze(0),
            self.end_pairing_token.unsqueeze(0),
        ], dim=0)                                    # (N+2, d_model)
        k = self.W_k(keys)

        scores = (k @ q) / math.sqrt(self.d_model)  # (N+2,)
        scores[mask == 0] = float('-inf')
        probs = F.softmax(scores, dim=-1)            # (N+2,)

        return probs
