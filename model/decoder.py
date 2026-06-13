# pointer attention 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerDecoder(nn.Module):
    """Pointer Network Decoder — 매 step 호출 (decode)

    현재 state를 query로, encoded flights를 key로 → dot-product attention → 확률
    """

    def __init__(self, d_model: int = 128, airport_emb_dim: int = 32, constraint_dim: int = 7,
                 skip_state_mlp: bool = False):
        """
        d_model: attention 공간 차원 (encoder d_model과 일치해야 함)
        airport_emb_dim: state_vec 내 공항 embedding 차원 (encoder airport_emb_dim과 일치해야 함)
        constraint_dim: constraint_vec 차원 — state_vec에 직접 concatenate하여 decoder가 constraint를 직접 볼 수 있게 함
        skip_state_mlp: ablation용 — True면 MLP(Linear→ReLU→Linear) 대신 linear projection만 사용
        """
        super().__init__()

        # state_vec(78,) = current_airport_emb(32) + base_airport_emb(32) + scalars(7) + constraint_vec(7)
        # scalars 7개: time_of_day, day_norm, duty_elapsed/NORM, legs/NORM, duty_period/NORM, is_resting, rest_remaining
        # constraint_vec 추가 이유: FiLM이 flight embedding을 변조해도 decoder가 constraint를 직접 못 보면
        #   policy가 constraint를 반영한 action을 선택하기 어려움. concatenate로 직접 전달.
        state_input_dim = airport_emb_dim * 2 + 7 + constraint_dim
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
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            encoded_flights: (N, d_model)
            state_vec: (state_dim,) 또는 (B, state_dim) — 배치 rollout 지원
            mask: (N+2,) 또는 (B, N+2)
            return_logits: True면 softmax 전 raw score 반환 — REINFORCE log_prob 계산 시 수치 안정성
        Returns:
            probs or logits: (N+2,) 또는 (B, N+2)
        """
        q = self.state_mlp(state_vec)
        if self.skip_state_mlp:
            q = q + self.skip_proj(state_vec)
        q = self.W_q(q)                              # (d_model,) or (B, d_model)

        keys = torch.cat([
            encoded_flights,
            self.end_duty_token.unsqueeze(0),
            self.end_pairing_token.unsqueeze(0),
        ], dim=0)                                    # (N+2, d_model)
        k = self.W_k(keys)

        if q.dim() == 2:
            # 배치: q (B, d_model), k (N+2, d_model) → scores (B, N+2)
            scores = torch.einsum('bd,nd->bn', q, k) / math.sqrt(self.d_model)
        else:
            scores = (k @ q) / math.sqrt(self.d_model)  # (N+2,)

        scores = scores.masked_fill(mask == 0, float('-inf'))

        if return_logits:
            return scores
        return F.softmax(scores, dim=-1)
