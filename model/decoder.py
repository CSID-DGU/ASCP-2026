# pointer attention 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerDecoder(nn.Module):
    """Pointer Network Decoder — 매 step 호출 (decode)

    현재 state를 query로, encoded flights를 key로 → dot-product attention → 확률
    """

    def __init__(self, d_model: int = 128, airport_emb_dim: int = 32):
        super().__init__()

        # state → query 변환용 MLP: 공항 embedding + current_time + duty_time
        state_input_dim = airport_emb_dim + 3  # 공항 ID + 현재 시간 + 근무 시간 + leg 수
        self.state_mlp = nn.Sequential( # 현재 에이전트의 상태 정보를 고차원 벡터인 d_model 크기로 확장 (이 결과값이 나중에 쿼리가 됨)
            nn.Linear(state_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Pointer attention projection
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)

        # END_PAIRING 학습 가능한 벡터
        self.end_token = nn.Parameter(torch.randn(d_model))

        self.d_model = d_model

    def forward(
        self,
        encoded_flights: torch.Tensor,
        state_vec: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoded_flights: (N, d_model) — encoder 출력 (에피소드당 1번 계산된 것)
            state_vec: (state_input_dim,) — [airport_emb, current_time, duty_time]
            mask: (N+1,) — 1=선택 가능, 0=불가. 마지막이 END_PAIRING
        Returns:
            probs: (N+1,) — 각 flight + END_PAIRING에 대한 확률
        """
        # state → query
        q = self.state_mlp(state_vec)                    # (d_model,)
        q = self.W_q(q)                                   # (d_model,)

        # 선택지들을 key로 구성
        keys = torch.cat([
            encoded_flights,                               # (N, d_model) - 실제 항공편들
            self.end_token.unsqueeze(0),                   # (1, d_model) - 종료 토큰
        ], dim=0)                                          # 결과: (N+1, d_model)
        k = self.W_k(keys)                                 # key로 변환 (N+1, d_model)



        # 이제 모델이 N개의 항공편 + 1개의 종료 옵션 중 하나를 고를 준비가 됨


        # dot-product attention
        scores = (k @ q) / math.sqrt(self.d_model)        # (N+1,)

        # hard masking
        # 갈 수 없는 곳은 -inf로 처리 (softmax 시 0이 되도록)
        scores[mask == 0] = float('-inf')

        # softmax → 확률
        probs = F.softmax(scores, dim=-1)                  # (N+1,)

        return probs
