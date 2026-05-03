# FiLM과 Pointer Decoder를 하나로 묶어주는 코드

import torch
import torch.nn as nn

from .film import FiLM


class FlightEncoder(nn.Module):
    """Embedding + FiLM + Transformer — 에피소드당 1번 호출 (encode)

    flight 데이터 -> embedding -> FiLM(constraint 반영) -> Transformer(관계 파악)
    """

    def __init__(
        self,
        n_airports: int,
        constraint_dim: int = 1,
        airport_emb_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        skip_film: bool = False,
        skip_transformer: bool = False,
    ):
        super().__init__()

        # Embedding: 공항(learnable) + 시간(float 2개)
        self.airport_emb = nn.Embedding(n_airports, airport_emb_dim)
        input_dim = airport_emb_dim * 2 + 2  # origin_emb + dest_emb + dep_time + arr_time
        self.flight_mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # FiLM: constraint로 flight 벡터 변조 (Transformer 전+후 양쪽)
        self.film_before = FiLM(constraint_dim, d_model, use_skip=skip_film)  # Transformer 전
        self.film_after  = FiLM(constraint_dim, d_model, use_skip=skip_film)  # Transformer 후

        # Transformer: flight 간 관계 파악
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # skip_transformer: FiLM 출력을 Transformer 출력에 더함
        self.skip_transformer = skip_transformer

    def forward(
        self,
        origins: torch.Tensor,
        dests: torch.Tensor,
        dep_times: torch.Tensor,
        arr_times: torch.Tensor,
        constraint: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            origins: (N,) int — 출발 공항 ID
            dests: (N,) int — 도착 공항 ID
            dep_times: (N,) float — 출발 시간 (정규화)
            arr_times: (N,) float — 도착 시간 (정규화)
            constraint: (constraint_dim,) float — e.g. [max_duty]
        Returns:
            (N, d_model) — encode된 flight 벡터들 (에피소드 내내 재사용함)
        """
        # 1. Embedding
        """
        airport_emb: 공항 id 넣으면 그 공항의 특징(위치나 규모 등)을 담은 벡터를 꺼내줌
        cat: 출발지 벡터 + 도착지 벡터 + 시간 정보 (2개)를 모두 이어 붙여서 하나의 거대한 정보 만듦
        flight_mlp: 거대한 정보를 d_model 크기로 다듬음
        """
        o_emb = self.airport_emb(origins)              # 출발 공항 -> 벡터로
        d_emb = self.airport_emb(dests)                # 도착 공항 -> 벡터로
        times = torch.stack([dep_times, arr_times], dim=-1)  # 출발/도착 시간 합치기
        x = torch.cat([o_emb, d_emb, times], dim=-1)  # 전부 가로로 붙이기
        x = self.flight_mlp(x)                         # 128차원으로 압축/변환

        # 2. FiLM (before) — Transformer 전에 constraint 반영
        x = self.film_before(x, constraint)            # (N, d_model)

        # 3. Transformer — flight 간 관계 파악
        x_before_transformer = x                       # skip_transformer용: Transformer 전 벡터 저장
        x = x.unsqueeze(0)                             # 배치 차원 추가 (1, N, d_model)
        x = self.transformer(x)                        # 비행기들끼리 서로 정보를 주고 받음 (attention) (1, N, d_model)
        x = x.squeeze(0)                               # 배치 차원 제거 (N, d_model)
        if self.skip_transformer:
            x = x + x_before_transformer               # skip: Transformer가 FiLM 정보를 잃지 않도록

        # 4. FiLM (after) — Transformer 후에 constraint 한번 더 반영
        x = self.film_after(x, constraint)             # (N, d_model)

        return x
