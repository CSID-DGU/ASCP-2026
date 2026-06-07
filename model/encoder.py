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
        constraint_dim: int = 7,
        airport_emb_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        skip_film: bool = False,
        skip_transformer: bool = False,
        use_film_before: bool = True,
        use_film_after: bool = True,
    ):
        super().__init__()

        # Embedding: 공항(learnable) + 시간(dep, arr, fly) 3개
        self.airport_emb = nn.Embedding(n_airports, airport_emb_dim)
        input_dim = airport_emb_dim * 2 + 3  # origin_emb + dest_emb + dep + arr + fly
        self.flight_mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # FiLM: constraint로 flight 벡터 변조 (Transformer 전+후 양쪽)
        # use_film_before/after=False이면 해당 FiLM을 건너뜀 (identity)
        self.film_before = FiLM(constraint_dim, d_model, use_skip=skip_film)
        self.film_after  = FiLM(constraint_dim, d_model, use_skip=skip_film)
        self.use_film_before = use_film_before
        self.use_film_after  = use_film_after

        # Transformer: flight 간 관계 파악
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.skip_transformer = skip_transformer

    def forward(
        self,
        origins: torch.Tensor,
        dests: torch.Tensor,
        dep_times: torch.Tensor,
        arr_times: torch.Tensor,
        fly_times: torch.Tensor,
        constraint: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            origins:    (N,) int
            dests:      (N,) int
            dep_times:  (N,) float — 정규화된 출발 시간 [0, 1]
            arr_times:  (N,) float — 정규화된 도착 시간 [0, 1]
            fly_times:  (N,) float — 정규화된 비행 시간 (arr - dep) [0, ~0.2]
            constraint: (constraint_dim,) float — 정규화된 constraint 벡터
        Returns:
            (N, d_model)
        """
        o_emb = self.airport_emb(origins)
        d_emb = self.airport_emb(dests)
        times = torch.stack([dep_times, arr_times, fly_times], dim=-1)  # (N, 3)
        x = torch.cat([o_emb, d_emb, times], dim=-1)
        x = self.flight_mlp(x)

        if self.use_film_before:
            x = self.film_before(x, constraint)

        x_pre = x
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        if self.skip_transformer:
            x = x + x_pre

        if self.use_film_after:
            x = self.film_after(x, constraint)

        return x
