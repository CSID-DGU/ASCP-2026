import torch
import torch.nn as nn

from .film import FiLM


class FlightEncoder(nn.Module):
    """Embedding + FiLM + Transformer -- called once per episode (encode).

    Paper Sec. "Constraint-Conditioned Pairing Generator", Eq. (5):
    H^enc = Transformer(FiLM_pre(H^(0); c~)), H = FiLM_post(H^enc; c~).
    flight data -> embedding -> FiLM_pre(rule profile) -> Transformer -> FiLM_post.
    FiLM_pre conditions the encoder input so self-attention can capture
    rule-dependent relationships among flight legs; FiLM_post then
    readjusts the contextualized representations before the pointer decoder.
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

        # Embedding: airport (learnable) + 3 temporal features (dep, arr, fly)
        self.airport_emb = nn.Embedding(n_airports, airport_emb_dim)
        input_dim = airport_emb_dim * 2 + 3  # origin_emb + dest_emb + dep + arr + fly
        self.flight_mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # FiLM_pre / FiLM_post (Eq. 5): modulate flight vectors with the rule
        # profile before and after the Transformer, respectively.
        # use_film_before/after=False skips the corresponding FiLM (identity, ablation use).
        self.film_before = FiLM(constraint_dim, d_model, use_skip=skip_film)
        self.film_after  = FiLM(constraint_dim, d_model, use_skip=skip_film)
        self.use_film_before = use_film_before
        self.use_film_after  = use_film_after

        # Transformer encoder: captures spatiotemporal dependencies among flight legs (Eq. 5)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.skip_transformer = skip_transformer

    def film_params(self):
        """Return only the FiLM layer parameters -- used to give FiLM a separate optimizer lr."""
        return list(self.film_before.parameters()) + list(self.film_after.parameters())

    def non_film_params(self):
        """Return all parameters except FiLM -- used to give FiLM a separate optimizer lr."""
        return (list(self.airport_emb.parameters()) +
                list(self.flight_mlp.parameters()) +
                list(self.transformer.parameters()))

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
            dep_times:  (N,) float -- normalized departure time [0, 1]
            arr_times:  (N,) float -- normalized arrival time [0, 1]
            fly_times:  (N,) float -- normalized flight time (arr - dep) [0, ~0.2]
            constraint: (constraint_dim,) float -- normalized rule-profile vector c~ (Eq. 5)
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
