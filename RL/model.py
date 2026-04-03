import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)

        self.film = nn.Linear(1, embed_dim * 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.state_mlp = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def encode(self, flights, constraint):
        x = torch.tensor(flights, dtype=torch.float32)

        x = self.embedding(x)

        c = torch.tensor([[constraint["max_duty"]]], dtype=torch.float32)
        gamma_beta = self.film(c)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        x = gamma * x + beta

        x = self.encoder(x.unsqueeze(0))

        return x.squeeze(0)  # (N, embed_dim)

    def decode(self, encoded, state, mask):
        state_vec = torch.tensor([
            state["current_airport"],
            state["current_time"],
            state["duty_time"],
            state["remaining"]
        ], dtype=torch.float32)

        query = self.state_mlp(state_vec)

        # (N,)
        scores = torch.matmul(encoded, query)

        # ---------------------------
        # ✔ END action score 추가
        # ---------------------------
        end_score = torch.tensor([0.0])  # 간단히 0으로 시작
        scores = torch.cat([scores, end_score], dim=0)

        # ---------------------------
        # ✔ mask 적용
        # ---------------------------
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        scores[~mask_tensor] = -1e9

        probs = F.softmax(scores, dim=-1)

        return probs