# decoder.py -- pointer-network decoder (pointer attention)
#
# Paper Sec. "Constraint-Conditioned Pairing Generator", Eq. (6)-(8):
# builds the pointer logit z_{t,j} = q_t^T k_j / sqrt(d) + w_gap * g_t(j) + b_mask_t(j; c),
# then pi_theta(j | s_t, F, c) = softmax_j(z_{t,j}). The additive feasibility
# mask b_mask_t (Eq. 6) is supplied by the caller via the `mask` argument
# (see RL/environment.py::get_mask, RL/utils.py::flight_gap_bias for g_t(j)).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerDecoder(nn.Module):
    """Pointer Network Decoder -- called once per decoding step (decode).

    Current state as query, encoded flights as keys -> dot-product attention
    -> action distribution (Eq. 7-8).
    """

    def __init__(self, d_model: int = 128, airport_emb_dim: int = 32, constraint_dim: int = 7,
                 n_scalars: int = 8, skip_state_mlp: bool = False):
        """
        d_model: attention space dimension (must match encoder d_model)
        airport_emb_dim: airport embedding dim inside state_vec (must match encoder airport_emb_dim)
        constraint_dim: dim of constraint_vec -- concatenated directly into state_vec so the
            decoder query q_t can see the rule profile c directly, in addition to FiLM conditioning
        skip_state_mlp: ablation flag -- if True, use a linear projection instead of the
            MLP (Linear->ReLU->Linear)
        """
        super().__init__()

        # state_vec(79,) = current_airport_emb(32) + base_airport_emb(32) + scalars(8) + constraint_vec(7)
        # 8 scalars: time_of_day, day_norm, duty_elapsed/NORM, legs/NORM, duty_period/NORM,
        #            is_resting, rest_remaining, total_legs/NORM (cumulative legs over the whole pairing)
        # total_legs is included so the decoder knows how many legs the current pairing has
        #   accumulated when deciding EndPairing timing (legs only reflects the current duty)
        state_input_dim = airport_emb_dim * 2 + n_scalars + constraint_dim
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

        # Learnable scalar w_gap in Eq. (7) -- initialized negative so larger
        # connection gaps lower the score.
        self.gap_weight = nn.Parameter(torch.tensor(-1.0))

        # Learnable key vectors for the two structural actions EndDuty, EndPairing (Eq. 4)
        self.end_duty_token    = nn.Parameter(torch.randn(d_model))
        self.end_pairing_token = nn.Parameter(torch.randn(d_model))

        self.d_model = d_model

    def forward(
        self,
        encoded_flights: torch.Tensor,
        state_vec: torch.Tensor,
        mask: torch.Tensor,
        gap_bias: torch.Tensor = None,
        action_bias: torch.Tensor = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            encoded_flights: (N, d_model)
            state_vec: (state_dim,) or (B, state_dim) -- batched rollout support
            mask: (N+2,) or (B, N+2) -- additive feasibility mask b_mask_t(j; c) (Eq. 6),
                0 for j in A_feas_t(c), -inf otherwise
            gap_bias: (N+2,) or (B, N+2) -- per-candidate normalized connection gap g_t(j)
                in [0,1] (0 for EndDuty/EndPairing). None reproduces the no-bias case.
                See `RL/utils.py::flight_gap_bias()`.
            return_logits: if True, return raw scores before softmax (Eq. 7) -- used for
                numerically stable REINFORCE log_prob computation
        Returns:
            probs or logits (Eq. 7-8): (N+2,) or (B, N+2)
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
            # Batched: q (B, d_model), k (N+2, d_model) -> scores (B, N+2)
            scores = torch.einsum('bd,nd->bn', q, k) / math.sqrt(self.d_model)
        else:
            scores = (k @ q) / math.sqrt(self.d_model)  # (N+2,) -- q_t^T k_j / sqrt(d) term of Eq. (7)

        if gap_bias is not None:
            scores = scores + self.gap_weight * gap_bias  # + w_gap * g_t(j)

        if action_bias is not None:
            scores = scores + action_bias

        scores = scores.masked_fill(mask == 0, float('-inf'))  # + b_mask_t(j; c) (Eq. 6)

        if return_logits:
            return scores
        return F.softmax(scores, dim=-1)
