"""
Phase 2: DNN Model for successor prediction
Tahir paper architecture (Section 3.2):
  Input:  X_i  shape (|D_ib|, 3*9=27)   — variable-length rows
  Output: probability over |D_ib| candidates  → softmax

Architecture:
  [Embedding(categorical)] → [Conv1D(0-3 layers)] → [Dense(1-5 layers)] → softmax

We use a Set-pooling approach:
  Each row (=one successor candidate) passes through shared Conv→Dense layers
  to produce a scalar score, then softmax over all rows.

Best hyperparameters found via Bayesian search in paper (Table 3):
  We use the reported best config as default.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import List, Tuple, Dict

# ── Feature layout in X_i rows (27 cols) ────────────────────────────────────
# Indices for categorical features (will be embedded):
# base_dep_airport=0, base_arr_airport=1, base_base=2, base_aircraft=3
# fi_dep_airport=9,   fi_arr_airport=10,  fi_base=11, fi_aircraft=12
# fj_dep_airport=18,  fj_arr_airport=19,  fj_base=20, fj_aircraft=21

CAT_INDICES  = [0, 1, 2, 3, 9, 10, 11, 12, 18, 19, 20, 21]   # 12 categorical cols
NUM_INDICES  = [4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26]  # 15 numerical cols


def build_model(
    n_airports: int,
    n_aircraft: int,
    embedding_dim: int = 10,
    num_conv_layers: int = 1,
    num_filters: int = 128,
    filter_size: int = 3,
    num_dense_layers: int = 2,
    neurons_per_layer: int = 256,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Build the DNN model.
    Input shape: (batch, max_candidates, 27)  — padded with mask
    Output shape: (batch, max_candidates)     — log-softmax scores

    We use a per-candidate shared network (weight-tied across candidates).
    """
    # ── Input ────────────────────────────────────────────────────────────────
    inp = keras.Input(shape=(None, 27), name="xi_matrix")  # (B, K, 27)

    # ── Embedding for categorical features ──────────────────────────────────
    # Airport features appear at positions 0,1,2 (base block) and 9,10,11 (fi) and 18,19,20 (fj)
    # Aircraft at 3, 12, 21. We use a shared embedding table per feature type.

    # Split: 4 airport indices per block × 3 blocks = 12 cat cols, 15 num cols
    # Embed airports → embedding_dim, aircraft → embedding_dim
    # then concatenate with numerical, project to hidden_dim

    # For simplicity, concatenate embedded categorical + raw numerical per block
    # Airport embedding (shared across all 9 airport positions)
    ap_emb_layer = layers.Embedding(n_airports + 1, embedding_dim, name="airport_emb")
    ac_emb_layer = layers.Embedding(n_aircraft + 1, embedding_dim, name="aircraft_emb")

    def embed_block(block_offset, raw):
        """Embed one 9-feature block starting at block_offset in raw."""
        dep_ap  = tf.cast(raw[..., block_offset + 0], tf.int32)
        arr_ap  = tf.cast(raw[..., block_offset + 1], tf.int32)
        base_ap = tf.cast(raw[..., block_offset + 2], tf.int32)
        ac      = tf.cast(raw[..., block_offset + 3], tf.int32)
        nums    = raw[..., block_offset + 4 : block_offset + 9]  # 5 numericals

        e_dep  = ap_emb_layer(dep_ap)
        e_arr  = ap_emb_layer(arr_ap)
        e_base = ap_emb_layer(base_ap)
        e_ac   = ac_emb_layer(ac)

        return tf.concat([e_dep, e_arr, e_base, e_ac, nums], axis=-1)
        # shape: (B, K, 4*emb_dim + 5)

    base_block = embed_block(0,  inp)
    fi_block   = embed_block(9,  inp)
    fj_block   = embed_block(18, inp)

    x = tf.concat([base_block, fi_block, fj_block], axis=-1)
    # shape: (B, K, 3*(4*emb_dim+5))

    # ── Conv1D layers (applied across K candidates dimension) ───────────────
    for _ in range(num_conv_layers):
        x = layers.Conv1D(num_filters, filter_size, padding="same", activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    # ── Dense layers (shared/time-distributed across K candidates) ──────────
    for i in range(num_dense_layers):
        x = layers.TimeDistributed(
            layers.Dense(neurons_per_layer, activation="relu"),
            name=f"dense_{i}"
        )(x)
        x = layers.Dropout(dropout_rate)(x)

    # ── Output: scalar score per candidate, then softmax ────────────────────
    logits = layers.TimeDistributed(layers.Dense(1), name="logits")(x)
    logits = tf.squeeze(logits, axis=-1)   # (B, K)

    # Mask: positions with all-zero input row are padding
    # We'll handle masking in the loss function instead
    out = layers.Activation("softmax", name="probs")(logits)

    model = keras.Model(inputs=inp, outputs=out, name="tahir_dnn")
    return model


def masked_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor,
                         mask: tf.Tensor) -> tf.Tensor:
    """
    Categorical cross-entropy with variable-length masking.
    y_true: (B,) integer labels
    y_pred: (B, K) probabilities
    mask:   (B, K) float32, 1=valid candidate, 0=padding
    """
    # One-hot
    K = tf.shape(y_pred)[1]
    oh = tf.one_hot(tf.cast(y_true, tf.int32), K)

    # Mask invalid positions (set to very small value before softmax would already handle,
    # but we also need to re-normalise for the loss)
    masked_pred = y_pred * mask
    masked_pred = masked_pred / (tf.reduce_sum(masked_pred, axis=-1, keepdims=True) + 1e-9)

    loss = -tf.reduce_sum(oh * tf.math.log(masked_pred + 1e-9), axis=-1)
    return tf.reduce_mean(loss)
