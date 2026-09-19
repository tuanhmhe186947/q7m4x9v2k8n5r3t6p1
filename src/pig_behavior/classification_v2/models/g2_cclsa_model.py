"""G2-CCLSA: M2 plus actor-only structured-conditioned localized spatial attention.

This is the final reserved model-level intervention for classification_v2:
  - Base scientific model: M2-VFT (ResNet + Spatial + Interaction + Visual Context).
  - Actor ResNet layer3 [B*T, 256, 8, 8] tapped before spatial pooling.
  - Query: bias-free Linear(46 -> 64) from canonical 46D structured features.
  - Keys: bias-free Conv2d(256 -> 64, kernel=1) from layer3 map.
  - Scaled dot-product attention over 8x8 = 64 spatial positions.
  - Values: original layer3 256D spatial vectors -> weighted sum -> 256D local vector.
  - Residual projection: bias-free Linear(256 -> 128), initialized to EXACT ZERO.
  - actor_token_new = actor_token_M2 + valid_frame_mask * local_correction.
  - Expected trainable parameter delta: 2944 + 16384 + 32768 = 52096.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from pig_behavior.classification_v2.features.spatial_schema import (
    SPATIAL_PREDICTIVE_GROUP_NAMES,
)
from pig_behavior.classification_v2.models.multimodal_fusion import (
    MultimodalFusionClassifier,
    MultimodalFusionConfig,
    _combined_mask,
    _masked_values,
)

G2_ARCHITECTURE_VERSION: str = "g2_cclsa_actor_conditioned_spatial_v1"
EXPECTED_G2_PARAM_DELTA: int = 52096
STRUCTURED_QUERY_DIM: int = 46
ATTENTION_KEY_DIM: int = 64
ATTENTION_VALUE_DIM: int = 256


class G2CCLSAClassifier(MultimodalFusionClassifier):
    """M2 late-fusion classifier enhanced with Actor-only CCLSA before pooling."""

    def __init__(self, config: MultimodalFusionConfig) -> None:
        super().__init__(config)
        if self.image_encoder is None:
            raise ValueError("G2-CCLSA requires enable_image=True")

        hidden_dim = config.image_embedding_dim  # 128
        # 1. Query projection: bias-free Linear(46 -> 64) [2,944 params]
        self.query_proj = nn.Linear(
            STRUCTURED_QUERY_DIM,
            ATTENTION_KEY_DIM,
            bias=False,
        )

        # 2. Key projection: bias-free Conv2d(256 -> 64, kernel=1) [16,384 params]
        self.key_proj = nn.Conv2d(
            ATTENTION_VALUE_DIM,
            ATTENTION_KEY_DIM,
            kernel_size=1,
            bias=False,
        )

        # 3. Residual projection: bias-free Linear(256 -> 128) [32,768 params]
        self.residual_proj = nn.Linear(
            ATTENTION_VALUE_DIM,
            hidden_dim,
            bias=False,
        )

        # Strict Step-0 zero initialization
        nn.init.zeros_(self.residual_proj.weight)

    def _encode_actor_branch(
        self,
        *,
        image: torch.Tensor,
        spatial_features: dict[str, torch.Tensor] | None = None,
        length_mask: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        available_mask: torch.Tensor | None = None,
        quality_mask: torch.Tensor | None = None,
        time_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute single-pass ResNet + CCLSA conditioned on canonical 46D features."""
        if self.image_encoder is None:
            raise RuntimeError("image_encoder is not initialized")
        if spatial_features is None:
            raise ValueError("G2-CCLSA requires spatial_features dict for conditioning")

        # Verify canonical 46D features presence
        missing_groups = [
            group for group in SPATIAL_PREDICTIVE_GROUP_NAMES
            if group not in spatial_features
        ]
        if missing_groups:
            raise ValueError(f"Missing spatial feature groups for G2-CCLSA: {missing_groups}")

        if image.ndim != 5:
            raise ValueError("image must have shape [B, T, 3, H, W]")
        if image.shape[2] != 3:
            raise ValueError("image channel dimension must be 3")

        mask = _combined_mask(
            length_mask,
            observed_mask,
            image.shape[:2],
            available_mask=available_mask,
            quality_mask=quality_mask,
            branch_name="image",
        )
        batch_size, sequence_len = image.shape[:2]
        clean_image = _masked_values(image, mask, branch_name="image")
        clean_image = self.image_encoder._normalize(clean_image)
        flat_image = clean_image.reshape(
            batch_size * sequence_len,
            *image.shape[2:],
        )

        # Single ResNet forward pass up to layer3, then layer4
        fe = self.image_encoder.frame_encoder
        if hasattr(fe, "layer3") and hasattr(fe, "layer4"):
            x = fe.conv1(flat_image)
            x = fe.bn1(x)
            x = fe.relu(x)
            x = fe.maxpool(x)
            x = fe.layer1(x)
            x = fe.layer2(x)
            layer3_map = fe.layer3(x)  # [N, 256, 8, 8]

            x4 = fe.layer4(layer3_map)  # [N, 512, 4, 4]
            x_pool = fe.avgpool(x4)     # [N, 512, 1, 1]
            x_flat = torch.flatten(x_pool, 1)  # [N, 512]
            encoded = fe.fc(x_flat)     # [N, 512]
        else:
            raise NotImplementedError(
                f"G2-CCLSA requires ResNet backbone with layer3/layer4, got {type(fe)}"
            )

        # 1. Unchanged M2 actor token sequence: [B, T, 128]
        encoded_seq = encoded.reshape(batch_size, sequence_len, -1)
        actor_token_m2 = self.image_encoder.temporal_projection(encoded_seq)

        # 2. Canonical 46D structured features: [B, T, 46]
        structured_tensors = [
            spatial_features[group_name]
            for group_name in SPATIAL_PREDICTIVE_GROUP_NAMES
        ]
        structured46 = torch.cat(structured_tensors, dim=-1)
        structured_flat = structured46.reshape(batch_size * sequence_len, STRUCTURED_QUERY_DIM)

        # 3. Query projection: [N, 1, 64]
        query = self.query_proj(structured_flat).unsqueeze(1)

        # 4. Key projection: [N, 64_spatial, 64_channels]
        keys = self.key_proj(layer3_map)  # [N, 64, 8, 8]
        keys = keys.flatten(2).transpose(1, 2)  # [N, 64, 64]

        # 5. Scaled dot-product attention over 8x8 = 64 spatial positions
        scale = 1.0 / math.sqrt(ATTENTION_KEY_DIM)  # 1.0 / 8.0 = 0.125
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) * scale  # [N, 1, 64]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N, 1, 64]

        # 6. Values: original layer3 256D spatial vectors -> [N, 64, 256]
        values = layer3_map.flatten(2).transpose(1, 2)  # [N, 64, 256]

        # 7. Weighted sum -> 256D local vector -> [N, 256]
        local_vector = torch.bmm(attn_weights, values).squeeze(1)

        # 8. Bias-free residual projection -> [N, 128]
        local_correction_flat = self.residual_proj(local_vector)
        local_correction = local_correction_flat.reshape(batch_size, sequence_len, -1)

        # 9. Gated by valid frame mask (no modification to invalid slots)
        valid_frame_mask = mask.unsqueeze(-1).to(actor_token_m2.dtype)
        actor_token_new = actor_token_m2 + valid_frame_mask * local_correction

        # 10. Unchanged temporal Transformer
        return self.image_encoder.temporal_encoder(
            actor_token_new,
            mask,
            time_delta=time_delta,
        )


__all__ = [
    "ATTENTION_KEY_DIM",
    "ATTENTION_VALUE_DIM",
    "EXPECTED_G2_PARAM_DELTA",
    "G2_ARCHITECTURE_VERSION",
    "G2CCLSAClassifier",
    "STRUCTURED_QUERY_DIM",
]
