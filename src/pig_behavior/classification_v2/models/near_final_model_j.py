"""Near-Final Model J.

Frozen Model A + Pre-GAP Local Spatial Attention + Class-Aware Source Gate.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pig_behavior.classification_v2.models.deep_local_joint_model import (
    DeepLocalJointRepresentationClassifier,
)

MODEL_J_ARCHITECTURE_VERSION = "near_final_model_j_local_attn_class_gate_v1"


class LocalSpatialAttention(nn.Module):
    """Tiny spatial attention pooling over pre-GAP feature maps [B*T, C, H, W]."""

    def __init__(self, in_channels: int = 256, token_dim: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.spatial_score = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.projection = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(
        self,
        spatial_map: torch.Tensor,
        batch_size: int,
        seq_len: int,
        temporal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pooled local token [B, token_dim] and attention weights [B, T, H, W]."""
        # [B*T, 1, H, W]
        scores = self.spatial_score(spatial_map)
        _, _, H, W = scores.shape

        # [B, T, H*W]
        scores_flat = scores.view(batch_size, seq_len, H * W)
        attn_flat = F.softmax(scores_flat, dim=-1)
        attn_map = attn_flat.view(batch_size, seq_len, 1, H, W)

        # Spatial pooling: [B, T, C, H*W] * [B, T, 1, H*W] -> sum over H*W -> [B, T, C]
        feat_flat = spatial_map.view(batch_size, seq_len, self.in_channels, H * W)
        pooled_frames = (feat_flat * attn_flat.unsqueeze(2)).sum(dim=-1)

        # Projection: [B, T, token_dim]
        frame_tokens = self.projection(pooled_frames)

        # Masked temporal mean: [B, token_dim]
        mask_weights = temporal_mask.to(frame_tokens.dtype).unsqueeze(-1)
        sum_tokens = (frame_tokens * mask_weights).sum(dim=1)
        denom = mask_weights.sum(dim=1).clamp_min(1e-7)
        local_token = sum_tokens / denom

        # Zero out if entire sequence is invalid
        has_valid = temporal_mask.any(dim=1, keepdim=True)
        local_token = torch.where(has_valid, local_token, torch.zeros_like(local_token))

        return local_token, attn_map.squeeze(2)


class NearFinalClassAwareGate(nn.Module):
    """Class-aware source gating over 12 multimodal sources."""

    def __init__(self, token_dim: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.num_classes = num_classes

        self.source_names = [
            "actor",
            "union",
            "localst",
            "structured",
            "interaction",
            "h5",
            "roi",
            "posture",
            "pb_probs",
            "pb_hidden",
            "actor_local",
            "union_local",
        ]

        self.source_projections = nn.ModuleDict(
            {
                "actor": nn.Sequential(
                    nn.Linear(128, token_dim), nn.LayerNorm(token_dim)
                ),
                "union": nn.Sequential(
                    nn.Linear(128, token_dim), nn.LayerNorm(token_dim)
                ),
                "localst": nn.Sequential(
                    nn.Linear(256, token_dim), nn.LayerNorm(token_dim)
                ),
                "structured": nn.Sequential(
                    nn.Linear(128, token_dim), nn.LayerNorm(token_dim)
                ),
                "interaction": nn.Sequential(
                    nn.Linear(64, token_dim), nn.LayerNorm(token_dim)
                ),
                "h5": nn.Sequential(
                    nn.Linear(32, token_dim), nn.LayerNorm(token_dim)
                ),
                "roi": nn.Sequential(
                    nn.Linear(32, token_dim), nn.LayerNorm(token_dim)
                ),
                "posture": nn.Sequential(
                    nn.Linear(3, token_dim), nn.LayerNorm(token_dim)
                ),
                "pb_probs": nn.Sequential(
                    nn.Linear(32, token_dim), nn.LayerNorm(token_dim)
                ),
                "pb_hidden": nn.Sequential(
                    nn.Linear(32, token_dim), nn.LayerNorm(token_dim)
                ),
                "actor_local": nn.Sequential(
                    nn.Linear(token_dim, token_dim), nn.LayerNorm(token_dim)
                ),
                "union_local": nn.Sequential(
                    nn.Linear(token_dim, token_dim), nn.LayerNorm(token_dim)
                ),
            }
        )

        self.score_net = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_classes),
        )

        self.residual_head = nn.Linear(token_dim, 1, bias=False)
        nn.init.zeros_(self.residual_head.weight)

    def forward(
        self,
        source_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return residual delta logits [B, 10] and gate weights [B, 10, 12]."""
        tokens = []
        masks = []
        for name in self.source_names:
            val = source_dict[name]
            tok = self.source_projections[name](val)
            tokens.append(tok)
            m = mask_dict[name]
            masks.append(m)

        # [B, 12_sources, 32]
        T = torch.stack(tokens, dim=1)
        # [B, 12_sources] -> [B, 1, 12_sources]
        M = torch.stack(masks, dim=1).unsqueeze(1)

        # [B, 12_sources, 10_classes] -> permute to [B, 10_classes, 12_sources]
        scores = self.score_net(T).transpose(1, 2)

        # Mask unavailable sources (use -1e4 for FP16 AMP stability)
        masked_scores = scores.masked_fill(~M, -1e4)

        # Softmax over sources: [B, 10_classes, 12_sources]
        weights = F.softmax(masked_scores, dim=-1)
        weights = torch.where(M, weights, torch.zeros_like(weights))

        # Class evidence: [B, 10_classes, 32]
        class_evidence = (T.unsqueeze(1) * weights.unsqueeze(-1)).sum(dim=2)

        # Residual logit: [B, 10]
        delta_logits = self.residual_head(class_evidence).squeeze(-1)

        return delta_logits, weights


class NearFinalModelJ(nn.Module):
    """Near-Final Model J: frozen Model A + pre-GAP local attention + 12-source gate."""

    def __init__(
        self,
        base_model: DeepLocalJointRepresentationClassifier,
        token_dim: int = 32,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.base_model = base_model

        # Freeze base model completely
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        # Preregistered tap: layer3 has 256 channels with 8x8 spatial resolution
        self.actor_spatial_attn = LocalSpatialAttention(
            in_channels=256, token_dim=token_dim
        )
        self.union_spatial_attn = LocalSpatialAttention(
            in_channels=256, token_dim=token_dim
        )
        self.gate = NearFinalClassAwareGate(
            token_dim=token_dim, num_classes=num_classes
        )

    def train(self, mode: bool = True) -> NearFinalModelJ:
        super().train(mode)
        # Keep base model in eval mode always
        self.base_model.eval()
        return self

    def extract_tapped_layer3_maps(
        self,
        images: torch.Tensor | None,
        visual_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Tap Layer 3 [B*T, 256, 8, 8] from Actor and Union ResNet backbones."""
        actor_map = None
        union_map = None

        if (
            images is not None
            and images.numel() > 0
            and self.base_model.image_encoder is not None
        ):
            b, t, c, h, w = images.shape
            x = images.view(b * t, c, h, w)
            backbone = self.base_model.image_encoder.frame_encoder.backbone
            with torch.no_grad():
                x1 = backbone.conv1(x)
                x1 = backbone.bn1(x1)
                x1 = backbone.relu(x1)
                x1 = backbone.maxpool(x1)
                x2 = backbone.layer1(x1)
                x3 = backbone.layer2(x2)
                x4 = backbone.layer3(x3)  # [B*T, 256, 8, 8]
                actor_map = x4

        if (
            visual_context is not None
            and visual_context.numel() > 0
            and self.base_model.visual_context_encoder is not None
        ):
            b, t, c, h, w = visual_context.shape
            x = visual_context.view(b * t, c, h, w)
            backbone = self.base_model.visual_context_encoder.frame_encoder.backbone
            with torch.no_grad():
                x1 = backbone.conv1(x)
                x1 = backbone.bn1(x1)
                x1 = backbone.relu(x1)
                x1 = backbone.maxpool(x1)
                x2 = backbone.layer1(x1)
                x3 = backbone.layer2(x2)
                x4 = backbone.layer3(x3)  # [B*T, 256, 8, 8]
                union_map = x4

        return actor_map, union_map

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if len(args) == 1 and hasattr(args[0], "training_batch"):
            aligned = args[0]
            tb = aligned.training_batch
            srcs = aligned.context_inputs
            call_inputs = dict(tb.model_inputs)
            call_inputs.update({
                "partner_behavior_probs": srcs.get("partner_behavior_probs"),
                "partner_behavior_hidden": srcs.get("partner_behavior_hidden"),
                "partner_behavior_mask": srcs.get("partner_behavior_mask"),
                "h5_structured": srcs.get("h5_structured"),
                "h5_mask": srcs.get("h5_mask"),
                "roi_sequence": srcs.get("roi_sequence"),
                "roi_validity": srcs.get("roi_validity"),
            })
        elif len(args) == 1 and isinstance(args[0], dict):
            call_inputs = dict(args[0])
            call_inputs.update(kwargs)
        else:
            call_inputs = dict(kwargs)

        # 1. Base Model Forward (Frozen)
        with torch.no_grad():
            out = self.base_model(**call_inputs)
            base_logits = out.behavior_logits

        device = base_logits.device
        dtype = base_logits.dtype
        B = base_logits.shape[0]

        # 2. Extract Layer 3 spatial feature maps
        actor_map = getattr(self.base_model, "_latest_spatial_maps", {}).get("actor", {}).get("layer3")
        union_map = getattr(self.base_model, "_latest_spatial_maps", {}).get("union", {}).get("layer3")
        expected_bt = B * 6
        if (
            actor_map is None
            or actor_map.shape[0] != expected_bt
            or union_map is None
            or union_map.shape[0] != expected_bt
        ):
            direct_act, direct_uni = self.extract_tapped_layer3_maps(
                call_inputs.get("image"),
                call_inputs.get("visual_context_image"),
            )
            if actor_map is None or actor_map.shape[0] != expected_bt:
                actor_map = direct_act
            if union_map is None or union_map.shape[0] != expected_bt:
                union_map = direct_uni

        # Temporal masks
        t_mask = call_inputs.get("length_mask", call_inputs.get("temporal_mask"))
        if t_mask is None:
            t_mask = torch.ones(B, 6, dtype=torch.bool, device=device)
        elif t_mask.dtype != torch.bool:
            t_mask = t_mask > 0
        T = t_mask.shape[1]

        # 3. Local spatial attention tokens
        if actor_map is not None:
            actor_local, actor_attn_map = self.actor_spatial_attn(
                actor_map, B, T, t_mask
            )
            actor_local_valid = t_mask.any(dim=1)
        else:
            actor_local = torch.zeros(B, 32, device=device, dtype=dtype)
            actor_local_valid = torch.zeros(B, dtype=torch.bool, device=device)
            actor_attn_map = torch.zeros(B, T, 8, 8, device=device, dtype=dtype)

        if union_map is not None:
            union_local, union_attn_map = self.union_spatial_attn(
                union_map, B, T, t_mask
            )
            union_local_valid = t_mask.any(dim=1)
        else:
            union_local = torch.zeros(B, 32, device=device, dtype=dtype)
            union_local_valid = torch.zeros(B, dtype=torch.bool, device=device)
            union_attn_map = torch.zeros(B, T, 8, 8, device=device, dtype=dtype)

        # 4. Source dictionary
        mb = out.m2_branches if hasattr(out, "m2_branches") else out.modulated_branches
        has_posture = (
            out.posture_logits is not None
            and out.posture_logits.numel() > 0
        )
        posture_logits = (
            out.posture_logits
            if has_posture
            else torch.zeros(B, 3, device=device, dtype=dtype)
        )
        posture_valid = (
            torch.ones(B, dtype=torch.bool, device=device)
            if has_posture
            else torch.zeros(B, dtype=torch.bool, device=device)
        )

        source_dict = {
            "actor": mb.actor128,
            "union": mb.union128,
            "localst": (
                out.local_representation
                if out.local_representation is not None
                else torch.zeros(B, 256, device=device, dtype=dtype)
            ),
            "structured": mb.structured128,
            "interaction": mb.interaction64,
            "h5": out.h5_context,
            "roi": out.roi_context,
            "posture": posture_logits,
            "pb_probs": out.partner_context,
            "pb_hidden": out.partner_hidden_context,
            "actor_local": actor_local,
            "union_local": union_local,
        }

        # 5. Source availability masks
        img_avail = call_inputs.get("image_available_mask")
        vis_avail = call_inputs.get("visual_context_available_mask")
        inter_avail = call_inputs.get("interaction_context_available_mask")
        h5_mask = call_inputs.get("h5_mask")
        roi_validity = call_inputs.get("roi_validity")
        pb_mask = call_inputs.get("partner_behavior_mask")
        pb_h_mask = call_inputs.get("partner_behavior_hidden_mask")

        if pb_h_mask is not None:
            pb_h_avail = pb_h_mask.any(dim=1)
        elif pb_mask is not None:
            pb_h_avail = pb_mask.any(dim=1)
        else:
            pb_h_avail = torch.zeros(B, dtype=torch.bool, device=device)

        mask_dict = {
            "actor": (
                img_avail.any(dim=1)
                if img_avail is not None
                else torch.ones(B, dtype=torch.bool, device=device)
            ),
            "union": (
                vis_avail.any(dim=1)
                if vis_avail is not None
                else torch.zeros(B, dtype=torch.bool, device=device)
            ),
            "localst": (
                img_avail.any(dim=1)
                if img_avail is not None
                else torch.ones(B, dtype=torch.bool, device=device)
            ),
            "structured": torch.ones(B, dtype=torch.bool, device=device),
            "interaction": (
                inter_avail.any(dim=1)
                if inter_avail is not None
                else torch.zeros(B, dtype=torch.bool, device=device)
            ),
            "h5": (
                h5_mask.any(dim=1)
                if h5_mask is not None
                else torch.zeros(B, dtype=torch.bool, device=device)
            ),
            "roi": (
                roi_validity.any(dim=(1, 2))
                if roi_validity is not None
                else torch.zeros(B, dtype=torch.bool, device=device)
            ),
            "posture": posture_valid,
            "pb_probs": (
                pb_mask.any(dim=1)
                if pb_mask is not None
                else torch.zeros(B, dtype=torch.bool, device=device)
            ),
            "pb_hidden": pb_h_avail,
            "actor_local": actor_local_valid,
            "union_local": union_local_valid,
        }

        delta_logits, weights = self.gate(source_dict, mask_dict)
        final_logits = base_logits + delta_logits

        return (
            final_logits,
            base_logits,
            weights,
            actor_attn_map,
            union_attn_map,
        )


__all__ = [
    "MODEL_J_ARCHITECTURE_VERSION",
    "LocalSpatialAttention",
    "NearFinalClassAwareGate",
    "NearFinalModelJ",
]
