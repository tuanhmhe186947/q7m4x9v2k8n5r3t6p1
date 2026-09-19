"""Final Single-Model K: Model J + Localized Motion Difference on Attended Visual Regions.

Task ID: FINAL-SINGLE-K-JPLUSLOCALMOTION-20260828
Base Model: DeepLocalJointRepresentationClassifier (Frozen High-Ceiling Model A)
14 Multimodal Sources:
1. actor, 2. union, 3. localst, 4. structured, 5. interaction, 6. h5, 7. roi,
8. posture, 9. pb_probs, 10. pb_hidden, 11. actor_local, 12. union_local,
13. actor_local_motion, 14. union_local_motion.
Zero-init Residual Output Correction.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pig_behavior.classification_v2.models.deep_local_joint_model import (
    DeepLocalJointRepresentationClassifier,
)

MODEL_K_ARCHITECTURE_VERSION = "final_single_k_jplus_localmotion_v1"


class LocalSpatialAndMotionAttention(nn.Module):
    """Spatial attention pooling and localized motion calculation on pre-GAP feature maps."""

    def __init__(self, in_channels: int = 256, token_dim: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.token_dim = token_dim
        self.spatial_score = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.spatial_projection = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.motion_projection = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(
        self,
        spatial_map: torch.Tensor,
        batch_size: int,
        seq_len: int,
        temporal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (local_token, motion_token, motion_valid_mask, attention_weights).

        spatial_map: [B*T, C, H, W]
        temporal_mask: [B, T] bool
        """
        expected_bt = batch_size * seq_len
        if spatial_map.shape[0] != expected_bt:
            if spatial_map.shape[0] < expected_bt:
                pad = torch.zeros(
                    (expected_bt - spatial_map.shape[0], *spatial_map.shape[1:]),
                    device=spatial_map.device,
                    dtype=spatial_map.dtype,
                )
                spatial_map = torch.cat([spatial_map, pad], dim=0)
            else:
                spatial_map = spatial_map[:expected_bt]

        # 1. Spatial Attention Scores: [B*T, 1, H, W]
        scores = self.spatial_score(spatial_map)
        _, _, H, W = scores.shape

        # [B, T, H*W]
        scores_flat = scores.view(batch_size, seq_len, H * W)
        attn_flat = F.softmax(scores_flat, dim=-1)
        attn_map = attn_flat.view(batch_size, seq_len, 1, H, W)

        # 2. Local Spatial Token Pooling
        # [B, T, C, H*W]
        feat_flat = spatial_map.view(batch_size, seq_len, self.in_channels, H * W)
        pooled_frames = (feat_flat * attn_flat.unsqueeze(2)).sum(dim=-1) # [B, T, C]

        frame_tokens = self.spatial_projection(pooled_frames) # [B, T, token_dim]
        mask_weights = temporal_mask.to(frame_tokens.dtype).unsqueeze(-1) # [B, T, 1]
        sum_tokens = (frame_tokens * mask_weights).sum(dim=1)
        denom = mask_weights.sum(dim=1).clamp_min(1e-7)
        local_token = sum_tokens / denom
        has_valid_spatial = temporal_mask.any(dim=1, keepdim=True)
        local_token = torch.where(has_valid_spatial, local_token, torch.zeros_like(local_token))

        # 3. Localized Motion Token
        if seq_len > 1:
            # Valid consecutive pairs: [B, T-1]
            pair_mask = temporal_mask[:, :-1] & temporal_mask[:, 1:]
            
            # Features and attention at consecutive timesteps
            feat_t0 = feat_flat[:, :-1] # [B, T-1, C, H*W]
            feat_t1 = feat_flat[:, 1:]  # [B, T-1, C, H*W]
            attn_t0 = attn_flat[:, :-1] # [B, T-1, H*W]
            attn_t1 = attn_flat[:, 1:]  # [B, T-1, H*W]

            # Pair attention: 0.5 * (A_(t-1) + A_t)
            pair_attn = 0.5 * (attn_t0 + attn_t1) # [B, T-1, H*W]

            # Feature-change magnitude: D_t = abs(F_t - F_(t-1))
            d_feat = torch.abs(feat_t1 - feat_t0) # [B, T-1, C, H*W]

            # Weighted sum over spatial dimensions: LOCAL_MOTION_t
            local_motion_t = (d_feat * pair_attn.unsqueeze(2)).sum(dim=-1) # [B, T-1, C]

            # Motion projection: [B, T-1, token_dim]
            motion_frame_tokens = self.motion_projection(local_motion_t)

            # Masked temporal mean across valid pairs
            pair_weights = pair_mask.to(motion_frame_tokens.dtype).unsqueeze(-1) # [B, T-1, 1]
            sum_motion = (motion_frame_tokens * pair_weights).sum(dim=1)
            denom_motion = pair_weights.sum(dim=1).clamp_min(1e-7)
            motion_token = sum_motion / denom_motion

            has_valid_pair = pair_mask.any(dim=1, keepdim=True) # [B, 1]
            motion_token = torch.where(has_valid_pair, motion_token, torch.zeros_like(motion_token))
            motion_valid = has_valid_pair.squeeze(-1) # [B]
        else:
            motion_token = local_token.new_zeros(batch_size, self.token_dim)
            motion_valid = local_token.new_zeros(batch_size, dtype=torch.bool)

        return local_token, motion_token, motion_valid, attn_map.squeeze(2)


class ModelKClassAwareGate(nn.Module):
    """Class-aware source gating over 14 multimodal sources."""

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
            "actor_local_motion",
            "union_local_motion",
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
                "actor_local_motion": nn.Sequential(
                    nn.Linear(token_dim, token_dim), nn.LayerNorm(token_dim)
                ),
                "union_local_motion": nn.Sequential(
                    nn.Linear(token_dim, token_dim), nn.LayerNorm(token_dim)
                ),
            }
        )

        self.score_net = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_classes),
        )

        # Zero-initialized residual head
        self.residual_head = nn.Linear(token_dim, 1, bias=False)
        nn.init.zeros_(self.residual_head.weight)

    def forward(
        self,
        source_dict: dict[str, torch.Tensor],
        source_mask: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute delta logits and source weights.

        Returns:
            delta_logits: [B, num_classes]
            source_weights: [B, num_classes, 14]
        """
        projected = []
        masks = []
        for name in self.source_names:
            feat = source_dict[name]
            p_feat = self.source_projections[name](feat)
            projected.append(p_feat)
            masks.append(source_mask[name])

        # S = 14
        # P: [B, 14, token_dim]
        P = torch.stack(projected, dim=1)
        # M: [B, 14] -> [B, 1, 14]
        M = torch.stack(masks, dim=1).unsqueeze(1)

        # Scores: [B, 14, num_classes] -> transpose to [B, num_classes, 14]
        scores = self.score_net(P).transpose(1, 2)

        # Masked Softmax over sources (use -1e4 for FP16 AMP compatibility)
        masked_scores = scores.masked_fill(~M, -1e4)
        W = F.softmax(masked_scores, dim=-1) # [B, num_classes, 14]
        W = torch.where(M, W, torch.zeros_like(W))

        # Class-conditioned source aggregation: [B, num_classes, token_dim]
        # P: [B, 1, 14, token_dim], W: [B, num_classes, 14, 1]
        C_rep = (P.unsqueeze(1) * W.unsqueeze(-1)).sum(dim=2)

        # Residual delta logits: [B, num_classes, 1] -> [B, num_classes]
        delta_logits = self.residual_head(C_rep).squeeze(-1)

        return delta_logits, W


class NearFinalModelK(nn.Module):
    """Near-Final Model K Wrapper.

    Frozen Base Model A + Local Spatial & Motion Attention + 14-Source Gate.
    """

    def __init__(self, base_model: DeepLocalJointRepresentationClassifier) -> None:
        super().__init__()
        self.base_model = base_model

        # Freeze base model completely
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        # Actor & Union Local Spatial & Motion Attention
        # Pre-GAP Layer 3 feature map has C=256
        self.actor_spatial_motion = LocalSpatialAndMotionAttention(in_channels=256, token_dim=32)
        self.union_spatial_motion = LocalSpatialAndMotionAttention(in_channels=256, token_dim=32)

        # 14-Source Class-Aware Gate
        self.gate = ModelKClassAwareGate(token_dim=32, num_classes=10)

    def train(self, mode: bool = True) -> NearFinalModelK:
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
                x4 = backbone.layer3(x3) # [B*T, 256, 8, 8]
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
                x4 = backbone.layer3(x3) # [B*T, 256, 8, 8]
                union_map = x4

        return actor_map, union_map

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass computing frozen Model A logits + zero-init gated residual delta.

        Returns:
            final_logits: [B, 10]
            base_logits: [B, 10]
            source_weights: [B, 10, 14]
            actor_attn_weights: [B, T, H, W]
            union_attn_weights: [B, T, H, W]
        """
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
            base_out = self.base_model(**call_inputs)
            base_logits = base_out.behavior_logits

        B = base_logits.shape[0]
        device = base_logits.device
        dtype = base_logits.dtype

        # 2. Extract Layer 3 spatial feature maps from base model hook or direct tap
        actor_map = self.base_model._latest_spatial_maps.get("actor", {}).get("layer3")
        union_map = self.base_model._latest_spatial_maps.get("union", {}).get("layer3")
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

        # 3. Spatial & Motion Attention Tokens
        t_mask = call_inputs.get("length_mask", call_inputs.get("temporal_mask"))
        if t_mask is None:
            t_mask = torch.ones(B, 6, dtype=torch.bool, device=device)
        elif t_mask.dtype != torch.bool:
            t_mask = t_mask > 0
        T = t_mask.shape[1]

        if actor_map is not None:
            act_local, act_motion, act_motion_valid, act_attn = self.actor_spatial_motion(
                actor_map, B, T, t_mask
            )
            act_local_valid = t_mask.any(dim=1)
        else:
            act_local = torch.zeros(B, 32, device=device, dtype=dtype)
            act_motion = torch.zeros(B, 32, device=device, dtype=dtype)
            act_local_valid = torch.zeros(B, dtype=torch.bool, device=device)
            act_motion_valid = torch.zeros(B, dtype=torch.bool, device=device)
            act_attn = torch.zeros(B, T, 8, 8, device=device, dtype=dtype)

        if union_map is not None:
            uni_local, uni_motion, uni_motion_valid, uni_attn = self.union_spatial_motion(
                union_map, B, T, t_mask
            )
            uni_local_valid = t_mask.any(dim=1)
        else:
            uni_local = torch.zeros(B, 32, device=device, dtype=dtype)
            uni_motion = torch.zeros(B, 32, device=device, dtype=dtype)
            uni_local_valid = torch.zeros(B, dtype=torch.bool, device=device)
            uni_motion_valid = torch.zeros(B, dtype=torch.bool, device=device)
            uni_attn = torch.zeros(B, T, 8, 8, device=device, dtype=dtype)

        # 4. Construct 14 Sources
        mb = base_out.m2_branches if hasattr(base_out, "m2_branches") else base_out.modulated_branches
        has_posture = (
            base_out.posture_logits is not None
            and base_out.posture_logits.numel() > 0
        )
        posture_logits = (
            base_out.posture_logits
            if has_posture
            else torch.zeros(B, 3, device=device, dtype=dtype)
        )
        posture_valid = (
            torch.ones(B, dtype=torch.bool, device=device)
            if has_posture
            else torch.zeros(B, dtype=torch.bool, device=device)
        )

        sources = {
            "actor": mb.actor128,
            "union": mb.union128,
            "localst": (
                base_out.local_representation
                if base_out.local_representation is not None
                else torch.zeros(B, 256, device=device, dtype=dtype)
            ),
            "structured": mb.structured128,
            "interaction": mb.interaction64,
            "h5": base_out.h5_context,
            "roi": base_out.roi_context,
            "posture": posture_logits,
            "pb_probs": base_out.partner_context,
            "pb_hidden": base_out.partner_hidden_context,
            "actor_local": act_local,
            "union_local": uni_local,
            "actor_local_motion": act_motion,
            "union_local_motion": uni_motion,
        }

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

        actor_v = (
            (img_avail.any(dim=1) if img_avail.ndim > 1 else img_avail.bool())
            if img_avail is not None
            else torch.ones(B, dtype=torch.bool, device=device)
        )
        union_v = (
            (vis_avail.any(dim=1) if vis_avail.ndim > 1 else vis_avail.bool())
            if vis_avail is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )
        localst_v = (
            (img_avail.any(dim=1) if img_avail.ndim > 1 else img_avail.bool())
            if img_avail is not None
            else torch.ones(B, dtype=torch.bool, device=device)
        )
        inter_v = (
            (inter_avail.any(dim=1) if inter_avail.ndim > 1 else inter_avail.bool())
            if inter_avail is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )
        h5_v = (
            (h5_mask.any(dim=1) if h5_mask.ndim > 1 else h5_mask.bool())
            if h5_mask is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )
        roi_v = (
            (
                roi_validity.any(dim=tuple(range(1, roi_validity.ndim)))
                if roi_validity.ndim > 1
                else roi_validity.bool()
            )
            if roi_validity is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )
        pb_v = (
            (pb_mask.any(dim=1) if pb_mask.ndim > 1 else pb_mask.bool())
            if pb_mask is not None
            else torch.zeros(B, dtype=torch.bool, device=device)
        )

        mask_dict = {
            "actor": actor_v,
            "union": union_v,
            "localst": localst_v,
            "structured": torch.ones(B, dtype=torch.bool, device=device),
            "interaction": inter_v,
            "h5": h5_v,
            "roi": roi_v,
            "posture": posture_valid,
            "pb_probs": pb_v,
            "pb_hidden": pb_h_avail,
            "actor_local": act_local_valid,
            "union_local": uni_local_valid,
            "actor_local_motion": act_motion_valid,
            "union_local_motion": uni_motion_valid,
        }

        delta_logits, weights = self.gate(sources, mask_dict)
        final_logits = base_logits + delta_logits

        return final_logits, base_logits, weights, act_attn, uni_attn
