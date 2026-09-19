"""Factor Ablation Models for Model J (Task FINAL-J-LOCAL-GATE-ABLATION-20260916).

Decomposes Model J into a 2x2 architectural factor matrix plus neutral scaffold control:
- Config N: Model A + 10 Sources + Fixed Neutral Source Weighting + Residual Head
- Config G: Model A + 10 Sources + Learned Class-Aware Gate + Residual Head
- Config L: Model A + 12 Sources (with LocalSpatial) + Fixed Neutral Weighting + Residual Head
- Config J: Full physical Model J (LocalSpatial + Learned Class-Aware Gate)
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pig_behavior.classification_v2.models.deep_local_joint_model import (
    DeepLocalJointRepresentationClassifier,
)
from pig_behavior.classification_v2.models.near_final_model_j import (
    LocalSpatialAttention,
)

ORIGINAL_10_SOURCES = [
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
]

FULL_12_SOURCES = ORIGINAL_10_SOURCES + [
    "actor_local",
    "union_local",
]

SOURCE_DIMS = {
    "actor": 128,
    "union": 128,
    "localst": 256,
    "structured": 128,
    "interaction": 64,
    "h5": 32,
    "roi": 32,
    "posture": 3,
    "pb_probs": 32,
    "pb_hidden": 32,
    "actor_local": 32,
    "union_local": 32,
}

SOURCE_SEED_OFFSETS = {
    "actor": 201,
    "union": 202,
    "localst": 203,
    "structured": 204,
    "interaction": 205,
    "h5": 206,
    "roi": 207,
    "posture": 208,
    "pb_probs": 209,
    "pb_hidden": 210,
    "actor_local": 211,
    "union_local": 212,
}


def build_deterministic_source_projections(
    source_names: list[str],
    token_dim: int = 32,
    base_seed: int = 240494961,
) -> nn.ModuleDict:
    """Build source projections with dedicated per-submodule RNG seeds.

    Guarantees that shared source projections have identical initial tensors
    across all ablation configurations (N, G, L, J).
    """
    projections = nn.ModuleDict()
    for name in source_names:
        dim = SOURCE_DIMS[name]
        offset = SOURCE_SEED_OFFSETS[name]
        torch.manual_seed(base_seed + offset)
        proj = nn.Sequential(
            nn.Linear(dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        projections[name] = proj
    return projections


def build_deterministic_lsa(
    name: str,
    in_channels: int = 256,
    token_dim: int = 32,
    base_seed: int = 240494961,
) -> LocalSpatialAttention:
    """Build LocalSpatialAttention with deterministic dedicated RNG seed."""
    offset = 101 if name == "actor" else 102
    torch.manual_seed(base_seed + offset)
    return LocalSpatialAttention(in_channels=in_channels, token_dim=token_dim)


def build_deterministic_score_net(
    token_dim: int = 32,
    num_classes: int = 10,
    base_seed: int = 240494961,
) -> nn.Sequential:
    """Build 2-layer score net with deterministic dedicated RNG seed."""
    torch.manual_seed(base_seed + 301)
    return nn.Sequential(
        nn.Linear(token_dim, token_dim),
        nn.GELU(),
        nn.Linear(token_dim, num_classes),
    )


def compute_module_subhashes(module: nn.Module) -> dict[str, str]:
    """Compute per-parameter SHA256 hex digests for auditing initialization parity."""
    hashes: dict[str, str] = {}
    for name, param in module.named_parameters():
        data_bytes = param.detach().cpu().contiguous().numpy().tobytes()
        hashes[name] = hashlib.sha256(data_bytes).hexdigest()
    return hashes


class FactorAblationGate(nn.Module):
    """Modular class-aware gate supporting learned and fixed neutral weighting."""

    def __init__(
        self,
        source_names: list[str],
        token_dim: int = 32,
        num_classes: int = 10,
        learned_gate: bool = True,
        base_seed: int = 240494961,
    ) -> None:
        super().__init__()
        self.source_names = list(source_names)
        self.token_dim = token_dim
        self.num_classes = num_classes
        self.learned_gate = learned_gate

        self.source_projections = build_deterministic_source_projections(
            self.source_names, token_dim=token_dim, base_seed=base_seed
        )

        self.score_net = build_deterministic_score_net(
            token_dim=token_dim, num_classes=num_classes, base_seed=base_seed
        )
        if not learned_gate:
            for param in self.score_net.parameters():
                nn.init.zeros_(param)
                param.requires_grad = False

        self.residual_head = nn.Linear(token_dim, 1, bias=False)
        nn.init.zeros_(self.residual_head.weight)

    def forward(
        self,
        source_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute residual delta logits [B, 10] and gate weights [B, 10, K]."""
        tokens = []
        masks = []
        for name in self.source_names:
            val = source_dict[name]
            tok = self.source_projections[name](val)
            tokens.append(tok)
            masks.append(mask_dict[name])

        # T: [B, K_sources, token_dim]
        T = torch.stack(tokens, dim=1)
        # M: [B, 1, K_sources]
        M = torch.stack(masks, dim=1).unsqueeze(1)

        if self.learned_gate:
            # [B, K_sources, num_classes] -> [B, num_classes, K_sources]
            scores = self.score_net(T).transpose(1, 2)
        else:
            # Deterministic zero scores for neutral uniform weighting
            B = T.shape[0]
            scores = torch.zeros(
                B, self.num_classes, len(self.source_names),
                device=T.device, dtype=T.dtype
            )

        # Mask unavailable sources (-1e4 for FP16 AMP stability)
        masked_scores = scores.masked_fill(~M, -1e4)

        # Softmax over available sources
        weights = F.softmax(masked_scores, dim=-1)
        weights = torch.where(M, weights, torch.zeros_like(weights))

        # Class evidence: [B, num_classes, token_dim]
        class_evidence = (T.unsqueeze(1) * weights.unsqueeze(-1)).sum(dim=2)

        # Residual delta logit: [B, num_classes]
        delta_logits = self.residual_head(class_evidence).squeeze(-1)

        return delta_logits, weights


class BaseAblationModel(nn.Module):
    """Common base forward logic for Model J ablation variants."""

    def __init__(
        self,
        base_model: DeepLocalJointRepresentationClassifier,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def train(self, mode: bool = True) -> BaseAblationModel:
        super().train(mode)
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

    def _prepare_call_inputs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Unify input dispatch across AlignedDataPlusBatch and dictionaries."""
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
        return call_inputs


class NearFinalModelN(BaseAblationModel):
    """Config N: Neutral Scaffold Control.

    Model A + 10 Original Evidence Sources + Fixed Neutral Gate + Residual Head.
    No local visual branches. Gate score net permanently frozen at zero.
    """

    def __init__(
        self,
        base_model: DeepLocalJointRepresentationClassifier,
        token_dim: int = 32,
        num_classes: int = 10,
        base_seed: int = 240494961,
    ) -> None:
        super().__init__(base_model)
        self.gate = FactorAblationGate(
            source_names=ORIGINAL_10_SOURCES,
            token_dim=token_dim,
            num_classes=num_classes,
            learned_gate=False,
            base_seed=base_seed,
        )

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        call_inputs = self._prepare_call_inputs(*args, **kwargs)

        with torch.no_grad():
            out = self.base_model(**call_inputs)
            base_logits = out.behavior_logits

        device = base_logits.device
        dtype = base_logits.dtype
        B = base_logits.shape[0]

        mb = out.m2_branches if hasattr(out, "m2_branches") else out.modulated_branches
        has_posture = (
            out.posture_logits is not None and out.posture_logits.numel() > 0
        )
        posture_logits = (
            out.posture_logits if has_posture else torch.zeros(B, 3, device=device, dtype=dtype)
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
        }

        delta_logits, weights = self.gate(source_dict, mask_dict)
        final_logits = base_logits + delta_logits

        return final_logits, base_logits, weights


class NearFinalModelG(BaseAblationModel):
    """Config G: Learned Class-Aware Gate Only.

    Model A + 10 Original Evidence Sources + Learned Class-Aware Gate + Residual Head.
    No local visual branches.
    """

    def __init__(
        self,
        base_model: DeepLocalJointRepresentationClassifier,
        token_dim: int = 32,
        num_classes: int = 10,
        base_seed: int = 240494961,
    ) -> None:
        super().__init__(base_model)
        self.gate = FactorAblationGate(
            source_names=ORIGINAL_10_SOURCES,
            token_dim=token_dim,
            num_classes=num_classes,
            learned_gate=True,
            base_seed=base_seed,
        )

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        call_inputs = self._prepare_call_inputs(*args, **kwargs)

        with torch.no_grad():
            out = self.base_model(**call_inputs)
            base_logits = out.behavior_logits

        device = base_logits.device
        dtype = base_logits.dtype
        B = base_logits.shape[0]

        mb = out.m2_branches if hasattr(out, "m2_branches") else out.modulated_branches
        has_posture = (
            out.posture_logits is not None and out.posture_logits.numel() > 0
        )
        posture_logits = (
            out.posture_logits if has_posture else torch.zeros(B, 3, device=device, dtype=dtype)
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
        }

        delta_logits, weights = self.gate(source_dict, mask_dict)
        final_logits = base_logits + delta_logits

        return final_logits, base_logits, weights


class NearFinalModelL(BaseAblationModel):
    """Config L: Local-Spatial Only with Fixed Neutral Source Weighting.

    Model A + 12 Sources (ActorLocal + UnionLocal) + Fixed Neutral Gate + Residual Head.
    Gate score net permanently frozen at zero.
    """

    def __init__(
        self,
        base_model: DeepLocalJointRepresentationClassifier,
        token_dim: int = 32,
        num_classes: int = 10,
        base_seed: int = 240494961,
    ) -> None:
        super().__init__(base_model)
        self.actor_spatial_attn = build_deterministic_lsa(
            name="actor", in_channels=256, token_dim=token_dim, base_seed=base_seed
        )
        self.union_spatial_attn = build_deterministic_lsa(
            name="union", in_channels=256, token_dim=token_dim, base_seed=base_seed
        )
        self.gate = FactorAblationGate(
            source_names=FULL_12_SOURCES,
            token_dim=token_dim,
            num_classes=num_classes,
            learned_gate=False,
            base_seed=base_seed,
        )

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        call_inputs = self._prepare_call_inputs(*args, **kwargs)

        with torch.no_grad():
            out = self.base_model(**call_inputs)
            base_logits = out.behavior_logits

        device = base_logits.device
        dtype = base_logits.dtype
        B = base_logits.shape[0]

        # Extract Layer 3 spatial feature maps
        actor_map = getattr(
            self.base_model, "_latest_spatial_maps", {}
        ).get("actor", {}).get("layer3")
        union_map = getattr(
            self.base_model, "_latest_spatial_maps", {}
        ).get("union", {}).get("layer3")
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

        t_mask = call_inputs.get("length_mask", call_inputs.get("temporal_mask"))
        if t_mask is None:
            t_mask = torch.ones(B, 6, dtype=torch.bool, device=device)
        elif t_mask.dtype != torch.bool:
            t_mask = t_mask > 0
        T = t_mask.shape[1]

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

        mb = out.m2_branches if hasattr(out, "m2_branches") else out.modulated_branches
        has_posture = (
            out.posture_logits is not None and out.posture_logits.numel() > 0
        )
        posture_logits = (
            out.posture_logits if has_posture else torch.zeros(B, 3, device=device, dtype=dtype)
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


class NearFinalModelJ_Factorial(BaseAblationModel):
    """Config J: Full Model J with deterministic submodule initialization.

    Model A + 12 Sources (ActorLocal + UnionLocal) + Learned Gate + Residual Head.
    """

    def __init__(
        self,
        base_model: DeepLocalJointRepresentationClassifier,
        token_dim: int = 32,
        num_classes: int = 10,
        base_seed: int = 240494961,
    ) -> None:
        super().__init__(base_model)
        self.actor_spatial_attn = build_deterministic_lsa(
            name="actor", in_channels=256, token_dim=token_dim, base_seed=base_seed
        )
        self.union_spatial_attn = build_deterministic_lsa(
            name="union", in_channels=256, token_dim=token_dim, base_seed=base_seed
        )
        self.gate = FactorAblationGate(
            source_names=FULL_12_SOURCES,
            token_dim=token_dim,
            num_classes=num_classes,
            learned_gate=True,
            base_seed=base_seed,
        )

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        call_inputs = self._prepare_call_inputs(*args, **kwargs)

        with torch.no_grad():
            out = self.base_model(**call_inputs)
            base_logits = out.behavior_logits

        device = base_logits.device
        dtype = base_logits.dtype
        B = base_logits.shape[0]

        actor_map = getattr(
            self.base_model, "_latest_spatial_maps", {}
        ).get("actor", {}).get("layer3")
        union_map = getattr(
            self.base_model, "_latest_spatial_maps", {}
        ).get("union", {}).get("layer3")
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

        t_mask = call_inputs.get("length_mask", call_inputs.get("temporal_mask"))
        if t_mask is None:
            t_mask = torch.ones(B, 6, dtype=torch.bool, device=device)
        elif t_mask.dtype != torch.bool:
            t_mask = t_mask > 0
        T = t_mask.shape[1]

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

        mb = out.m2_branches if hasattr(out, "m2_branches") else out.modulated_branches
        has_posture = (
            out.posture_logits is not None and out.posture_logits.numel() > 0
        )
        posture_logits = (
            out.posture_logits if has_posture else torch.zeros(B, 3, device=device, dtype=dtype)
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
    "ORIGINAL_10_SOURCES",
    "FULL_12_SOURCES",
    "build_deterministic_source_projections",
    "build_deterministic_lsa",
    "build_deterministic_score_net",
    "compute_module_subhashes",
    "FactorAblationGate",
    "NearFinalModelN",
    "NearFinalModelG",
    "NearFinalModelL",
    "NearFinalModelJ_Factorial",
]
