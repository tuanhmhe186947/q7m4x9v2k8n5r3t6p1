from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pig_behavior.classification_v2.features.spatial_schema import (
    load_current_spatial_tensor_bundle,
)
from pig_behavior.classification_v2.models.spatial_tcn import SpatialTCNClassifier, SpatialTCNConfig
from pig_behavior.classification_v2.training.spatial_tcn_smoke import MODEL_GROUPS

DEFAULT_ROOT = Path("outputs/classification_v2/train_ready_windows")
DEFAULT_OUTPUT = Path("outputs/classification_v2/model_smoke/spatial_tcn_overfit_smoke.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit-one-batch smoke test for classification_v2 SpatialTCN.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    y = pd.read_csv(args.root / "y_behavior.csv").iloc[:, 0].fillna("").astype(str)
    train_mask = _read_bool(args.root / "train_mask.csv")
    split = pd.read_csv(args.root / "split_manifest.csv", low_memory=False)
    data, _ = load_current_spatial_tensor_bundle(
        args.root / "X_spatial_sequences.npz",
        args.root / "spatial_sequence_audit.json",
    )
    label_order = sorted(y.unique().tolist())
    label_to_idx = {label: i for i, label in enumerate(label_order)}
    selected = _select_balanced_indices(
        y,
        train_mask,
        split,
        per_class=args.per_class,
        batch_size=args.batch_size,
    )
    if len(selected) < 2:
        raise ValueError("Could not select enough rows for overfit smoke batch")

    features = {name: torch.from_numpy(data[name][selected]).float() for name in MODEL_GROUPS}
    length_mask = torch.from_numpy(data["length_mask"][selected]).float()
    observed_mask = torch.from_numpy(data["observed_mask"][selected]).float()
    target = torch.tensor(
        [label_to_idx[label] for label in y.iloc[selected].tolist()],
        dtype=torch.long,
    )

    model = SpatialTCNClassifier(
        SpatialTCNConfig(
            input_dims={name: int(features[name].shape[-1]) for name in MODEL_GROUPS},
            num_classes=len(label_order),
            hidden_dim=args.hidden_dim,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    losses: list[float] = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, length_mask=length_mask, observed_mask=observed_mask)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    model.eval()
    with torch.no_grad():
        logits_before_save = model(features, length_mask=length_mask, observed_mask=observed_mask)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_json.with_suffix(".pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_order": label_order,
            "config": {
                "input_dims": {name: int(features[name].shape[-1]) for name in MODEL_GROUPS},
                "num_classes": len(label_order),
                "hidden_dim": args.hidden_dim,
                "dropout": 0.0,
            },
        },
        checkpoint_path,
    )
    reloaded = SpatialTCNClassifier(
        SpatialTCNConfig(
            input_dims={name: int(features[name].shape[-1]) for name in MODEL_GROUPS},
            num_classes=len(label_order),
            hidden_dim=args.hidden_dim,
            dropout=0.0,
        )
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(state["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        logits_after_reload = reloaded(
            features,
            length_mask=length_mask,
            observed_mask=observed_mask,
        )
    reload_max_delta = float((logits_before_save - logits_after_reload).abs().max().cpu().item())

    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = initial_loss - final_loss
    errors = []
    if not np.isfinite(losses).all():
        errors.append("loss_nonfinite")
    if final_loss >= initial_loss * 0.80:
        errors.append(f"loss_did_not_drop_enough initial={initial_loss:.6f} final={final_loss:.6f}")
    if reload_max_delta > 1e-6:
        errors.append(f"reload_parity_failed_delta={reload_max_delta}")

    audit = {
        "root": str(args.root),
        "output_json": str(args.output_json),
        "checkpoint_path": str(checkpoint_path),
        "batch_size": int(len(selected)),
        "steps": int(args.steps),
        "hidden_dim": int(args.hidden_dim),
        "lr": float(args.lr),
        "label_order": label_order,
        "batch_label_counts": y.iloc[selected].value_counts(dropna=False).to_dict(),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "reload_max_delta": reload_max_delta,
        "errors": errors,
    }
    args.output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(2)


def _select_balanced_indices(
    y: pd.Series,
    train_mask: pd.Series,
    split: pd.DataFrame,
    *,
    per_class: int,
    batch_size: int,
) -> np.ndarray:
    valid_train = split["split"].astype(str).eq("train") & train_mask
    selected: list[int] = []
    for label in sorted(y.unique()):
        label_indices = np.flatnonzero((valid_train & y.eq(label)).to_numpy())
        selected.extend(label_indices[:per_class].tolist())
    if len(selected) > batch_size:
        selected = selected[:batch_size]
    return np.array(selected, dtype=np.int64)


def _read_bool(path: Path) -> pd.Series:
    series = pd.read_csv(path).iloc[:, 0]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


if __name__ == "__main__":
    main()
