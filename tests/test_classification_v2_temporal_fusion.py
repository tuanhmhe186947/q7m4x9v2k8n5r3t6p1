import torch

from pig_behavior.classification_v2.models.multimodal_fusion import MaskedTemporalConvEncoder


def test_temporal_encoder_ignores_values_in_padded_slots() -> None:
    """Invalid sequence positions must not affect valid temporal embeddings."""

    torch.manual_seed(4)
    encoder = MaskedTemporalConvEncoder(embedding_dim=4, layers=2, dropout=0.0).eval()
    value = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
    changed = value.clone()
    changed[mask.eq(0)] = 10_000.0

    original_embedding = encoder(value, mask)
    changed_embedding = encoder(changed, mask)

    torch.testing.assert_close(original_embedding, changed_embedding)


def test_temporal_encoder_is_sensitive_to_valid_frame_order() -> None:
    """Temporal fusion must distinguish a sequence from its reversed dynamics."""

    torch.manual_seed(7)
    encoder = MaskedTemporalConvEncoder(embedding_dim=4, layers=2, dropout=0.0).eval()
    value = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [2.0, 0.0, 1.0, 4.0], [5.0, 1.0, 0.0, 2.0], [1.0, 3.0, 4.0, 0.0]]]
    )
    mask = torch.ones(1, 4)

    forward = encoder(value, mask)
    reversed_sequence = encoder(value.flip(1), mask)

    assert not torch.allclose(forward, reversed_sequence, atol=1e-6, rtol=1e-6)
