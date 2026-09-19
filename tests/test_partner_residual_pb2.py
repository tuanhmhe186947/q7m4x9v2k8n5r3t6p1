import torch

from pig_behavior.classification_v2.models.partner_residual import (
    PartnerBehaviorResidualConfig,
    PartnerBehaviorResidualHead,
    build_pb2_partner_tokens,
)


def test_pb1_partner_residual_contract():
    config = PartnerBehaviorResidualConfig(
        num_classes=10,
        k=2,
        partner_input_dim=10,
        partner_embedding_dim=32,
        dropout=0.0,
    )
    head = PartnerBehaviorResidualHead(config)
    
    total_params = sum(p.numel() for p in head.parameters())
    assert total_params == 746
    
    partner_probs = torch.rand(4, 2, 10)
    partner_mask = torch.tensor([[True, True], [True, False], [False, True], [False, False]])
    
    res = head(partner_probs, partner_mask)
    assert res.shape == (4, 10)
    assert torch.allclose(res, torch.zeros_like(res))
    
    all_false_mask = torch.zeros(4, 2, dtype=torch.bool)
    res_zero = head(partner_probs, all_false_mask)
    assert torch.equal(res_zero, torch.zeros_like(res_zero))


def test_pb2_partner_residual_contract():
    config = PartnerBehaviorResidualConfig(
        num_classes=10,
        k=2,
        partner_input_dim=20,
        partner_embedding_dim=32,
        dropout=0.0,
    )
    head = PartnerBehaviorResidualHead(config)
    
    total_params = sum(p.numel() for p in head.parameters())
    assert total_params == 1066
    
    actor_logits = torch.randn(4, 10)
    partner_probs = torch.rand(4, 2, 10)
    partner_mask = torch.tensor([[True, True], [True, False], [False, True], [False, False]])
    
    pb2_tokens = build_pb2_partner_tokens(partner_probs, actor_logits)
    assert pb2_tokens.shape == (4, 2, 20)
    
    res = head(pb2_tokens, partner_mask)
    assert res.shape == (4, 10)
    assert torch.allclose(res, torch.zeros_like(res))
    
    all_false_mask = torch.zeros(4, 2, dtype=torch.bool)
    res_zero = head(pb2_tokens, all_false_mask)
    assert torch.equal(res_zero, torch.zeros_like(res_zero))
