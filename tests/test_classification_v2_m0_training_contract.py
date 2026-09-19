import hashlib
import json
from pathlib import Path


def test_m0_full_t6_scientific_training_contract_schema_and_derivation():
    contract_path = (
        Path('configs/classification_v2')
        / 'm0_full_t6_scientific_training_contract_v1.json'
    )
    assert contract_path.exists(), f'Contract file not found: {contract_path}'

    data = json.loads(contract_path.read_text(encoding='utf-8'))
    assert data['contract_name'] == 'm0_full_t6_scientific_training_contract'
    assert data['version'] == 'v1'
    assert data['campaign_id'] == 'm0_full_t6_r128_v1'

    # Model
    assert data['model']['architecture'] == 'FullMultimodal-R34-T6-Concat'
    assert data['model']['backbone_name'] == 'resnet34'
    assert data['model']['image_size'] == 128
    assert data['model']['temporal_input_frames'] == 6
    assert data['model']['code_sha'] == 'a35428c2'

    # Dataset
    assert data['dataset']['population']['total_windows'] == 33287
    assert data['dataset']['population']['train_windows'] == 27834
    assert data['dataset']['population']['validation_windows'] == 5453
    assert data['dataset']['spatial_features'] == 'canonical_46d'

    # Execution Profile
    assert data['execution_profile']['microbatch_size'] == 128
    assert data['execution_profile']['gradient_accumulation_steps'] == 1
    assert data['execution_profile']['effective_batch_size'] == 128
    assert data['execution_profile']['amp_precision'] == 'fp16'
    assert data['execution_profile']['num_workers'] == 0
    assert data['execution_profile']['pin_memory'] is True

    # Optimization
    assert data['optimization']['optimizer'] == 'adamw'
    assert data['optimization']['learning_rate'] == 0.003
    assert data['optimization']['lr_batch_compatibility'] == 'RESOLVED'

    # Checkpoint Metric and Early Stopping
    assert (
        data['evaluation_and_checkpointing']['primary_metric']
        == 'primary_native_macro_f1'
    )
    assert (
        data['evaluation_and_checkpointing']['early_stopping_metric']
        == 'validation_native_unit_macro_f1_supported'
    )
    assert data['evaluation_and_checkpointing']['early_stopping_patience'] == 5
    assert (
        data['evaluation_and_checkpointing']['validation_frequency']
        == 'once_per_completed_epoch'
    )

    # Seed derivation
    campaign_id = data['campaign_id']
    expected_seeds = [
        int.from_bytes(
            hashlib.sha256(f'{campaign_id}/seed/{i}'.encode()).digest()[:4],
            'big',
        )
        % (2**31 - 1)
        for i in range(3)
    ]
    assert data['scientific_seeds']['seeds'] == expected_seeds
    assert data['scientific_seeds']['seeds'] == [240494961, 382529781, 166101551]
    assert data['scientific_seeds']['historical_s1_seeds_reused'] is False

    historical_s1_seeds = {20260804, 20260805, 20260806}
    assert not historical_s1_seeds.intersection(set(data['scientific_seeds']['seeds']))

    # Budget resolved
    assert data['budget']['status'] == 'RESOLVED'
    assert data['budget']['max_epochs'] == 30
    assert data['budget']['drop_last'] is False
    assert data['budget']['train_batches_per_epoch'] == 218
    assert data['budget']['optimizer_steps_per_epoch'] == 218
    assert data['budget']['max_optimizer_steps'] == 6540
    assert data['budget']['historical_s1_budget_reused'] is False
    assert data['unresolved_fields'] == []
