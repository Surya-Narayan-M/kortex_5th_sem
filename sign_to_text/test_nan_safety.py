"""Test NaN safety fixes"""
import torch
import sys
import numpy as np
sys.path.insert(0, '.')

def test_label_smoothing():
    print('\n[Test 1] LabelSmoothingLoss with edge cases...')
    from train_hybrid import LabelSmoothingLoss
    loss_fn = LabelSmoothingLoss(vocab_size=75, smoothing=0.1, padding_idx=0)

    # Normal case
    logits = torch.randn(2, 10, 75)
    targets = torch.randint(1, 75, (2, 10))
    loss = loss_fn(logits, targets)
    print(f'  Normal case loss: {loss.item():.4f} (should be finite)')
    assert torch.isfinite(loss), 'Normal case failed!'

    # All padding case
    targets_pad = torch.zeros(2, 10, dtype=torch.long)
    loss_pad = loss_fn(logits, targets_pad)
    print(f'  All padding loss: {loss_pad.item():.4f} (should be 0.0)')
    assert torch.isfinite(loss_pad), 'Padding case failed!'

    # Extreme logits
    logits_extreme = torch.randn(2, 10, 75) * 1000
    loss_extreme = loss_fn(logits_extreme, targets)
    print(f'  Extreme logits loss: {loss_extreme.item():.4f} (should be finite)')
    assert torch.isfinite(loss_extreme), 'Extreme logits case failed!'
    print('  [PASS] LabelSmoothingLoss')

def test_hybrid_loss():
    print('\n[Test 2] HybridLoss with CTC length validation...')
    from train_hybrid import HybridLoss
    hybrid_loss = HybridLoss(vocab_size=75, blank_idx=0)

    # Create test data where enc_length < ctc_target_length (should trigger fallback)
    batch_size = 4
    ctc_log_probs = torch.randn(batch_size, 10, 75).log_softmax(dim=-1)  # 10 frames
    gru_logits = torch.randn(batch_size, 20, 75)
    transformer_logits = torch.randn(batch_size, 20, 75)
    ctc_targets = torch.randint(1, 75, (batch_size, 15))  # 15 tokens - longer than 10 frames!
    ctc_target_lens = torch.tensor([15, 12, 8, 5])  # Mixed: 2 invalid, 2 valid
    attn_targets = torch.randint(1, 75, (batch_size, 20))
    src_lens = torch.tensor([10, 10, 10, 10])  # All 10 frames

    total, ctc_l, gru_l, tf_l = hybrid_loss(
        ctc_log_probs, gru_logits, transformer_logits,
        ctc_targets, ctc_target_lens, attn_targets, src_lens, ctc_weight=0.3
    )

    print(f'  Total loss: {total.item():.4f}')
    print(f'  CTC loss: {ctc_l.item():.4f}')
    print(f'  GRU loss: {gru_l.item():.4f}')
    print(f'  TF loss: {tf_l.item():.4f}')
    assert torch.isfinite(total), 'HybridLoss produced NaN!'
    print('  [PASS] HybridLoss handles short sequences')

def test_ctc_head():
    print('\n[Test 3] CTCHead NaN safety...')
    from model_hybrid_v2 import CTCHead
    ctc_head = CTCHead(hidden_dim=384, vocab_size=75)

    # Normal input
    x_normal = torch.randn(2, 50, 384)
    out_normal = ctc_head(x_normal)
    print(f'  Normal input: min={out_normal.min().item():.2f}, max={out_normal.max().item():.2f}')
    assert torch.isfinite(out_normal).all(), 'Normal input failed!'
    assert out_normal.min() >= -100, 'Log probs not clamped!'

    # Extreme input
    x_extreme = torch.randn(2, 50, 384) * 1000
    out_extreme = ctc_head(x_extreme)
    print(f'  Extreme input: min={out_extreme.min().item():.2f}, max={out_extreme.max().item():.2f}')
    assert torch.isfinite(out_extreme).all(), 'Extreme input failed!'
    print('  [PASS] CTCHead')

def test_full_training():
    print('\n[Test 4] Full training iteration...')
    from train_hybrid import TrainConfig, Trainer

    config = TrainConfig()
    config.epochs = 1
    # Use single worker for testing on Windows
    config.num_workers = 0
    trainer = Trainer(config)

    # Get a batch and run through training
    batch = next(iter(trainer.train_loader))
    src = batch['src'].to(trainer.device)
    tgt = batch['tgt'].to(trainer.device)
    ctc_tgt = batch['ctc_tgt'].to(trainer.device)
    src_lens = batch['src_lens'].to(trainer.device)
    ctc_lens = batch['ctc_lens']

    print(f'  Batch src shape: {src.shape}')
    print(f'  Batch tgt shape: {tgt.shape}')

    # Forward pass
    trainer.model.train()
    outputs = trainer.model(src, src_lens, tgt, tf_ratio=0.5, train_both_decoders=True)

    ctc_log_probs = outputs['ctc_log_probs']
    gru_logits = outputs['gru_outputs']
    transformer_logits = outputs['transformer_outputs']
    enc_lengths = outputs['enc_lengths']

    print(f'  CTC log probs shape: {ctc_log_probs.shape}')
    print(f'  Enc lengths (first 4): {enc_lengths[:4].tolist()}')
    print(f'  CTC target lens (first 4): {ctc_lens[:4].tolist()}')

    # Check if any samples have enc_length < ctc_length
    valid_mask = enc_lengths.cpu() >= torch.tensor(ctc_lens)
    valid_pct = valid_mask.float().mean() * 100
    print(f'  Valid CTC samples in batch: {valid_pct:.1f}%')

    # Compute loss
    loss, ctc_l, gru_l, tf_l = trainer.criterion(
        ctc_log_probs, gru_logits, transformer_logits,
        ctc_tgt, ctc_lens, tgt, enc_lengths, 0.3
    )

    print(f'  Loss: {loss.item():.4f} (should be finite)')
    print(f'  CTC: {ctc_l.item():.4f}, GRU: {gru_l.item():.4f}, TF: {tf_l.item():.4f}')
    assert torch.isfinite(loss), 'Training iteration failed!'

    # Test backward pass
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
    print(f'  Gradient norm: {grad_norm.item():.4f} (should be finite)')
    assert torch.isfinite(grad_norm), 'Gradient explosion!'

    print('  [PASS] Full training iteration')


if __name__ == '__main__':
    print('='*60)
    print('Testing NaN-safe training components')
    print('='*60)
    
    test_label_smoothing()
    test_hybrid_loss()
    test_ctc_head()
    test_full_training()
    
    print('\n' + '='*60)
    print('All tests passed! Training should be stable now.')
    print('='*60)
