"""M4 debugging gate: a tiny batch must overfit before any long run."""

import math
import time

import torch
import torch.nn.functional as functional

from src.gpt import MiniGPT


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_tiny_repeated_batch_overfits_below_twenty_percent_of_initial_loss() -> None:
    torch.manual_seed(123)
    device = _device()
    model = MiniGPT(vocab_size=8, context_length=8, d_model=32, n_heads=4, n_layers=1, ffn_dim=128, dropout=0.0).to(device)
    # The same cyclic window is deliberately repeated: this is a gradient/debugging
    # gate, not a language-quality test.
    x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]] * 8, device=device)
    y = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]] * 8, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    started = time.perf_counter()
    model.train()
    initial = None
    for _ in range(250):
        logits = model(x)
        loss = functional.cross_entropy(logits.reshape(-1, 8), y.reshape(-1))
        if initial is None:
            initial = float(loss.detach())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    final = float(loss.detach())
    print(f"overfit device={device.type} initial_loss={initial:.6f} final_loss={final:.6f} ratio={final / initial:.4f} elapsed_s={time.perf_counter() - started:.2f}")
    assert initial is not None
    assert final < 0.20 * initial


def test_initial_loss_is_near_log_vocab_size() -> None:
    torch.manual_seed(55)
    vocab_size = 17
    model = MiniGPT(vocab_size=vocab_size, context_length=6, d_model=24, n_heads=4, n_layers=1, dropout=0.0).eval()
    ids = torch.randint(0, vocab_size, (64, 6))
    targets = torch.randint(0, vocab_size, (64, 6))
    loss = functional.cross_entropy(model(ids).reshape(-1, vocab_size), targets.reshape(-1))
    assert abs(float(loss) - math.log(vocab_size)) < 0.35


def test_full_gpt_is_causal_for_earlier_logits() -> None:
    torch.manual_seed(77)
    model = MiniGPT(vocab_size=19, context_length=7, d_model=24, n_heads=4, n_layers=2, dropout=0.0).eval()
    original = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    changed = original.clone(); changed[:, 4:] = torch.tensor([[15, 16, 17]])
    original_logits = model(original)
    changed_logits = model(changed)
    assert torch.allclose(original_logits[:, :4], changed_logits[:, :4], atol=1e-6)

