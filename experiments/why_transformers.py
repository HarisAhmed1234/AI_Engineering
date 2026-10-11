"""Compare recurrent and self-attention sequence processing for M1.

The complexity table is an architectural analysis.  The timing/memory values are
measured when this module runs; they are never hard-coded into the report.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from src.attention import SelfAttention


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "logs" / "m1_rnn_vs_attention.csv"
FIGURE_PATH = ROOT / "outputs" / "figures" / "fig_m1_rnn_vs_attention.png"


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward(module: nn.Module, x: torch.Tensor) -> None:
    """Run either the RNN baseline or the project attention layer."""
    if isinstance(module, SelfAttention):
        module(x, causal=True)
    else:
        module(x)


def measure(module: nn.Module, x: torch.Tensor, device: torch.device, repeats: int = 12) -> tuple[float, float, str]:
    """Return median time, memory measurement, and its accurate measurement label."""
    module.eval()
    with torch.inference_mode():
        for _ in range(3):
            _forward(module, x)
        synchronize_if_cuda(device)
        timings = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(repeats):
            synchronize_if_cuda(device)
            start = time.perf_counter()
            _forward(module, x)
            synchronize_if_cuda(device)
            timings.append((time.perf_counter() - start) * 1000)
        if device.type == "cuda":
            memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
            memory_label = "peak CUDA allocated MiB"
        else:
            # CPU has no torch allocator peak statistic. The profiler provides a
            # real per-forward allocation measurement, labelled honestly below.
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as profiler:
                _forward(module, x)
            allocated_bytes = sum(
                max(event.self_cpu_memory_usage, 0) for event in profiler.key_averages()
            )
            memory_mib = allocated_bytes / (1024**2)
            memory_label = "CPU-profiler allocated MiB per forward"
    return float(torch.tensor(timings).median().item()), memory_mib, memory_label


def main() -> None:
    torch.manual_seed(7)
    batch_size, d_model = 8, 64
    lengths = [32, 64, 128, 256]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    print("WHY TRANSFORMERS: ARCHITECTURAL COMPARISON")
    print("\n| Layer | Per-layer complexity | Sequential operations | Maximum path length |")
    print("| --- | --- | --- | --- |")
    print("| Simple RNN | O(n * d^2) | O(n) | O(n) |")
    print("| Self-attention | O(n^2 * d + n * d^2) | O(1) | O(1) |")
    print("\nInterpretation: an RNN must finish token t before computing t+1. A self-attention layer forms all query-key scores in parallel, but stores a T x T score matrix, so its memory grows quadratically with sequence length.")
    print("\nTiming note: short GPU timings are noisy and affected by device state, clocks, and libraries. "
          "CUDA synchronization is performed before and after every timed forward pass.")

    rows: list[dict[str, float | int | str]] = []
    for device in devices:
        device_label = "CPU" if device.type == "cpu" else f"CUDA ({torch.cuda.get_device_name(device)})"
        print(f"\nMeasured forward-pass benchmark on {device_label} (batch={batch_size}, d_model={d_model}, median of 12 runs):")
        for length in lengths:
            x = torch.randn(batch_size, length, d_model, device=device)
            rnn = nn.RNN(d_model, d_model, batch_first=True).to(device)
            attention = SelfAttention(d_model=d_model, dropout=0.0).to(device)
            rnn_ms, rnn_memory, memory_label = measure(rnn, x, device)
            attention_ms, attention_memory, attention_memory_label = measure(attention, x, device)
            assert memory_label == attention_memory_label
            rows.extend(
                [
                    {"device": device.type, "layer": "RNN", "sequence_length": length, "time_ms": rnn_ms, "memory_mib": rnn_memory, "memory_measurement": memory_label},
                    {"device": device.type, "layer": "SelfAttention", "sequence_length": length, "time_ms": attention_ms, "memory_mib": attention_memory, "memory_measurement": memory_label},
                ]
            )
            print(
                f"  T={length:3d} | RNN {rnn_ms:8.3f} ms | attention {attention_ms:8.3f} ms"
                f" | {memory_label}: RNN {rnn_memory:.3f}, attention {attention_memory:.3f}"
            )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["device", "layer", "sequence_length", "time_ms", "memory_mib", "memory_measurement"])
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, len(devices), figsize=(7 * len(devices), 9), squeeze=False)
    for column, device in enumerate(devices):
        device_rows = [row for row in rows if row["device"] == device.type]
        memory_label = str(device_rows[0]["memory_measurement"])
        for layer in ("RNN", "SelfAttention"):
            layer_rows = [row for row in device_rows if row["layer"] == layer]
            lengths_for_layer = [row["sequence_length"] for row in layer_rows]
            axes[0, column].plot(lengths_for_layer, [row["time_ms"] for row in layer_rows], marker="o", label=layer)
            axes[1, column].plot(lengths_for_layer, [float(row["memory_mib"]) for row in layer_rows], marker="o", label=layer)
        device_name = "CPU" if device.type == "cpu" else "CUDA RTX 4050"
        axes[0, column].set_title(f"{device_name}: forward timing (noisy)")
        axes[0, column].set_ylabel("Median milliseconds")
        axes[1, column].set_title(f"{device_name}: memory")
        axes[1, column].set_ylabel(memory_label)
        for axis in axes[:, column]:
            axis.set_xlabel("Sequence length")
            axis.legend()
            axis.grid(alpha=0.3)
    figure.suptitle("M1 - RNN versus manually implemented self-attention\nTiming is noisy; CUDA timings are synchronized")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"\nSaved measurements: {CSV_PATH.relative_to(ROOT)}")
    print(f"Saved comparison figure: {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

