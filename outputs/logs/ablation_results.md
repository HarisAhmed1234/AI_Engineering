# M6 controlled ablation results

Each primary run used seed 1337, 8,000 steps, lr=1e-3, and identical non-varied settings.
The `noise_baseline` row is a separate seed-2024 rerun of the 4L/4H/128 baseline. Its validation-loss difference from the seed-1337 baseline is 0.005491, a rough two-seed noise estimate.
Timing is synchronized CUDA wall time but remains indicative because laptop GPU thermal state can vary.

| Study | Setting | Seed | Val loss | Perplexity | Parameters | Time (s) | Tokens/s | Peak GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| depth | 2 | 1337 | 1.592932 | 4.918146 | 413,440 | 81.395 | 805162.80 | 190.21 |
| depth | 4 | 1337 | 1.497624 | 4.471055 | 809,984 | 141.828 | 462079.78 | 335.78 |
| depth | 6 | 1337 | 1.474871 | 4.370474 | 1,206,528 | 201.429 | 325354.56 | 481.36 |
| heads | 2 | 1337 | 1.513320 | 4.541787 | 809,984 | 123.061 | 532548.73 | 274.93 |
| heads | 4 | 1337 | 1.497624 | 4.471055 | 809,984 | 143.113 | 457933.25 | 335.78 |
| heads | 8 | 1337 | 1.489331 | 4.434127 | 809,984 | 223.110 | 293739.11 | 463.78 |
| context | 64 | 1337 | 1.550919 | 4.715803 | 809,984 | 123.970 | 264321.86 | 154.67 |
| context | 128 | 1337 | 1.497624 | 4.471055 | 809,984 | 141.940 | 461715.70 | 335.78 |
| context | 256 | 1337 | 1.467529 | 4.338500 | 809,984 | 455.030 | 288051.65 | 898.69 |
| noise_baseline | 4L/4H/128 | 2024 | 1.492133 | 4.446568 | 809,984 | 143.828 | 455656.01 | 335.78 |
