# Design Assumptions

This log records design choices that are not fixed by the assignment handout.
It will be updated before implementing a choice and will distinguish assumptions
from measured results.

| ID | Assumption | Justification | Status |
| --- | --- | --- | --- |
| A-01 | The current workspace root is the repository root, even though its folder name is `AI_ENG A1` rather than `transformer-gpt-assignment`. | The user requested work “in this folder”; all required paths are therefore created directly beneath it. The repository layout itself matches the handout. | Active |
| A-02 | Python modules use `src` as an importable package, so `src/__init__.py` is included. | This makes the required `python -m src.train` and `python -m src.generate` commands reliable. | Active |
| A-03 | Step 0 contains only placeholders and project metadata; it does not claim any experimental result or implement an ML milestone. | The user explicitly requires approval before M1. | Active |
| A-04 | The report will be authored in Markdown first and exported to PDF only after evidence exists. | This avoids inventing results and supports the requested Medium-article-ready format. | Active |
| A-05 | Dataset download, model-size choice, hardware record, deployment URL, and final measurements are deferred until their relevant milestones. | The handout requires measured, reproducible evidence; choosing values now would be unsupported. | Active |
| A-06 | Training and ablations will run locally on the student’s Windows laptop using CUDA: PyTorch `2.6.0+cu124` and an NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM. | The student confirmed this setup and `torch.cuda.is_available()` is now true. M4 will re-check this in the training environment and record the exact device name. | Confirmed |
| A-07 | DataLoaders will use `num_workers=0` on Windows. | This is the conservative, reproducible Windows setting and avoids multiprocessing/spawn issues in a student project. | Active |
| A-08 | Mixed precision will use bfloat16 or float16 only after a short stability check; otherwise training will use float32. | Numerical stability and correct learning are more important than speed. The chosen precision and evidence will be recorded in M4/M5. | Active |
| A-09 | Any later attention-weight visualization will set the loaded model to `eval()` before extracting weights. | Attention dropout is disabled in evaluation mode, so displayed weights are normalized probabilities rather than post-dropout training values. | Active |
| A-10 | Training windows use stride 32 while validation windows use non-overlapping stride equal to context length (128). | Overlap increases diverse training contexts; non-overlapping validation reduces near-duplicate evaluation windows. Both splits remain positional and never cross the 90/10 boundary. | Active |
| A-11 | The M5 full run uses learning rate `1e-3` rather than the initial `3e-4` suggestion. | In the matched 300-step, seed-1337 check, `1e-3` achieved validation loss 2.515353 versus 3.242578 at `3e-4`; this is recorded in `outputs/logs/m5_lr_tuning.txt`. | Confirmed by measurement |
| A-12 | Inference and fixed samples use the best-validation checkpoint, exported as `checkpoints/minigpt_final.pt`. | The lowest recorded validation loss was 1.491856 at step 21,000; later validation loss was higher while training loss continued downward. Selecting best validation weights is an evidence-based checkpoint-selection rule. | Confirmed by measurement |

