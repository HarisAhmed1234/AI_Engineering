# MiniGPT from Scratch: a Tiny Shakespeare Causal Language Model

> **Experimental-output notice:** This model's generated text is an experimental
> artifact. It may be incoherent or inaccurate and must not be treated as factual
> information.

## Overview

This project implements a small GPT-style causal language model in Python and
PyTorch, without pretrained language-model weights or high-level Transformer model
classes. It trains a 809,984-parameter character-level model on Tiny Shakespeare,
then exposes reproducible CLI and local FastAPI inference. The goal is
understandability: Q/K/V attention, causal masking, multi-head split/merge,
positional encoding, pre-LayerNorm Transformer blocks, shifted next-token targets,
and sampling are implemented in project code.

## Repository map

| Location | Contents |
| --- | --- |
| `src/` | Core model: tokenizer, dataset windows, manual attention, positions, blocks, MiniGPT, training, generation, utilities. |
| `data/` | Dataset documentation and `download_tiny_shakespeare.py`; the downloaded `input.txt` is ignored. |
| `configs/tiny_gpt.yaml` | Reproducible training configuration. |
| `tests/` | Attention, mask, shape, dataset-shift, overfit, and API route tests. |
| `experiments/` | Hand calculation, Transformer comparison, MLM/CLM, evaluation, ablations, attention visualization, and figure rendering. |
| `outputs/` | Locally generated figures, logs, and text samples; ignored from Git. |
| `checkpoints/` | Local training checkpoints. Only the small weights-only `minigpt_final.pt` may be deliberately released. |
| `app/` | FastAPI inference service. |
| `notebooks/colab_demo.ipynb` | Colab walkthrough with explicit URL placeholders. |
| `report/report.md` | Medium-article-ready technical report. |

## Setup

Use Python 3.12 (the executed environment used Python 3.12.5). Create and activate a
virtual environment, then install project dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### Windows CUDA note

The measured training run used Windows, PyTorch `2.6.0+cu124`, CUDA 12.4, and an
NVIDIA RTX 4050 Laptop GPU (6 GiB). If CUDA training is desired, first install the
matching CUDA-enabled PyTorch wheel using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then run the requirements commands above. Do not replace a working CUDA Torch installation with a CPU wheel. CPU inference and tests are supported, but full training/ablations will be slower.

Download the data:

```powershell
python data/download_tiny_shakespeare.py
```

## Dataset source and licence

The downloader uses the public Tiny Shakespeare text at
[`karpathy/char-rnn`](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare),
then a Hugging Face mirror only if the primary URL fails. The executed download was
1,155,394 bytes and 40,000 lines. The `char-rnn` repository is MIT-licensed; the
underlying Shakespeare works are public domain. `data/input.txt` is not committed.

Preprocessing uses no filtering, builds a sorted deterministic 65-character
vocabulary, makes a positional 90/10 train/validation split, and uses shifted
fixed-length windows. There are no special tokens because every corpus character is
in the vocabulary and fixed windows require no padding.

## Tests and acceptance checks

Run the complete suite in a normal local Python environment:

```powershell
python -m pytest
```

The current Codex host cannot create the sockets required by FastAPI `TestClient`.
For that host only, the verified model/test suite command is:

```powershell
python -m pytest -q --ignore=tests/test_api.py
```

The cross-platform acceptance scripts install dependencies, run the test/overfit
checks, fresh-load the checkpoint, generate three fixed prompts, test causality, and
scan for secrets:

```powershell
.\scripts\acceptance.ps1
```

```bash
bash scripts/acceptance.sh
```

On the constrained Codex host only, use `-SkipApiTests` (PowerShell) or set
`SKIP_API_TESTS=1` (Bash); a normal clean-environment acceptance run must leave API
tests enabled.

## Training

```powershell
python -m src.train --config configs/tiny_gpt.yaml
```

The configuration uses seed 1337, 128-character context, `d_model=128`, four heads,
four blocks, 512-dimensional FFN, batch size 64, dropout 0.1, AdamW, train stride
32, validation stride 128, 100 warmup steps, cosine decay, gradient clipping 1.0,
and up to 40,000 steps. It auto-selects CUDA/MPS/CPU and tests float16 CUDA AMP
stability before using it. Resuming uses `--resume checkpoints/last.pt`.

## Obtain the trained checkpoint

For a fresh clone, download the weights-only best-validation checkpoint from the
project release and place it here:

```text
https://github.com/HarisAhmed1234/AI_Engineering/releases/download/v1.0/minigpt_final.pt
checkpoints/minigpt_final.pt
```

`minigpt_final.pt` is 3.177 MiB and contains only `model_state`, configuration, and
vocabulary. It was exported from `best.pt` at the lowest validation-loss step,
21,000. Do not commit optimizer/resume checkpoints such as `best.pt` or `last.pt`.

## Inference

```powershell
python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80 --temperature 0.8 --top-k 40 --seed 7 --checkpoint checkpoints/minigpt_final.pt
```

Actual saved CLI output (from `outputs/logs/fig10_cli.txt`) included:

```text
Checkpoint: checkpoints/minigpt_final.pt
Device: cuda
Seed: 7
Model parameters: 809,984
Context length: 128
Temperature: 0.8
Top-k: 40; Top-p: None; Greedy: False
Prompt: The future of artificial intelligence
Generated:
The future of artificial intelligence
Of mine arms, he comes in this as he,
Which is the honour man that I should see
```

This Shakespeare-style continuation is expected: Tiny Shakespeare is the only
training corpus, so the model has no basis for modern factual prose. Other options
include `--top-p`, `--greedy`, and context cropping is automatic.

## Local API and deployment

The FastAPI service exposes `GET /health`, `GET /model-info`, `POST /generate`,
`/docs`, and a small HTML page at `/`. Start locally only where sockets are supported:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Linux/macOS:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"ROMEO:","max_new_tokens":80,"temperature":0.8,"top_k":40}'
```

PowerShell-safe request:

```powershell
@'
{"prompt":"ROMEO:","max_new_tokens":80,"temperature":0.8,"top_k":40}
'@ | Set-Content -NoNewline -Encoding utf8 request.json
curl.exe -X POST http://127.0.0.1:8000/generate `
  -H "Content-Type: application/json" `
  --data-binary "@request.json"
```

The API loads the CPU model once at startup. If it is absent, set `CHECKPOINT_URL`
to a public direct-download URL. The Dockerfile uses CPU Torch; `render.yaml` has
the same variable with no embedded secret. Local HTTP calls were verified; deployment
remains manual:

```text
https://minigpt-api.onrender.com
```

## Colab

Open [the Colab notebook](notebooks/colab_demo.ipynb) after uploading/publishing the
repository. Replace both explicit placeholders in its first configuration cell:

https://colab.research.google.com/drive/1bF2CPcHV34PHYP4P9T-CPHkvKpG9zYR_#scrollTo=I0n1_nZb0hi6

```text
COLAB_NOTEBOOK_URL_PLACEHOLDER
GITHUB_URL_PLACEHOLDER
CHECKPOINT_URL_PLACEHOLDER
```

It clones the repository, keeps Colab's Torch build, installs remaining/runtime-test
dependencies, downloads data and the checkpoint, runs tests, generates three fixed
prompts, and displays the hand calculation, causal-mask evidence, and shape trace.
The optional short training and ablation cells are disabled by default.

## Reproduce experiments

All commands run from the repository root after data is downloaded. The full
training command is above; it produced the recorded M5 results and should not be
re-run merely to view existing figures.

```powershell
python experiments/hand_calculation.py
python experiments/why_transformers.py
python experiments/mlm_vs_clm.py
python experiments/evaluate.py --checkpoint checkpoints/best.pt
python experiments/generate_fixed_samples.py
python experiments/attention_visualization.py --checkpoint checkpoints/minigpt_final.pt
python experiments/ablation.py
```

`ablation.py` uses CUDA and the recorded 8,000-step runs per condition; it is not a
quick smoke test. Existing completed rows are reused unless `--rerun` is supplied.
Figure 10 is rendering-only and reads the saved real-call logs:

```powershell
python -m experiments.fig10_inference
```

## Medium

https://medium.com/@shuzaifaali05/build-a-small-llm-from-scratch-73897648ada1?postPublishedType=initial


## Configuration and measured results

The actual configuration is in `configs/tiny_gpt.yaml` and the full executed copy is
in `outputs/logs/m5_training_summary.json`. The chosen configuration retained the
handout’s 128/4/4 starting shape because it fit the 6 GiB GPU. A matched 300-step
learning-rate check selected `1e-3`: validation loss was 2.515353 versus 3.242578 at
`3e-4`. The run used 40,000 steps; the best-validation weights, rather than final
weights, power inference.

| Measurement | Result |
| --- | ---: |
| Parameters | 809,984 |
| Hardware | RTX 4050 Laptop GPU, 6 GiB; PyTorch 2.6.0+cu124 |
| Precision | float16 CUDA AMP after finite stability probe |
| Wall-clock training time | 1,516.183 s (25 min 16.183 s) |
| Final average train loss | 1.129923 |
| Final validation loss / perplexity | 1.519580 / 4.570307 |
| Best validation loss / step | 1.491856 / 21,000 |
| Best weights-only checkpoint | 3.177 MiB |
| Ablation rough two-seed baseline difference | 0.005491 |

## Known limitations

- The 65-character, 809,984-parameter model imitates local Shakespeare formatting
  but is not reliably semantically coherent; actual samples include malformed forms
  such as `pursuive` and `lready`.
- Its usable generation history is capped at 128 characters because older tokens are
  cropped before each forward pass.
- Greedy and low-temperature decoding showed real repetition, including “the
  strength of the state” for one `ROMEO:` continuation.
- The best validation score selects among the same validation windows used while
  training, so it is slightly optimistic compared with an independent test set.
- Head-count loss differences in the 8,000-step study were not reliable with only a
  two-seed rough noise estimate; do not treat them as a quality ranking.

## Citations and adaptations

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762):
  conceptual source for scaled attention, sinusoidal positions, and Transformer
  terminology. The code here is a small independent PyTorch implementation.
- Radford et al., [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language-models.pdf):
  conceptual reference for decoder-only causal language modeling; no GPT-2 code or
  weights were used.
- Karpathy, [char-rnn / Tiny Shakespeare](https://github.com/karpathy/char-rnn):
  raw corpus download source; no model code was adapted.
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/): API reference for
  allowed tensor, autograd, CUDA AMP, optimizer, and data-loader primitives.
- [FastAPI documentation](https://fastapi.tiangolo.com/tutorial/): API-routing and
  request-validation reference.
- Assignment curriculum lessons 01–07: conceptual framing for the required
  attention, positional, Transformer, MLM, and causal-LM milestones; no external
  implementation was copied.

## No-secrets and artifact policy

`.gitignore` excludes `.env`, the raw dataset, full/resume/tuning checkpoints, and
generated output caches. The acceptance scripts scan for common API-key, token, and
private-key patterns. Do not commit a real `CHECKPOINT_URL`, credentials, private
data, `best.pt`, or `last.pt`. Publish only `minigpt_final.pt` deliberately via a
release if the instructor needs a downloadable checkpoint.
