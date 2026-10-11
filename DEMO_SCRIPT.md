# 5–7 Minute Demonstration Script

This script follows the Section 17 order. Say only what is visible or measured.

1. **0:00–0:35 — Repository and core model.** Open the repository tree, then
   `src/gpt.py`. Point out `token embedding -> positional encoding -> N Transformer
   blocks -> final LayerNorm -> LM head`. State the trained configuration: 809,984
   parameters, character vocabulary 65, context 128.
2. **0:35–1:20 — Attention and causal mask.** Open `src/attention.py` and explain
   that Q, K, V are learned projections and `QK^T/sqrt(d_k)` is masked before
   softmax. Show Fig1 and Fig2. Say that the upper triangle is exactly zero after
   softmax, so changing future tokens cannot change earlier logits.
3. **1:20–1:55 — Tests and architecture checks.** Run `python -m pytest` (or show
   the saved passing output where running live is impractical). Show the M2 tensor
   trace in Fig3 and mention the tiny overfit gate was passed before full training.
4. **1:55–2:30 — Configuration and scale.** Open `configs/tiny_gpt.yaml` and Fig5.
   Explain 128 context, width 128, four heads, four blocks, FFN 512, AdamW and the
   selected `1e-3` learning rate from the 300-step comparison.
5. **2:30–3:10 — Learning curve.** Show Fig6. State the real values: best validation
   loss 1.491856 at step 21,000; final train/validation losses 1.129923/1.519580.
   Explain that inference uses best-validation weights, and this same-validation
   selection is slightly optimistic.
6. **3:10–3:45 — Fixed-prompt generation.** Run:
   ```powershell
   python -m src.generate --prompt "ROMEO:" --max-new-tokens 80 --temperature 0.8 --top-k 40 --seed 7 --checkpoint checkpoints/minigpt_final.pt
   ```
   Point out the printed parameter count, context length and temperature. Label the
   continuation experimental rather than factual.
7. **3:45–4:25 — Real attention.** Show Fig7. Explain rows are query characters and
   columns keys. Report the verified maximum future weight 0.0 and maximum row-sum
   error `1.788e-07`; describe only the visibly bright near-diagonal/newline areas.
8. **4:25–5:10 — Packaging/live inference.** Show Fig10 and `app/main.py`. The CLI
   and local API call were verified. If Render has been deployed, open the filled-in
   `LIVE_RENDER_URL`; otherwise say deployment is the documented manual action and
   do not imply it is online.
9. **5:10–6:00 — Ablation and limitation.** Show Fig9. State that depth 2→4 and
   context 64→128→256 changed loss beyond the rough two-seed 0.005491 noise check;
   head-count quality differences were not reliable. End with a concrete limitation:
   output can be grammatically styled but incoherent, and history is cropped to 128
   characters.

