# Presentation Outline (11 Slides)

1. **Problem and learning objective** — Build a causal LLM from Tiny Shakespeare,
   from scratch; experimental-output disclaimer.
2. **Data and tokenization** — 1,155,394 bytes / 40,000 lines, sorted 65-character
   vocabulary, 90/10 positional split, shifted targets.
3. **Why Transformers** — RNN sequential dependency versus attention parallel depth;
   measured CPU/CUDA caveat.
4. **Attention by hand** — Q/K/V, scale by `sqrt(d_k)`, softmax, value mixing, causal
   blocking; show Fig1/Fig2.
5. **Multi-head and position** — split/merge shape trace and sinusoidal encoding;
   Fig3/Fig4.
6. **MiniGPT architecture** — pre-LN block and full model diagrams; Fig5 parameter
   breakdown.
7. **Training setup** — RTX 4050 6 GiB, 40,000 steps, float16 AMP, learning-rate
   check selected `1e-3`.
8. **Training results** — Fig6, best 1.491856 at step 21,000; final 1.129923/1.519580
   train/validation losses and best-checkpoint caveat.
9. **Generated text and attention inspection** — Fig8/Fig7; formatting imitation,
   causal-triangle check, experimental-output warning.
10. **Ablations and failure analysis** — Fig9, two-seed 0.005491 rough noise,
    context/depth findings and coherence/context limitations.
11. **Packaging and conclusion** — CLI/FastAPI Fig10, reproducibility assets,
    manual deployment/publication actions.

