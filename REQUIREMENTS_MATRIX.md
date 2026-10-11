# Requirements Matrix

Status legend: **Scaffolded** means the destination exists but has no implementation
or evidence yet; **Planned** means it is not yet started; **Manual** means it needs
student-controlled external action. The handout is the source of truth. Rows marked
`User addendum` preserve additional requested work that does not contradict it.

| Requirement | PDF section | File or output that satisfies it | Status |
| --- | --- | --- | --- |
| Build a small causal LM from a raw corpus through next-token generation, without pretrained GPT weights | 1 | `src/`, `report/report.md`, trained checkpoint | Complete (trained and fresh-load checked) |
| Reproducible clean-checkout system with architecture evidence | 1 | `README.md`, tests, configs, outputs | Complete locally (public clean-clone check remains Manual) |
| Explain sequential versus parallel depth | 2, 3 (Lesson 01) | `experiments/why_transformers.py`, report | Complete (measured) |
| Derive and implement scaled dot-product attention with masking | 2, 3 (Lesson 02) | `experiments/hand_calculation.py`, `src/attention.py`, tests | Complete (tested) |
| Implement multi-head attention and track every tensor transformation | 2, 3 (Lesson 03) | `src/attention.py`, shape trace, Fig3 | Complete (tested) |
| Explain and implement positional information | 2, 3 (Lesson 04) | `src/positional_encoding.py`, Fig4, report | Complete (tested) |
| Assemble Transformer attention, FFN, residual, and normalization components | 2, 3 (Lesson 05) | `src/transformer_block.py`, diagrams, tests | Complete (tested) |
| Distinguish bidirectional MLM from autoregressive CLM | 2, 3 (Lesson 06) | `experiments/mlm_vs_clm.py`, report | Complete (measured) |
| Construct shifted next-token data and loss for GPT | 2, 3 (Lesson 07) | `src/dataset.py`, `src/train.py`, tests | Complete (tested) |
| Train/evaluate a compact model without a pretrained checkpoint | 2 | `outputs/logs/m5_training_summary.json`, checkpoints, report §5/§6 | Complete (40,000-step measured run) |
| Run controlled ablations with measurements | 2 | `experiments/ablation.py`, `outputs/logs/ablation_results.csv`, Fig9, report §7 | Complete (three 8,000-step controlled studies plus two-seed baseline noise check) |
| Package an API and deploy/host it online | 2 | `app/`, `Dockerfile`, `render.yaml`, local Fig10 evidence; hosted URL | Complete locally (Render deployment is Manual) |
| Lesson 01 evidence: hand analysis plus measured or reasoned speed/memory comparison | 3 | `experiments/why_transformers.py`, report | Complete (CPU timing/profiler memory and CUDA synchronized timing/peak memory measured) |
| Lesson 02 evidence: worked numeric example, implementation, unit test | 3 | hand-calculation script, attention module, tests | Complete (tested) |
| Lesson 03 evidence: shape trace and attention output | 3 | shape trace text/image, Fig3 | Complete (verified) |
| Lesson 04 evidence: position visualization and explanation | 3 | Fig4, report | Complete (verified) |
| Lesson 05 evidence: block code, diagram, shape checks | 3 | block module, diagrams, tests | Complete (tested/visually reviewed) |
| Lesson 06 evidence: mask matrix and short comparison experiment | 3 | MLM/CLM script and figure | Complete (measured) |
| Lesson 07 evidence: training loop, loss curves, text generation | 3 | `src/train.py`, Fig6, Fig8, samples | Complete (full run/evaluation/generation verified) |
| Public or instructor-accessible repository | 4 | Git remote URL | Manual |
| Google Colab script | 4 | `notebooks/colab_demo.ipynb` | Complete structurally (JSON and all code cells compile; student must set public URL placeholders) |
| Trained checkpoint or reproducible command that obtains it | 4 | local `checkpoints/minigpt_final.pt`, README release placeholder | Complete locally (public release URL is Manual) |
| README documents setup, dataset, training, inference, expected output, limitations | 4 | `README.md` | Complete (static command/artifact check passed) |
| Medium-style technical report, recommended 8–12 pages excluding appendix | 4, 15 | `report/report.md` | Complete (authoring; publication is Manual) |
| At least six required evidence figures | 4, 12 | `outputs/figures/` and report | Complete (all ten required figures embedded) |
| Ablation-results table and discussion | 4, 13 | `outputs/logs/ablation_results.md`, Fig9, report §7 | Complete (measured) |
| 5–10 minute screen-recorded demonstration | 4 | recording plus `DEMO_SCRIPT.md` | Manual |
| Class presentation | 4 | `SLIDES_OUTLINE.md`, delivered slides | Complete planning; delivery is Manual |
| Definition of done: clone, install, tests, architecture checks, generation without undocumented steps | 4 | `notebooks/colab_demo.ipynb`, `scripts/acceptance.sh`, `scripts/acceptance.ps1`, README | Complete locally (public clone/Colab execution remains Manual) |
| M1 checkpoint: numerical attention example and passing self-attention tests | 5 | hand calculation and tests | Complete (5 tests passed) |
| M2 checkpoint: shape-verified attention module and position visualization | 5 | shape trace and Fig4 | Complete (8 tests passed) |
| M3 checkpoint: complete block and mask comparison | 5 | block code and MLM/CLM evidence | Complete (10 tests passed) |
| M4 checkpoint: loss decreases on a small overfit set | 5 | `tests/test_overfit.py` output | Complete (CUDA gate passed) |
| M5 checkpoint: full-data training and generated samples | 5 | training log, `minigpt_final.pt`, five samples | Complete (40,000-step run; fresh-load verified) |
| M5: record final parameter count | 5.5 | `outputs/logs/m5_training_summary.json`, report §5/§6 | Complete (809,984) |
| M5: record full training configuration | 5.5 | training summary, YAML/checkpoint metadata, report §5 | Complete |
| M5: record hardware used | 5.5 | training summary and report §5 | Complete (RTX 4050, PyTorch 2.6.0+cu124) |
| M5: record wall-clock training time | 5.5 | training log and report §6 | Complete (1,516.183 s) |
| M5: record final training loss | 5.5 | training log and report §6 | Complete (1.129923 average) |
| M5: record final validation loss | 5.5 | training log and report §6 | Complete (1.519580; best 1.491856 at step 21,000) |
| M5: record at least five fixed-prompt generated samples | 5.5 | `outputs/samples/m5_sample_*.txt`, Fig8, report §6 | Complete (five fixed 200-new-character samples/settings saved) |
| M4: log training loss | 5.4 | `outputs/logs/training_log.csv` | Complete (smoke run verified) |
| M4: log validation loss | 5.4 | `outputs/logs/training_log.csv` | Complete (smoke run verified) |
| M4: log learning rate | 5.4 | `outputs/logs/training_log.csv` | Complete (smoke run verified) |
| M4: log step and epoch | 5.4 | `outputs/logs/training_log.csv` | Complete (smoke run verified) |
| M4: log at least one sample generation during training or evaluation | 5.4 | `outputs/samples/step_2.txt` | Complete (smoke run verified) |
| M6 checkpoint: ablation, CLI/API, final report | 5 | experiments, app, report, docs | Complete locally (public deployment/publication remains Manual) |
| Hand calc: use three tokens with d_k=2 and d_v=2 and explicit Q/K/V | 6 | `experiments/hand_calculation.py`, report, Fig1 | Complete (verified) |
| Hand calc: calculate QK^T | 6 | hand-calculation output | Complete (verified) |
| Hand calc: divide scores by sqrt(d_k) | 6 | hand-calculation output | Complete (verified) |
| Hand calc: apply row-wise softmax | 6 | hand-calculation output | Complete (verified) |
| Hand calc: multiply weights by V | 6 | hand-calculation output | Complete (verified) |
| Hand calc: add and show causal masked scores | 6 | hand-calculation output, `outputs/figures/fig2_causal_mask.png` | Complete (verified and visually reviewed) |
| Hand calc: explain each position’s accessible information | 6 | report §2 and hand-calculation output | Complete (verified) |
| Deterministic tokenizer with documented vocabulary size and special-token policy | 7 | `src/tokenizer.py`, report | Complete (65-character vocabulary; no special tokens) |
| Token embeddings map IDs to d_model vectors | 7 | `src/gpt.py` | Complete (tested) |
| Sinusoidal position encoding implemented by student | 7 | `src/positional_encoding.py` | Complete (tested) |
| Self-attention has Q/K/V projections, scaled dot product, softmax, optional mask | 7 | `src/attention.py` | Complete (tested) |
| Multi-head attention has configurable heads and explicit split/merge | 7 | `src/attention.py` | Complete (tested) |
| FFN has two linear layers and nonlinearity | 7 | `src/transformer_block.py` | Complete (Linear -> GELU -> Linear) |
| Normalization uses LayerNorm or justified equivalent | 7 | block code and report | Complete (pre-LayerNorm justified) |
| Residual connections surround attention and FFN | 7 | `src/transformer_block.py` | Complete (tested) |
| Causal masking prevents future access | 7 | attention module and `tests/test_mask.py` | Complete (tested) |
| LM head projects d_model to vocabulary logits | 7 | `src/gpt.py` | Complete (tested) |
| Token-level cross-entropy uses correctly shifted targets | 7 | dataset/train modules and test | Complete (tested) |
| Autoregressive generation supports max length and temperature | 7 | `src/generate.py`, M5 CLI output | Complete (fresh-load CLI verified) |
| No pretrained language-model checkpoint | 8 | source review, README/report, `outputs/logs/forbidden_api_check.txt` | Complete (static source review; trained weights are this project’s own) |
| No transformers high-level model class in submitted model | 8 | `outputs/logs/forbidden_api_check.txt` | Complete (static search: no matches in src/app/experiments) |
| Use only permitted PyTorch-level building blocks | 8 | source review, `outputs/logs/forbidden_api_check.txt` | Complete (no prohibited high-level APIs found) |
| Explain produced tokenizer/vocabulary even if a tokenizer library is used | 8 | report §3/§4, `EXPLAINED.md` | Complete |
| Cite and understand externally adapted implementation | 8, 18 | README/report citations, `VIVA_QA.md` | Complete (citations and adaptation statements added) |
| Reproduce experiments from clean checkout with documented commands | 8 | README, configs, scripts, Colab notebook | Complete structurally (student must verify public clean clone) |
| Do not commit keys, credentials, private data, or unnecessary large artifacts | 8, 18, 19 | `.gitignore`, `.env.example`, `scripts/acceptance.*` | Complete locally (ignore policy and secret scan verified; final Git review remains Section 19) |
| Use Tiny Shakespeare (~1.11 MB, ~40k lines) | 9 | `data/README.md`, downloader, report | Complete (1,155,394 bytes; 40,000 lines) |
| Make model/data/tests/experiments/report locations obvious in README | 10 | `README.md` | Complete |
| Provide tokenizer-determined configurable vocab size | 11 | YAML and tokenizer | Complete (65 observed) |
| Provide configurable context length (starting recommendation 128) | 11 | YAML | Complete |
| Provide configurable d_model (starting recommendation 128 or 256) | 11 | YAML | Complete |
| Provide configurable n_heads (starting recommendation 4) | 11 | YAML | Complete |
| Provide configurable n_layers (starting recommendation 4) | 11 | YAML | Complete |
| Provide configurable ffn_dim (starting recommendation 4*d_model) | 11 | YAML | Complete |
| Provide configurable dropout (starting recommendation 0.0–0.1) | 11 | YAML | Complete |
| Use configurable AdamW optimizer | 11 | YAML/train module | Complete |
| Provide configurable learning rate (start around 3e-4) | 11 | YAML/train module | Complete |
| Provide configurable batch size | 11 | YAML/train module | Complete |
| Provide configurable generation temperature (baseline 0.8) | 11 | `src/generate.py` CLI | Complete (default 0.8; verified) |
| Justify actual training configuration and document reductions | 11 | report §5, `outputs/logs/m5_lr_tuning.txt` | Complete (measured LR choice, hardware, precision, and configuration documented) |
| Fig1: hand-calculated attention proves Q/K/V/scaling/softmax | 12 | `outputs/figures/fig1_hand_calculated_attention.png` | Complete (visually reviewed) |
| Section 12: every screenshot includes enough context to prove it came from this implementation | 12 | figure-generation scripts, terminal evidence, report captions | Complete (scripts/logs/captions identify implementation and settings) |
| Fig2: causal-mask matrix/heatmap proves future blocking | 12 | `outputs/figures/fig2_causal_mask.png` | Complete (visually reviewed) |
| Fig3: multi-head tensor-shape printout proves split/merge | 12 | `outputs/figures/fig3_multihead_shape_trace.png` | Complete (visually reviewed) |
| Fig4: positional-encoding visualization proves systematic positional effect | 12 | `outputs/figures/fig4_positional_encoding.png` | Complete (visually reviewed) |
| Fig5: model summary/parameter count identifies actual scale | 12 | `outputs/figures/fig5_model_summary.png` | Complete (809,984 parameters; visually reviewed) |
| Fig6: train/validation loss curve proves learning/monitoring | 12 | `outputs/figures/fig6_loss_curve.png` | Complete (best step marked; visually reviewed) |
| Fig7: trained-model attention visualization proves real behavior inspection | 12 | `outputs/figures/fig7_attention.png`, `outputs/logs/m6_attention_visualization.json` | Complete (eval-mode real weights; causal/row-sum checks passed) |
| Fig8: generation samples prove autoregressive generation | 12 | `outputs/figures/fig8_samples.png` | Complete (five fixed 200-new-character samples with real line breaks) |
| Fig9: ablation table/plot proves measured controlled change | 12 | `outputs/figures/fig9_ablations.png`, CSV/Markdown table | Complete (visually reviewed) |
| Fig10: CLI/API call proves usable packaging | 12 | `outputs/figures/fig10_inference.png`, `outputs/logs/fig10_cli.txt`, `outputs/logs/fig10_api.txt` | Complete (student-run local real CLI and HTTP calls) |
| Terminal evidence records prompt, generated text, parameters, context length, temperature | 12.1 | `outputs/samples/m5_sample_*.txt`, `outputs/logs/fig10_cli.txt`, Fig10 | Complete (M5 and Fig10 CLI evidence) |
| Controlled depth study: 2 vs 4 vs 6 layers; val loss, parameters, time | 13 | `outputs/logs/ablation_results.csv`, Fig9, report §7 | Complete (8,000 steps/seed 1337) |
| Controlled head study: 2 vs 4 vs 8 heads; val loss and generation quality | 13 | head sample files, CSV/table/plot, report §7 | Complete (same fixed prompts saved; no unsupported quality ranking) |
| Controlled context study: 64 vs 128 vs 256; val loss, memory, throughput | 13 | CSV/table/plot, report §7 | Complete (8,000 steps/seed 1337) |
| Separate observations from evidence-supported interpretation | 13 | report §7 | Complete (noise-qualified) |
| Conceptual-understanding rubric coverage | 14 | report, `EXPLAINED.md`, `VIVA_QA.md` | Complete (study material and evidence reviewed) |
| Attention-implementation rubric coverage | 14 | source and tests | Complete (static/test evidence reviewed) |
| Transformer-architecture rubric coverage | 14 | source, diagram, tests | Complete (static/test evidence reviewed) |
| Language-modeling-pipeline rubric coverage | 14 | data/train/checkpoint code | Complete (tested/full-run evidence reviewed) |
| Experiments-and-analysis rubric coverage | 14 | experiments and report | Complete (measured figures/logs reviewed) |
| Generation-quality/evaluation rubric coverage | 14 | evaluation and samples | Complete (fixed prompts, losses, and failure analysis reviewed) |
| Software-engineering rubric coverage | 14 | project structure, tests, docs, API | Complete locally (public deployment remains Manual) |
| Report-and-demo rubric coverage | 14 | report, figures, demo script/recording | Complete authoring; recording is Manual |
| Report §1: problem and scenario | 15 | `report/report.md` | Complete |
| Report §2: Transformer foundations, equations, shapes, motivation | 15 | `report/report.md` | Complete |
| Report §3: implementation | 15 | `report/report.md` | Complete |
| Report §4: dataset and preprocessing | 15 | `report/report.md` | Complete |
| Report §5: training setup | 15 | `report/report.md` | Complete (measured configuration/hardware/training method) |
| Report §6: results | 15 | `report/report.md` | Complete (measured losses, checkpoint selection, and samples) |
| Report §7: ablation study | 15 | `report/report.md` | Complete (M6 Part A) |
| Report §8: two failure modes/limitations with likely causes | 15 | `report/report.md` | Complete (four evidence-linked modes) |
| Report §9: engineering packaging | 15 | `report/report.md` | Complete (local verification; deployment Manual) |
| Report §10: conclusion | 15 | `report/report.md` | Complete |
| Acceptance test: install dependencies | 16 | `scripts/acceptance.sh`, `scripts/acceptance.ps1` | Complete structurally (scripts install `requirements.txt`; execution deferred) |
| Acceptance test: run unit tests | 16 | `python -m pytest -q` output | Complete (15 passed) |
| Acceptance test: run tiny overfit test | 16 | `tests/test_overfit.py` | Complete (M4 CUDA gate previously passed; included in 15-test suite) |
| Acceptance test: load final checkpoint in fresh process | 16 | `experiments/m5_acceptance_checks.py`, log | Complete (verified) |
| Acceptance test: generate from three fixed prompts | 16 | M5 acceptance log / samples | Complete (verified) |
| Acceptance test: future tokens never affect earlier generation | 16 | M5 acceptance log | Complete (trained model difference 0.00000000) |
| Acceptance test: README matches repository and no secrets are committed | 16 | `scripts/acceptance.*`, `.gitignore`, README static check | Complete locally (Git history/push review remains Manual) |
| Unit tests pass without manual edits | 16 | `python -m pytest -q` output | Complete (15 passed) |
| Tiny repeated batch loss falls substantially | 16 | overfit test/log | Complete (M4 gate passed; suite re-run in M5) |
| Fresh process loads weights and generates | 16 | `outputs/logs/m5_acceptance_checks.txt` | Complete |
| Causal correctness blocks positions > t | 16 | `outputs/logs/m5_acceptance_checks.txt` | Complete (trained logits unchanged) |
| README has exact inference/configuration commands | 16 | README static verification | Complete (commands/targets found) |
| Demo: show repository and core model | 17 | `DEMO_SCRIPT.md` | Complete (script prepared) |
| Demo: show attention and causal mask | 17 | `DEMO_SCRIPT.md` | Complete (script prepared) |
| Demo: run tests/architecture checks | 17 | `DEMO_SCRIPT.md` | Complete (script prepared) |
| Demo: show model configuration and parameter count | 17 | `DEMO_SCRIPT.md` | Complete (script prepared) |
| Demo: show train/validation loss curve | 17 | `DEMO_SCRIPT.md` | Complete (script prepared) |
| Demo: run fixed-prompt generation | 17 | `DEMO_SCRIPT.md` | Complete (script prepared) |
| Demo: show and explain attention visualization | 17 | `DEMO_SCRIPT.md` | Complete (script prepared) |
| Demo: show deployed live inference | 17 | `DEMO_SCRIPT.md` | Complete script; live deployment is Manual |
| Demo: summarize ablation and limitation | 17 | `DEMO_SCRIPT.md` | Complete (script prepared) |
| Each member understands submitted model | 18 | `EXPLAINED.md`, `VIVA_QA.md`, student preparation | Complete resources; individual study is Manual |
| Cite external code/tutorials/equations/figures and identify adaptations | 18, 20 | README/report citations | Complete |
| Exclude unpermitted private/personal/confidential/licensed data | 18 | dataset documentation, README | Complete locally (Tiny Shakespeare only; student Git review Manual) |
| Label generated text as experimental, not factually reliable | 18 | README/report/API notice | Complete |
| Keep project reproducible and avoid large checkpoints unless requested | 18 | `.gitignore`, checkpoint-download guidance | Complete locally |
| Section 19: repository link/access correct | 19 | `FINAL_CHECKLIST.md`, Git URL | Manual |
| Section 19: README setup works | 19 | `FINAL_CHECKLIST.md`, README static check | Complete locally; public clean-clone check Manual |
| Section 19: hand calculation included | 19 | `FINAL_CHECKLIST.md`, Fig1 | Complete |
| Section 19: causal mask shown | 19 | `FINAL_CHECKLIST.md`, Fig2 | Complete |
| Section 19: multi-head shape trace included | 19 | `FINAL_CHECKLIST.md`, Fig3 | Complete |
| Section 19: positional evidence included | 19 | `FINAL_CHECKLIST.md`, Fig4 | Complete |
| Section 19: architecture diagram included | 19 | `outputs/figures/fig_m3_transformer_block.png`, `outputs/figures/fig_m3_minigpt_architecture.png` | Complete (visually reviewed) |
| Section 19: training/validation curves included | 19 | Fig6, later `FINAL_CHECKLIST.md` | Complete (M5 evidence produced) |
| Section 19: at least five generated samples included | 19 | Fig8/samples, later `FINAL_CHECKLIST.md` | Complete (M5 evidence produced) |
| Section 19: three required ablations complete | 19 | Fig9/CSV, later `FINAL_CHECKLIST.md` | Complete (M6 Part A evidence produced) |
| Section 19: parameter count/configuration documented | 19 | `FINAL_CHECKLIST.md`, Fig5/report | Complete |
| Section 19: CLI/API inference works | 19 | `FINAL_CHECKLIST.md`, Fig10 | Complete locally (public hosting Manual) |
| Section 19: report submitted as Medium Article | 19 | `FINAL_CHECKLIST.md`, report | Manual |
| Section 19: no secrets/unneeded large artifacts committed | 19 | `FINAL_CHECKLIST.md`, `.gitignore` | Complete locally; Git history review Manual |
| Section 19: all sources cited | 19 | `FINAL_CHECKLIST.md`, citations | Complete |
| Use/review listed curriculum and starter resources; cite any adaptation | 20 | citations sections | Complete |
| User addendum: no `nn.MultiheadAttention`, `nn.Transformer`, or `F.scaled_dot_product_attention` | User addendum | `outputs/logs/forbidden_api_check.txt` | Complete (static source search returned no matches) |
| M2: show attention output per head, concatenation, and final projection | 5.2 | `src/attention.py`, Fig3, report | Complete (verified) |
| M2 extra credit: compare learned positional embeddings and/or RoPE as a switchable option | 5.2 | `src/positional_encoding.py`, Fig4 and positional experiment | Complete (structural comparison; quality comparison deferred until training) |
| User addendum: warmup + cosine schedule, clipping, device detection, AMP, resume, rich CSV/JSON logs | User addendum | `src/train.py`, logs | Complete (2-step smoke run verified) |
| User addendum: top-k (optional top-p/greedy) generation CLI with reproducible printout | User addendum | `src/generate.py`, M5 CLI output | Complete (top-k/top-p/greedy CLI verified) |
| User addendum: five fixed-prompt samples with seeds/settings | User addendum | `outputs/samples/m5_sample_*.txt`, Fig8 | Complete |
| User addendum: trained attention heatmaps and causal-triangle verification | User addendum | `experiments/attention_visualization.py`, Fig7, JSON evidence | Complete (eval mode; future max 0.0; row sums checked) |
| User addendum: FastAPI endpoints, validation, checkpoint download, Docker/Render assets | User addendum | `app/`, `Dockerfile`, `render.yaml`, Fig10 logs | Complete locally (Render deployment remains Manual) |
| User addendum: Colab notebook runs top-to-bottom | User addendum | `notebooks/colab_demo.ipynb` | Complete structurally (top-to-bottom cells compile; URLs are explicit student placeholders) |
| User addendum: explained study guide, 30+ viva Q&A, slides outline, final checklist, manual todo | User addendum | named Markdown files | Complete (45 Q&A; 11-slide outline) |
| User addendum: publish the completed Medium-style article/report | User addendum | published Medium URL | Manual |
| User addendum: re-read PDF and verify every matrix row against repo at completion | User addendum | updated matrix and final verification | Complete (11-page PDF re-read; static artifact audit completed) |

