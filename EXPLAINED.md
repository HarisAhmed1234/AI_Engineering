# Study Notes

## M1 - Scaled dot-product self-attention

`SelfAttention` lets every token decide which token representations are relevant to
it. An input `x` has shape `(B, T, D)`: batch size, sequence length, and model
width. Three learned linear layers make `Q` and `K` of shape `(B, T, d_k)` and `V`
of shape `(B, T, d_v)`. Multiplying `Q @ K.transpose(-2, -1)` compares every query
position with every key position, yielding `(B, T, T)` scores. Dividing by
`sqrt(d_k)` keeps score magnitudes controlled before softmax. Softmax across the
last axis makes each query row a probability distribution. Those probabilities
weight `V`, producing `(B, T, d_v)`, which the output projection returns to
`(B, T, D)`.

For a causal language model, a token may only use its own and earlier positions.
The upper triangle of the score matrix is set to negative infinity before softmax;
softmax then assigns it exactly zero probability. That is why replacing future
tokens cannot alter earlier outputs in the causal test.

## M1 - Why attention changes the computation pattern

An RNN has a dependency chain: it must compute position 1 before position 2, then
position 3, and so on. A self-attention layer computes all rows of `QK^T` through
parallel matrix multiplication, so it has constant sequential depth within a
layer. Its cost is the `T x T` score matrix: attention can use more time and memory
than an RNN at long sequence lengths. The measured benchmark records this trade-off
instead of assuming a particular speed winner.

## M2 - Multi-head attention

One attention head has one similarity pattern. Multi-head attention lets separate
small heads learn different patterns. With `D=4` and `H=2`, each head has
`d_head=D/H=2` features. The projected Q, K, and V tensors begin as `(B,T,D)` and
are reshaped to `(B,H,T,d_head)`. Each head creates a separate `(B,T,T)` score and
weight matrix, then produces `(B,T,d_head)` values. Joining all heads returns
`(B,T,D)`, and a final linear layer mixes the heads. Heads do not attend to each
other during the score calculation; they are combined only after their individual
weighted-value outputs are made.

## M2 - Positional encoding

Attention by itself has no left-to-right order: if we consistently permute token
rows, keys, and queries, its calculation follows that permutation. A positional
vector is added to each token embedding to break this symmetry. Fixed sinusoidal
encodings use sine on even dimensions and cosine on odd dimensions at several
frequencies. They have no learned parameters and can describe positions not seen
during training. The optional learned alternative stores a separately trained vector
per position. Its M2 visual is only a structural comparison because it has not been
trained; no quality claim is made until a controlled training experiment exists.

## M3 - Pre-LayerNorm Transformer block

The block keeps the tensor shape `(B,T,D)` throughout. First it normalizes `x`, runs
causal multi-head attention, and adds that result back to the original `x`. It then
normalizes this updated tensor, sends each position independently through a wider
FFN `D -> 4D -> D` with GELU, and adds that result back too. “Pre-LN” means the
LayerNorm happens before each sublayer. The residual path stays direct, which usually
makes gradients more stable and reduces sensitivity to warmup choices. The causal
test changes later inputs and verifies earlier block outputs are unchanged.

## M3 - MLM versus CLM

BERT-style MLM hides selected tokens but lets an unmasked position attend both left
and right. That is useful when reconstructing a missing word from full context.
GPT-style CLM predicts the next token while seeing only the prefix. In the M3 toy
experiment, target `t` was input token `t+1`. With bidirectional attention, query
`t` can look directly at that future token; its trained loss fell from 2.7976 to
0.0001. The causal version cannot take that shortcut and ended at 1.6146 in the same
150-step setup. This is why a GPT cannot use bidirectional attention for next-token
generation: it would train by looking at answers it will not have at generation time.
The toy tokens are random, so its causal loss reduction is memorization of 64 repeated
sequences, not evidence of language ability.

## M4 - Data, model, and training pipeline

The character tokenizer assigns an ID to each sorted unique character, making the
mapping deterministic: the downloaded corpus produced 65 IDs. There are no special
tokens because every corpus character is valid and all windows are the same length.
For a window length `T`, the dataset returns `x=tokens[i:i+T]` and
`y=tokens[i+1:i+T+1]`: the target is input shifted left by one character.

MiniGPT maps IDs `(B,T)` to embeddings `(B,T,D)`, adds the selected position vectors,
passes them through N M3 blocks, applies final LayerNorm, and projects to logits
`(B,T,V)`. It does not scale token embeddings by `sqrt(D)`; this follows the simple
GPT-style learned embedding initialization used here. We initialize embeddings and
linear layers with normal standard deviation 0.02, and scale residual output
projections by `1/sqrt(2N)` to control accumulation across blocks.

Training uses AdamW, linear warmup then cosine decay, gradient clipping, and token
cross-entropy over flattened `(B*T,V)` logits/targets. The M4 CUDA probe passed with
float16 AMP, so the smoke run logged that precision. The tiny repeated-batch gate
fell from 2.122749 to 0.000243 (0.01% of its initial loss), proving gradients,
targets, and updates work before long training.

## M5 - Full training, checkpoint selection, and generation

The full run feeds batches of character IDs with shape `(64, 128)` into MiniGPT.
The model returns logits `(64, 128, 65)`, one unnormalized score for every possible
next character. Cross-entropy compares these logits with the same-shaped shifted
targets after flattening batch and time into `(8192, 65)` predictions. The logged
training loss is an average over all optimization steps since the previous
evaluation, so it is less sensitive to one unusually easy or difficult batch.

The configuration used 40,000 steps, AdamW, a 100-step linear warmup followed by a
cosine schedule, a 128-character context, four 128-wide Transformer blocks, and
float16 CUDA autocast only after a finite forward/backward probe. A short matched
300-step check selected learning rate `1e-3`: its validation loss was 2.515353,
lower than 3.242578 for `3e-4`. That is a local tuning measurement, not a general
rule about learning rates.

The final row is useful for documenting the whole run, but it is not automatically
the best model. Here, the lowest validation loss was 1.491856 at step 21,000; at
step 40,000 the average training loss was lower (1.129923) while validation loss was
higher (1.519580). Therefore `minigpt_final.pt` copies the *best-validation* weights
and includes only `model_state`, `config`, and `vocabulary`; it deliberately omits
optimizer moments because generation does not need them.

Generation is autoregressive. Start with the prompt IDs, crop to the most recent
128 IDs if necessary, obtain the final-position logits, divide them by temperature,
keep the most likely 40 candidates, sample one token, append it, and repeat. Calling
`model.eval()` makes dropout inactive. The trained-model causality check changes only
future IDs and compares earlier logits; the measured maximum difference was exactly
zero, which is the expected consequence of the triangular causal mask.

## M6 Part A - Ablations, attention inspection, and failure analysis

An ablation changes one design choice while holding the others fixed. Here every
primary run used the same 8,000-step budget, seed 1337, data split, optimizer, and
learning rate. The depth study changes only the number of blocks; the heads study
changes only `n_heads`; the context study changes only the context length. This makes
the CSV comparison meaningful in a way that comparing the 40,000-step M5 run with a
short run would not be. A second 4-layer/4-head/128 baseline used seed 2024. Its
validation-loss difference from seed 1337 was 0.005491, which is a useful but very
rough two-seed noise check.

The measured depth and context loss changes are substantially larger than that rough
noise number. The adjacent head-count loss changes are only 0.015696 and 0.008293,
so two seeds are not enough to claim a reliable quality winner. Head count does not
change parameter count here because the total model width stays 128; it changes how
that width is split across heads. The cost panels show the measured trade-offs, but
laptop-GPU timing can vary with thermal state.

Attention visualization calls the trained model with `return_attention=True` after
`model.eval()`. Each layer returns one tensor shaped `(B, H, T, T)`: for a query
character at row `t`, the last axis is its probability distribution over key
characters. Every row should sum to one (up to floating-point rounding), and every
entry above the causal diagonal must be exactly zero. In the recorded figure, the
largest forbidden future weight was 0.0 and the largest row-sum error was
1.788e-07. Bright near-diagonals mean some heads focus heavily on the current or
nearby prior character; they do not by themselves prove the model understands a
particular linguistic rule.

The failure analysis uses saved model output and the loss curve rather than assumed
weaknesses. The model preserves stage-like line breaks and speaker labels but can
produce malformed-looking words and locally fluent yet incoherent statements. It also
has a hard 128-character usable history during generation, because older characters
are cropped before every forward pass.

An additional inference-only check used an early `MONTAGUE` label and generated 400
characters. After the first 128 generated characters, the original prompt was no
longer in the input window. `MONTAGUE` was not generated again, but that single
non-reappearance does not prove why: sampling could also explain it. In contrast,
the greedy `ROMEO:` continuation visibly repeated “the strength of the state,” so
that repetition is direct observed evidence rather than an assumption.

## M6 Part B - Packaging, reproducibility, and final evidence

The CLI and API are thin wrappers around the same saved MiniGPT weights; they do not
contain a second model or hidden pretrained dependency. `minigpt_final.pt` is a
small, weights-only export from the best-validation checkpoint. It stores the model
state, configuration, and vocabulary, which are exactly the pieces inference needs;
optimizer state is intentionally left out because it is only needed to resume
training.

`src.generate` runs autoregressive decoding: it encodes the prompt, retains at most
the newest 128 IDs, obtains the final-position logits, applies temperature/top-k (or
greedy decoding), appends one sampled ID, and repeats. The FastAPI route validates
the same generation controls and loads the CPU model once at application startup.
If a local checkpoint is absent, `CHECKPOINT_URL` supplies a public download URL at
deployment time. That environment variable must be set by the deployer; it is not a
secret embedded in source code.

Reproducibility has two layers. The code/configuration layer is local and checked:
the YAML, fixed seeds, checkpoint metadata, tests, acceptance scripts, output logs,
and notebook spell out the workflow. The publication/deployment layer requires the
student: push the repository, publish the small checkpoint, insert the real public
URLs, run the Colab notebook in its intended environment, and deploy Render if a
live endpoint is required. Keeping those distinctions explicit prevents a local
artifact from being misrepresented as a public service.

