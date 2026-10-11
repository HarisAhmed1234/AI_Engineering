# Viva Questions and Answers

1. **What does this model predict?** One next character at each position. The
   training target is the input window shifted one character left.
2. **Why is the target shifted?** At position `t`, the model sees the prefix through
   character `t` and learns to score character `t+1`.
3. **What are Q, K, and V?** Learned linear projections of token representations:
   queries ask what to look for, keys describe what is available, and values carry
   the information to mix.
4. **Why divide attention scores by `sqrt(d_k)`?** Dot-product variance grows with
   key width. Scaling keeps softmax from becoming needlessly saturated.
5. **What did the hand calculation prove?** With three tokens and `d_k=d_v=2`, it
   shows QK^T, division by `sqrt(2)`, row softmax, value mixing, then the same steps
   after masking future scores.
6. **What does causal masking do?** It writes negative infinity above the diagonal
   before softmax, giving exactly zero future attention weight.
7. **How was causality tested?** Future input IDs were perturbed and earlier logits
   were unchanged; the trained-model maximum difference was `0.00000000`.
8. **Why can attention be parallel within a layer?** All Q/K dot products form one
   matrix multiplication; an RNN must finish each state before the next.
9. **What is attention’s memory cost?** The score/weight matrix is `T x T`, so its
   memory grows quadratically in sequence length within a layer.
10. **What did the CUDA RNN/attention memory test show?** At tiny lengths cuDNN RNN
    workspace overhead made RNN peak allocation higher in several cases. That result
    is reported honestly and is not used as evidence against attention’s asymptotic
    `O(T^2)` memory cost.
11. **Why multiple heads?** Each head has a separate similarity pattern. Their
    outputs are concatenated and mixed by a final linear projection.
12. **What shapes occur in multi-head attention?** `(B,T,D)` projections split to
    `(B,H,T,D/H)`; per-head scores are `(B,H,T,T)`; concatenation returns `(B,T,D)`.
13. **Why positional encoding?** Attention alone is permutation-equivariant, so it
    has no inherent left-to-right order.
14. **Why sinusoidal positions?** They are deterministic, parameter-free vectors at
    multiple frequencies. A learned-position option is also implemented.
15. **What is pre-LayerNorm?** Each sublayer receives normalized input, then its
    result is added to the residual stream: `x + sublayer(LN(x))`.
16. **Why pre-LayerNorm?** The direct residual gradient path is typically more stable
    and less dependent on carefully tuned warmup than post-normalization.
17. **What is the FFN?** A per-position `Linear(128,512) -> GELU -> Linear(512,128)`
    network in each Transformer block.
18. **Why GELU?** It is a smooth nonlinearity commonly used in Transformer FFNs; it
    adds capacity beyond a linear attention mixture.
19. **Why residual connections?** They preserve an identity path and let sublayers
    learn incremental transformations, aiding gradient flow.
20. **What is cross-entropy here?** It compares flattened `(B*T,65)` logits to the
    corresponding flattened shifted target IDs.
21. **What is perplexity?** `exp(cross_entropy)`; it expresses loss on a positive
    multiplicative scale. Final validation perplexity was 4.570307.
22. **How does BERT-style MLM differ from GPT-style CLM?** MLM can use both sides of
    a masked token; CLM only sees the left prefix, so it is suitable for generation.
23. **What was the M3 information leak?** With next-token labels visible through a
    bidirectional mask, a direct copier got loss 0.0000; that shortcut is unavailable
    under the causal mask.
24. **Why is the M3 causal toy result not language ability?** Its tokens were random
    and the causal model memorized 64 repeated sequences.
25. **What tokenizer was used?** A deterministic sorted character vocabulary of 65
    corpus characters. There are no pad/unknown tokens because all windows are fixed
    length and all observed corpus characters are represented.
26. **Why use 90/10 positional splitting?** It preserves corpus order and prevents a
    window from crossing the train/validation boundary.
27. **Why train stride 32 but validation stride 128?** Overlap supplies more varied
    training contexts; non-overlapping validation reduces near-duplicate evaluation
    windows.
28. **Why exclude biases, LayerNorm, and embeddings from weight decay?** Weight decay
    is useful for ordinary weight matrices, while decaying scale/offset parameters or
    lookup embeddings is commonly avoided and is explicit in the optimizer groups.
29. **What learning rate was selected, and why?** `1e-3`: a matched 300-step check
    measured validation loss 2.515353 versus 3.242578 for `3e-4`.
30. **What full configuration was trained?** Context 128, width 128, four heads,
    four blocks, FFN 512, dropout 0.1, batch 64, AdamW, warmup/cosine schedule,
    clipping 1.0, 40,000 steps, seed 1337.
31. **What hardware and precision were used?** RTX 4050 Laptop GPU with 6 GiB,
    PyTorch 2.6.0+cu124, float16 CUDA AMP after a finite stability probe.
32. **What were the principal results?** 809,984 parameters; best validation loss
    1.491856 at step 21,000; final train/validation loss 1.129923/1.519580; 1,516.183
    seconds total.
33. **Why choose the best checkpoint rather than the final checkpoint?** Validation
    loss rose after step 21,000 while train loss fell, so `minigpt_final.pt` exports
    the best-validation weights.
34. **What caveat applies to the best validation score?** It was chosen using the
    same validation windows monitored during training, so it is slightly optimistic
    versus an independent unseen test set.
35. **What did the depth ablation support?** At equal 8,000-step budgets, depth 2→4
    was a meaningful measured loss change; 4→6 was more modest.
36. **What did the head ablation support?** With a rough two-seed baseline difference
    of 0.005491, the 2/4/8 head loss differences were not reliable enough to claim a
    quality winner. Heads changed timing/memory, not parameter count.
37. **What did the context ablation support?** Context 64→128→256 changed validation
    loss beyond that rough noise amount, while memory/time rose substantially.
38. **Why qualify ablation timing?** Laptop GPU thermal and clock state can change
    timings; the reported numbers are indicative measurements on one machine.
39. **What real failure modes appeared?** Malformed-looking words such as `pursuive`
    and `lready`, locally grammatical but semantically incoherent text, repetition in
    greedy/low-temperature samples, and the fixed 128-character history limit.
40. **What did attention visualization verify?** On the eval-mode trained model,
    future weights were exactly 0.0 and maximum row-sum error was `1.788e-07`.
41. **What does temperature do?** It divides logits before sampling: lower values are
    more concentrated; higher values are more varied.
42. **What does top-k do?** It removes every candidate except the k most likely before
    sampling; the fixed samples used `temperature=0.8`, `top_k=40`.
43. **Was weight tying used?** No. The configuration keeps input embedding and LM head
    separate; the option exists for experimentation.
44. **Can this model answer factual questions?** No. It was trained only on Tiny
    Shakespeare and its text is explicitly labelled experimental.
45. **How is deployment made reproducible?** The API checks a local weights-only
    checkpoint and otherwise downloads from a user-set `CHECKPOINT_URL`; no personal
    URL or secret is committed.

