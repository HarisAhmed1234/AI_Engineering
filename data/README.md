# Data

Run `python data/download_tiny_shakespeare.py` to fetch the public Tiny Shakespeare
corpus. The script first uses Karpathy's `char-rnn` raw corpus URL and then attempts
a Hugging Face mirror. It writes `data/input.txt`, which is deliberately ignored by
Git. The executed download was 1,155,394 bytes and 40,000 lines.

The preprocessing pipeline makes a 90/10 train/validation split by **corpus
position** before making windows; no example can cross that boundary. It uses a
deterministic character vocabulary from sorted unique characters. The observed
vocabulary contains 65 characters. No filtering or special tokens are used: every
character, including newlines and punctuation, is a token and fixed-length windows
need no padding.

Source: Karpathy Tiny Shakespeare corpus (public `char-rnn` starter corpus), with a
Hugging Face mirror as fallback. The final report/README will preserve source and
license/attribution information before external submission.

