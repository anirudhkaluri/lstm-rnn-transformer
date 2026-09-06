# Chapter 1 — The Sequential Bottleneck

---

## What you already know

You finished LSTM. You know the cell state update:

c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

where:
- c_t ∈ ℝ^(H×1) — cell state at time t, H = hidden size
- f_t ∈ ℝ^(H×1) — forget gate, each entry ∈ (0,1), from a sigmoid
- c̃_t ∈ ℝ^(H×1) — candidate new memory
- i_t ∈ ℝ^(H×1) — input gate, each entry ∈ (0,1)
- ⊙ — elementwise multiplication

You know that the "+" in this equation is the **constant error carousel** — gradients pass through addition without being crushed by a weight matrix or squashed by tanh. This fixed RNN's vanishing gradient problem for *moderate* distances.

So why isn't LSTM the final answer? Two problems remain. Neither is about gradients vanishing in the *math* sense — they're about **architecture**.

---

## Problem 1 — LSTM is fundamentally sequential

Look at the equation again:

c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

To compute c_t, you need c_{t-1}. To compute c_{t-1}, you need c_{t-2}. And so on, all the way back to c_0.

This means: **you cannot compute c_5 before c_4 exists.** Not "it's hard" — it's mathematically impossible, because c_5's formula has c_4 sitting inside it.

For a sequence of T = 1000 tokens, training requires **1000 sequential steps**, one after another, no shortcuts. Each step is a small matrix multiplication. GPUs are built to do *millions of multiplications at once in parallel* — but here, step 743 cannot start until step 742 finishes. You're forcing a massively parallel machine to work like a single-file line at a checkout counter.

This is a **speed/hardware** problem, not a math problem. But it's a huge one — it's why training large LSTMs on long sequences is painfully slow.

---

## Problem 2 — The "highway" still has friction

The forget gate f_t has entries strictly between 0 and 1 (that's what sigmoid guarantees — it never outputs exactly 0 or exactly 1). Even though the gradient through "+" doesn't get multiplied by a *weight matrix*, the cell state itself still gets multiplied by f_t **elementwise, at every single timestep**, in the forward pass — and the gradient inherits a similar elementwise product going backward.

Let's make this concrete. Suppose one entry of f_t is consistently **0.9** (meaning: "keep 90% of this memory slot, forget 10%, every step"). Watch what happens to a piece of information stored at t=0 as it travels forward:

| Steps traveled (k) | 0.9^k | Meaning |
|---|---|---|
| 1 | 0.9000 | barely touched |
| 5 | 0.5905 | already lost ~41% |
| 10 | 0.3487 | lost ~65% |
| 50 | 0.0052 | lost ~99.5% |
| 100 | 0.0000266 | essentially gone |

Even at f_t = 0.9 — which is "remember almost everything" — information surviving 100 steps is multiplied by **0.0000266**. That's not zero, but for a real network with finite numerical precision and noisy gradients, it's indistinguishable from zero.

You could push f_t closer to 1, say 0.99:

| Steps traveled (k) | 0.99^k |
|---|---|
| 100 | 0.366 |
| 500 | 0.0066 |
| 1000 | 0.0000432 |

Same story, just slower. **Any f_t < 1 means information decays exponentially with distance.** LSTM widened the highway and reduced the friction — it did not remove it. For sequences of hundreds or thousands of tokens (a paragraph, a document, a codebase), the decay still adds up.

---

## The concrete example that motivates everything

Recall this sentence from the LSTM course:

> **"The cats that were sitting on the mat are hungry."**

```
t=1      t=2    t=3     t=4    t=5      t=6   t=7    t=8    t=9
"The"  "cats" "that" "were" "sitting" "on"  "the"  "mat"  "are"  → predict: "hungry"
```

T = 9 (sequence length — 9 tokens). Position 9 ("are") needs information from position 2 ("cats") to correctly predict "hungry" (subject-verb agreement: cats → are, not "cat" → "is").

**With LSTM:** information from position 2 must pass through c_3, c_4, c_5, c_6, c_7, c_8 before reaching c_9. That's **7 hops**, each one multiplying by some f_t < 1. The path is *short* in this toy example (only 7 steps), so LSTM handles it fine. But imagine the subject were 200 words back, in a long paragraph — 200 hops of decay.

**The big idea for this course:** what if position 9 could just... look directly at position 2? Not through a chain of 7 intermediate states, but with **one direct connection** — a single operation that says "how relevant is 'cats' to me, right now, while I'm processing 'are'?"

That direct connection, computed between *every pair of positions* in the sequence, is called **self-attention**. The distance between any two tokens becomes **1 hop**, no matter whether they're 2 words apart or 2000 words apart.

---

## Setting up notation for the rest of this course

We'll build a running example with small, concrete numbers. Define:

- **T** = sequence length = number of tokens = **9** (our sentence above)
- **d** = embedding dimension = size of the vector that represents each token's meaning. We'll use **d = 4** for hand-computable examples.
- **x_t ∈ ℝ^(d×1)** = the embedding vector for the token at position t. E.g. x_2 ∈ ℝ^(4×1) is the 4-number vector representing "cats". (This is the exact same x_t you used as input to your RNN/LSTM — nothing new here.)
- **X ∈ ℝ^(T×d)** = the matrix you get by stacking all T embedding vectors as **rows**. For our sentence, X is a 9×4 matrix — row 2 is x_2ᵀ, the embedding for "cats".

Note the shift in mindset: in RNN/LSTM, you processed **one x_t at a time**, carrying state forward. In a Transformer, you'll have **the entire matrix X — all 9 tokens at once** — available simultaneously. Nothing is hidden in a "memory" that has to survive a journey through time. Everything is just... there, all at once, in X.

The question Chapter 2 answers is: **given X (all tokens, all at once), how does token 9 ("are") figure out that token 2 ("cats") is the one it should pay attention to — and not token 5 ("sitting") or token 8 ("mat")?**

---

## Summary

| Question | Answer |
|---|---|
| Can LSTM compute c_t without c_{t-1}? | **No** — strictly sequential, blocks parallelization |
| Does LSTM's gradient highway fully solve decay? | **No** — f_t < 1 still causes exponential decay over distance, just slower |
| What's the fix? | **Self-attention**: every position connects directly to every other position, 1 hop |
| New notation | T = sequence length (9), d = embedding dim (4), x_t ∈ ℝ^(d×1), X ∈ ℝ^(T×d) |
| Coming next | Chapter 2 — how a token computes "relevance" to other tokens (Query, Key, Value) |
