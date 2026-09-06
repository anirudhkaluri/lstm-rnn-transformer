# RNN Course — Chapter Index

> Read chapters in the order listed below. Each chapter builds on the previous one.

---

## The Learning Path

```
Chapter 1 → Chapter 2 → Chapter 3 → Chapter 4 → Chapter 5
    → Chapter 6 → Chapter 6b → Chapter 7 → Chapter 8
```

---

## Chapters

| Order | File | What You Will Learn |
|---|---|---|
| 1 | `chapter-1-what-is-an-rnn.md` | What problem RNNs solve, the memory loop, where RNNs are used |
| 2 | `chapter-2-rnn-vs-regular-networks.md` | Regular network vs RNN, hidden state, tanh introduction |
| 3 | `chapter-3-rnn-loop-with-numbers.md` | The RNN formula, one-hot encoding, step by step with real numbers |
| 4 | `chapter-4-bptt-how-rnn-learns.md` | Full training loop, loss function, softmax, Wy, epochs, batches, BPTT overview |
| 5 | `chapter-5-vanishing-gradient.md` | Why gradients shrink going back through time, exploding gradient, why LSTM was needed |
| 6 | `chapter-6-notations-vectors-dimensions.md` | Every symbol defined with size, matrix multiplication rules, all weight matrices, full Q&A on notation |
| 6b | `chapter-6b-embeddings.md` | Why one-hot fails for words, word embeddings, Word2Vec, GloVe, training vs inference, determinism |
| 7 | `chapter-7-bptt-detailed.md` | Full BPTT worked example with real numbers — forward pass, backward pass, weight update |
| 8 | `chapter-8-derivative-flow.md` | Partial derivatives, chain rule, cross entropy derivation, why ∂L/∂raw = ŷ - y, every gradient path expanded |

---

## What Each Chapter Answers

**Chapter 1** — What is memory in a neural network? Why does it matter?

**Chapter 2** — How is an RNN different from a regular network? What is a hidden state?

**Chapter 3** — How does the RNN formula actually work? What are the numbers doing?

**Chapter 4** — How does the RNN learn? What is Wy? What is softmax? What is a training loop?

**Chapter 5** — Why does training fail for long sequences? What is the vanishing gradient?

**Chapter 6** — What is the size of every matrix? What is xₜ, hₜ, Wₓ, Wₕ, Wy, b, by?

**Chapter 6b** — Why do we not use one-hot encoding for words? What are embeddings?

**Chapter 7** — Can you show me BPTT with actual numbers from start to finish?

**Chapter 8** — Can you show me every derivative in the chain and why ∂L/∂raw = ŷ - y?

---

## The Bridge to What Comes Next

After completing all chapters above, the learning path continues:

```
RNN (done ✅)
    ↓
LSTM — fixes vanishing gradient with cell state + 3 gates
    ↓
Encoder-Decoder — handles variable length input and output
    ↓
Attention (Bahdanau 2015) — decoder looks at all encoder states
    ↓
Transformer (Vaswani 2017) — removes RNN entirely, attention only
    ↓
GPT / BERT — large scale transformers
```

---

## Key Papers to Read Eventually

| Paper | Year | What it introduced |
|---|---|---|
| Hochreiter & Schmidhuber | 1997 | LSTM |
| Cho et al. | 2014 | Encoder-Decoder (GRU) |
| Bahdanau et al. | 2015 | Attention mechanism |
| Vaswani et al. | 2017 | Transformer — Attention is All You Need |
