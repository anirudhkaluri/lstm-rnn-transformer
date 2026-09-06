# Transformer Course — Index

Builds on: RNN (9 chapters) → LSTM (6 chapters). You know h_t, c_t, gates, BPTT, vanishing gradients, embeddings.

| # | Chapter | What it covers |
|---|---|---|
| 1 | The Sequential Bottleneck | Why even LSTMs aren't enough — recurrence itself is the problem |
| 2 | Self-Attention — The Core Idea | Query/Key/Value intuition, tiny numeric example |
| 3 | Self-Attention — The Full Math | The formula softmax(QKᵀ/√d_k)V, dimension tracking |
| 4 | Multi-Head Attention | Why multiple heads, splitting and recombining dimensions |
| 5 | Positional Encoding — Problem and Fix | Why self-attention can't see word order; e_t + PE_t = x_t |
| 5b | Sine, Cosine, and Why Circles | Unit circle from scratch; why circles (not alternatives) encode position |
| 5c | Decoding the PE Formula | PE(pos,2i)=sin(...), PE(pos,2i+1)=cos(...) — every index explained, worked example |
| 6 | The Transformer Block | Residual connections, LayerNorm, feed-forward sublayer |
| 7 | Encoder vs Decoder | Stacking blocks, causal masking, cross-attention |
| 8 | Full Forward Pass | End-to-end tiny example with real numbers |
| 9 | Training | Loss, teacher forcing, why gradients flow well through attention |
| 10 | The Modern Landscape | GPT vs BERT vs T5 — decoder-only, encoder-only, encoder-decoder |

Status: Chapters 1–6 saved (including 5b, 5c).
