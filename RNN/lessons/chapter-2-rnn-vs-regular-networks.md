# Chapter 2: Regular Neural Networks vs RNNs

---

## Regular Neural Network — No Memory

Think of it like a **vending machine**. You press a button (input), it gives you a snack (output). It does not care what you bought yesterday. Every transaction is brand new.

```
Input: "cat"  →  [Network]  →  Output: "animal"
Input: "dog"  →  [Network]  →  Output: "animal"
```

Each word goes in **alone**, with zero memory of what came before.

---

## Recurrent Neural Network — Has Memory

Think of it like a **conversation with a friend**. Your friend remembers everything you said earlier, so their replies make sense in context.

An RNN does this with something called a **hidden state** — think of it as a sticky note the network writes on and carries forward.

```
Step 1:  "The"  → [RNN] → updates sticky note
Step 2:  "dog"  → [RNN + sticky note] → updates sticky note
Step 3:  "ran"  → [RNN + sticky note] → updates sticky note
Step 4:  "fast" → [RNN + sticky note] → OUTPUT
```

Each step the RNN sees the **new word PLUS its memory** from before.

---

## The Core Formula

An RNN runs this one formula over and over at every step:

```
hₜ = tanh( xₜ·Wₓ + hₜ₋₁·Wₕ + b )
```

In plain English:
- Look at the **new input** (xₜ)
- Look at what you **remember so far** (hₜ₋₁)
- Mix them together through weights
- Squish with tanh
- That becomes your **new memory** (hₜ)

---

## Side by Side Comparison

| | Regular Network | RNN |
|---|---|---|
| Memory | None | Hidden state (sticky note) |
| Sees | One input at a time | Input + past memory |
| Good for | Images, single labels | Sequences, text, speech |
| Formula | `output = f(input)` | `hₜ = f(xₜ, hₜ₋₁)` |

---

## What is tanh?

`tanh` squishes any number to stay between **-1 and +1**.

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

- Feed it 1,000,000 → gives you ≈ 1.0
- Feed it -999 → gives you ≈ -1.0
- Feed it 0 → gives you 0

This prevents the hidden state from exploding to huge numbers as it passes through many time steps.

---

## Key Takeaway

> The hidden state is the RNN's memory. It is just a vector of numbers that gets updated at every step using the same formula. Same cell, same weights, just loops over and over.
