# Chapter 5: The Vanishing Gradient Problem

---

## Quick Recap — What is a Gradient?

The gradient tells us how much a particular weight is responsible for the mistake. The bigger the gradient, the more we adjust the weight, and the faster we learn.

---

## The Game of Telephone

Remember the game of telephone as a kid?

```
Person 1 → Person 2 → Person 3 → Person 4 → Person 5
"I have a red dog"                           "I have a dead frog"
```

Every time the message passes through a person it gets a little distorted. By person 10 it is completely unrecognizable.

BPTT has the exact same problem — but with gradients.

---

## What Actually Happens

When we backpropagate through time, the gradient travels backwards through every single time step:

```
FORWARD:
  h₁ → h₂ → h₃ → h₄ → h₅ → h₆ → h₇ → h₈ → LOSS

BACKWARD:
  LOSS → h₈ → h₇ → h₆ → h₅ → h₄ → h₃ → h₂ → h₁
```

At each step the gradient gets **multiplied by the slope of tanh**. The slope of tanh is always less than or equal to 1 (maximum slope = 1 at zero, typical slope = 0.25).

```
At h₈:   gradient = 1.000
At h₇:   gradient = 1.000 × 0.25 = 0.250
At h₆:   gradient = 0.250 × 0.25 = 0.062
At h₅:   gradient = 0.062 × 0.25 = 0.015
At h₄:   gradient = 0.004
At h₃:   gradient = 0.001
At h₂:   gradient = 0.0002
At h₁:   gradient = 0.00006  ← basically ZERO
```

By the time we reach the early time steps, the gradient has **vanished**. Those early weights learn nothing.

---

## Why This is Catastrophic

Imagine trying to complete this sentence:

> "The cats that were sitting on the warm sunny roof were meowing loudly because they ____"

The answer is "were hungry" — and it depends on "cats" way back at the start.

A regular RNN would forget "cats" ever existed by the time it reaches "they". The gradient from the end of the sentence simply never makes it back to the word "cats".

---

## The Exploding Gradient — The Opposite Problem

If weights are slightly bigger than 1, gradients explode instead of vanish:

```
At h₈:   gradient = 1
At h₇:   gradient = 2
At h₆:   gradient = 4
At h₅:   gradient = 8
At h₄:   gradient = 16
At h₃:   gradient = 32
At h₁:   gradient = 128  ← EXPLOSION
```

The weights swing wildly and the network learns garbage.

---

## The Two Villains

| | Vanishing Gradient | Exploding Gradient |
|---|---|---|
| Gradient becomes | → 0 | → ∞ |
| Network effect | Forgets early inputs | Goes completely crazy |
| Fix | LSTM / GRU | Gradient Clipping |

---

## The Fix

This problem was so bad it nearly killed RNN research in the 1990s. Then in 1997, two researchers named **Hochreiter & Schmidhuber** invented a smarter memory cell called **LSTM** that completely solved the vanishing gradient problem.

That is Lesson 6.

---

## Key Takeaway

> Every step backwards multiplies the gradient by a number less than 1. After many steps the gradient becomes essentially zero. Early time steps stop learning. This is the vanishing gradient problem — the core weakness of plain RNNs.
