# Chapter 8: How Derivatives Flow in BPTT — Complete Guide

---

## What Are These Derivatives?

All derivatives in BPTT are **partial derivatives**.

A partial derivative asks:

```
"If I change ONLY this one variable, keeping everything else fixed,
how much does the output change?"

Written as ∂ (curly d) instead of d.
```

Our goal is to find how the total loss changes with respect to every weight:

```
∂L/∂Wₓ   → how does L change if I change Wₓ slightly?
∂L/∂Wₕ   → how does L change if I change Wₕ slightly?
∂L/∂Wy   → how does L change if I change Wy slightly?
∂L/∂b    → how does L change if I change b slightly?
```

---

## The Chain Rule — One Simple Idea

L does not directly contain Wₓ. There are many steps in between. The chain rule handles this.

If A affects B and B affects C:

```
∂C/∂A  =  ∂C/∂B  ×  ∂B/∂A

How much does A affect C
= how much does B affect C
× how much does A affect B
```

---

## The Computation Graph — What Depends on What

Read this before anything else:

```
Wₓ, x₁, Wₕ, h₀, b
        ↓
        z₁  =  Wₓ·x₁  +  Wₕ·h₀  +  b
        ↓
        h₁  =  tanh(z₁)
       ↙ ↘
      ↙   ↘
(feeds L₁)  (feeds z₂)

FOR L₁:
  Wy, h₁
      ↓
    raw₁  =  Wy·h₁  +  by
      ↓
     ŷ₁   =  softmax(raw₁)
      ↓
     L₁   =  -log(ŷ₁[correct])

FOR z₂:
  Wₓ, x₂, Wₕ, h₁, b
          ↓
          z₂  =  Wₓ·x₂  +  Wₕ·h₁  +  b
          ↓
          h₂  =  tanh(z₂)
          ↓
    Wy, h₂
        ↓
      raw₂  =  Wy·h₂  +  by
        ↓
       ŷ₂   =  softmax(raw₂)
        ↓
       L₂   =  -log(ŷ₂[correct])

TOTAL LOSS:
  L = L₁ + L₂
```

---

## Notation — Two Different Uses of Subscript

**Important:** subscripts mean different things in different contexts.

```
hₜ   → subscript t means TIME STEP     h at step 1, h at step 2
y₀   → subscript 0 means CLASS INDEX   label for class 0 (first character in vocab)
```

For a vocab = {a, b}:

```
index 0 = 'a'
index 1 = 'b'

y₀  = true label for class 'a'    (1 if correct answer is 'a', else 0)
y₁  = true label for class 'b'    (1 if correct answer is 'b', else 0)

ŷ₀  = predicted probability for 'a'
ŷ₁  = predicted probability for 'b'
```

---

## The Loss Function — Cross Entropy

### Full Formula

```
L = - Σ yᵢ · log(ŷᵢ)
      i

Where:
  i    loops over every class in the vocabulary
  yᵢ   is the true label for class i
  ŷᵢ   is the predicted probability for class i
  Σ    means sum over all classes
```

### Why it Simplifies to -log(ŷ_correct)

Because y is a **one-hot vector** — only one position is 1, everything else is 0:

```
y = [0, 1, 0, 0]   correct class is index 1

L = -(y₀·log(ŷ₀)  +  y₁·log(ŷ₁)  +  y₂·log(ŷ₂)  +  y₃·log(ŷ₃))
  = -(0·log(ŷ₀)   +  1·log(ŷ₁)   +  0·log(ŷ₂)   +  0·log(ŷ₃))
  = -log(ŷ₁)
```

Every term dies except the correct class. So:

```
L = -log( ŷ_correct )
```

### Why Negative Log?

```
If model gives correct answer probability 1.0  → L = -log(1.0) = 0.000   perfect
If model gives correct answer probability 0.5  → L = -log(0.5) = 0.693   moderate
If model gives correct answer probability 0.1  → L = -log(0.1) = 2.303   bad
If model gives correct answer probability 0.01 → L = -log(0.01)= 4.605   very bad

High probability of correct answer → low loss  ✅
Low probability of correct answer  → high loss ❌
```

---

## Softmax — What It Is

Raw scores come out of Wy·h. They can be any number including negatives. Softmax converts them into probabilities that sum to 1.

For vocab size 2 with raw scores `raw = [r₀, r₁]`:

```
ŷ₀ = e^r₀ / (e^r₀ + e^r₁)     probability for class 'a'
ŷ₁ = e^r₁ / (e^r₀ + e^r₁)     probability for class 'b'
```

In general for V classes:

```
ŷᵢ = e^rᵢ / (e^r₀ + e^r₁ + ... + e^r_V)
```

Why exponentiate? Raw scores can be negative. `e^x` is always positive no matter what x is:

```
e^(-5)  = 0.007   small but positive ✅
e^(0)   = 1.000   ✅
e^(5)   = 148.4   ✅
```

---

## Why ∂L/∂raw = ŷ - y

This is the starting point of all backpropagation. It is not a shortcut — it is what you get when you differentiate cross entropy through softmax using the chain rule.

### Derivation — Correct Class (index c)

Assume correct class is index 0:

```
L = -log(ŷ₀)
  = -log( e^r₀ / (e^r₀ + e^r₁) )

Using log rules:
  = -log(e^r₀)  +  log(e^r₀ + e^r₁)
  = -r₀         +  log(e^r₀ + e^r₁)

Differentiate with respect to r₀:
  ∂L/∂r₀ = -1  +  e^r₀ / (e^r₀ + e^r₁)
           = -1  +  ŷ₀
           = ŷ₀ - 1
           = ŷ₀ - y₀    ← because y₀ = 1 for correct class
```

### Derivation — Wrong Class (index 1)

r₁ is not the correct class but appears in the softmax denominator:

```
L = -r₀  +  log(e^r₀ + e^r₁)

Differentiate with respect to r₁:
  ∂L/∂r₁ = 0  +  e^r₁ / (e^r₀ + e^r₁)
           = ŷ₁
           = ŷ₁ - 0
           = ŷ₁ - y₁    ← because y₁ = 0 for wrong class
```

### The Pattern

```
Correct class:    ∂L/∂r_c  =  ŷ_c - 1  =  ŷ_c - y_c
Wrong classes:    ∂L/∂r_j  =  ŷ_j - 0  =  ŷ_j - y_j

Both cases are the same formula:

∂L/∂raw  =  ŷ - y
```

### What This Tells You

```
∂L/∂raw = ŷ - y

Correct class:   ŷ_c - 1  → negative  → push this score UP
Wrong classes:   ŷ_j - 0  → positive  → push these scores DOWN
```

---

## Why Full Forward Pass Before Backward Pass?

### Reason 1 — Hidden States Depend on Each Other

```
h₁ depends on h₀
h₂ depends on h₁
h₃ depends on h₂

You cannot compute h₂ until you have h₁.
You cannot go backward from L₂ until h₂ exists.
So you must complete the full forward pass first.
```

### Reason 2 — Gradient of L₂ Needs h₁

```
∂L₂/∂Wₕ = ∂L₂/∂z₂ · h₁ᵀ

To compute how much Wₕ contributed to L₂,
you need h₁ — which only exists after forward pass at t=1.
```

### Reason 3 — Weights Are Shared Across All Time Steps

If you update Wₕ after t=1 before running t=2:

```
h₁ was produced by Wₕ_old
h₂ = Wₓ·x₂ + Wₕ_new·h₁   ← different Wₕ than what produced h₁

The gradient formula for t=2 assumes SAME Wₕ was used throughout.
That assumption is now broken. The math is inconsistent.
```

### Reason 4 — The Sequence is One Training Example

```
"ab" is ONE training example — not two separate examples.
You do not update weights halfway through one training example.
Process it fully, compute total loss, then update.
```

---

## The Complete Chain of Derivatives — Every Path

We trace every path that gradients travel. L = L₁ + L₂ so we compute for each separately then add.

---

### CHAIN FROM L₂

#### PATH 1 — L₂ → raw₂ → Wy

```
Wy → raw₂ → ŷ₂ → L₂

∂L₂/∂Wy  =  ∂L₂/∂raw₂  ×  ∂raw₂/∂Wy

∂L₂/∂raw₂  =  ŷ₂ - y₂              softmax + cross entropy combined
∂raw₂/∂Wy  =  h₂ᵀ                  because raw₂ = Wy·h₂

∂L₂/∂Wy   =  (ŷ₂ - y₂) · h₂ᵀ
```

#### PATH 2 — L₂ → raw₂ → h₂

```
h₂ → raw₂ → ŷ₂ → L₂

∂L₂/∂h₂  =  ∂L₂/∂raw₂  ×  ∂raw₂/∂h₂

∂L₂/∂raw₂  =  ŷ₂ - y₂
∂raw₂/∂h₂  =  Wyᵀ                  because raw₂ = Wy·h₂

∂L₂/∂h₂   =  Wyᵀ · (ŷ₂ - y₂)
```

#### PATH 3 — L₂ → raw₂ → h₂ → z₂

```
z₂ → h₂ → raw₂ → L₂

∂L₂/∂z₂  =  ∂L₂/∂h₂  ×  ∂h₂/∂z₂

∂L₂/∂h₂   =  Wyᵀ · (ŷ₂ - y₂)     already computed above
∂h₂/∂z₂   =  (1 - h₂²)            derivative of tanh

∂L₂/∂z₂  =  ∂L₂/∂h₂  ⊙  (1 - h₂²)
```

#### PATH 4 — L₂ → ... → z₂ → Wₓ

```
Wₓ → z₂ → h₂ → ... → L₂

∂L₂/∂Wₓ(via z₂)  =  ∂L₂/∂z₂  ×  ∂z₂/∂Wₓ

∂z₂/∂Wₓ   =  x₂ᵀ                  because z₂ = Wₓ·x₂ + ...

∂L₂/∂Wₓ(via z₂)  =  ∂L₂/∂z₂ · x₂ᵀ
```

#### PATH 5 — L₂ → ... → z₂ → Wₕ

```
Wₕ → z₂ → h₂ → ... → L₂

∂L₂/∂Wₕ(via z₂)  =  ∂L₂/∂z₂  ×  ∂z₂/∂Wₕ

∂z₂/∂Wₕ   =  h₁ᵀ                  because z₂ = ... + Wₕ·h₁ + ...

∂L₂/∂Wₕ(via z₂)  =  ∂L₂/∂z₂ · h₁ᵀ
```

#### PATH 6 — L₂ → ... → z₂ → b

```
b → z₂ → h₂ → ... → L₂

∂L₂/∂b(via z₂)  =  ∂L₂/∂z₂  ×  ∂z₂/∂b

∂z₂/∂b   =  1                      because z₂ = ... + b, derivative of b is 1

∂L₂/∂b(via z₂)  =  ∂L₂/∂z₂ · 1  =  ∂L₂/∂z₂
```

#### PATH 7 — GRADIENT CROSSES TIME BOUNDARY — L₂ → ... → z₂ → h₁

```
h₁ → z₂ → h₂ → ... → L₂

∂L₂/∂h₁  =  ∂L₂/∂z₂  ×  ∂z₂/∂h₁

∂z₂/∂h₁  =  Wₕᵀ                   because z₂ = ... + Wₕ·h₁ + ...

∂L₂/∂h₁  =  Wₕᵀ · ∂L₂/∂z₂
```

This gradient just traveled back in time from t=2 to t=1 through Wₕᵀ.

#### PATH 8 — L₂ → ... → h₁ → z₁

```
z₁ → h₁ → z₂ → h₂ → ... → L₂

∂L₂/∂z₁  =  ∂L₂/∂h₁  ×  ∂h₁/∂z₁

∂h₁/∂z₁  =  (1 - h₁²)             derivative of tanh at t=1

∂L₂/∂z₁  =  ∂L₂/∂h₁  ⊙  (1 - h₁²)
```

FULL EXPANSION — every link written out:

```
∂L₂/∂z₁  =  ∂L₂/∂z₂  ·  ∂z₂/∂h₁  ·  ∂h₁/∂z₁

           =  [ Wyᵀ·(ŷ₂-y₂) ⊙ (1-h₂²) ]  ·  Wₕᵀ  ·  (1-h₁²)
                ↑                              ↑         ↑
            gradient at z₂              crosses     tanh at
            (computed at t=2)           time        t=1
                                        boundary
```

This is the heart of BPTT — the gradient from L₂ has traveled all the way back through:
t=2 output → h₂ → z₂ → h₁ → z₁

---

#### PATH 9 — L₂ → ... → z₁ → Wₓ (at t=1)

```
Wₓ → z₁ → h₁ → z₂ → ... → L₂

∂L₂/∂Wₓ(via z₁)  =  ∂L₂/∂z₁ · x₁ᵀ
```

FULL EXPANSION — every link written out:

```
∂L₂/∂Wₓ(via z₁)  =  ∂L₂/∂z₂  ·  ∂z₂/∂h₁  ·  ∂h₁/∂z₁  ·  ∂z₁/∂Wₓ

                   =  [ Wyᵀ·(ŷ₂-y₂) ⊙ (1-h₂²) ]  ·  Wₕᵀ  ·  (1-h₁²)  ·  x₁ᵀ
                        ↑                              ↑         ↑           ↑
                    gradient at z₂              crosses     tanh at      current
                    (from t=2)                  time        t=1          input
                                                boundary                 at t=1
```

---

#### PATH 10 — L₂ → ... → z₁ → Wₕ (at t=1)

```
Wₕ → z₁ → h₁ → z₂ → ... → L₂

∂L₂/∂Wₕ(via z₁)  =  ∂L₂/∂z₁ · h₀ᵀ
```

FULL EXPANSION — every link written out:

```
∂L₂/∂Wₕ(via z₁)  =  ∂L₂/∂z₂  ·  ∂z₂/∂h₁  ·  ∂h₁/∂z₁  ·  ∂z₁/∂Wₕ

                   =  [ Wyᵀ·(ŷ₂-y₂) ⊙ (1-h₂²) ]  ·  Wₕᵀ  ·  (1-h₁²)  ·  h₀ᵀ
                        ↑                              ↑         ↑           ↑
                    gradient at z₂              crosses     tanh at      previous
                    (from t=2)                  time        t=1          hidden state
                                                boundary                 at t=1 (h₀)
```

---

#### PATH 11 — L₂ → ... → z₁ → b (at t=1)

```
b → z₁ → h₁ → z₂ → ... → L₂

∂L₂/∂b(via z₁)  =  ∂L₂/∂z₁ · 1  =  ∂L₂/∂z₁
```

FULL EXPANSION — every link written out:

```
∂L₂/∂b(via z₁)  =  ∂L₂/∂z₂  ·  ∂z₂/∂h₁  ·  ∂h₁/∂z₁  ·  ∂z₁/∂b

                 =  [ Wyᵀ·(ŷ₂-y₂) ⊙ (1-h₂²) ]  ·  Wₕᵀ  ·  (1-h₁²)  ·  1
                      ↑                              ↑         ↑           ↑
                  gradient at z₂              crosses     tanh at      bias derivative
                  (from t=2)                  time        t=1          is always 1
                                              boundary
```

---

## THE FULL BPTT CHAIN — Everything Expanded

This is the complete picture. Every derivative written from scratch for every weight, showing exactly how the gradient traveled from the loss all the way back to each weight.

```
STARTING POINT (same for all paths):

∂L₂/∂raw₂  =  ŷ₂ - y₂
               ↑
               this is where every chain begins


AT t=2 (gradients that stay at t=2):

∂L₂/∂Wy         =  (ŷ₂-y₂)  ·  h₂ᵀ

∂L₂/∂Wₓ(t=2)   =  (ŷ₂-y₂)  ·  Wyᵀ  ·  (1-h₂²)  ·  x₂ᵀ

∂L₂/∂Wₕ(t=2)   =  (ŷ₂-y₂)  ·  Wyᵀ  ·  (1-h₂²)  ·  h₁ᵀ

∂L₂/∂b(t=2)    =  (ŷ₂-y₂)  ·  Wyᵀ  ·  (1-h₂²)  ·  1


CROSSES TIME BOUNDARY (gradient travels from t=2 back to t=1):

∂L₂/∂h₁        =  (ŷ₂-y₂)  ·  Wyᵀ  ·  (1-h₂²)  ·  Wₕᵀ
                    ↑            ↑         ↑           ↑
                 start        through    through    crosses
                              Wy         tanh       time


AT t=1 (gradients that traveled back through time):

∂L₂/∂Wₓ(t=1)   =  (ŷ₂-y₂)  ·  Wyᵀ  ·  (1-h₂²)  ·  Wₕᵀ  ·  (1-h₁²)  ·  x₁ᵀ
                                                        ↑         ↑
                                                    crosses    tanh at
                                                    time       t=1

∂L₂/∂Wₕ(t=1)   =  (ŷ₂-y₂)  ·  Wyᵀ  ·  (1-h₂²)  ·  Wₕᵀ  ·  (1-h₁²)  ·  h₀ᵀ

∂L₂/∂b(t=1)    =  (ŷ₂-y₂)  ·  Wyᵀ  ·  (1-h₂²)  ·  Wₕᵀ  ·  (1-h₁²)  ·  1
```

Reading left to right shows the full journey of the gradient:

```
ŷ₂-y₂  →  ×Wyᵀ  →  ×(1-h₂²)  →  ×Wₕᵀ  →  ×(1-h₁²)  →  ×xᵀ or ×hᵀ or ×1
  ↑           ↑          ↑            ↑           ↑              ↑
start      through    through      crosses     through        branch to
           output     tanh         time        tanh           each weight
           layer      at t=2       boundary    at t=1
```

Every extra time step going back adds one more `·  Wₕᵀ  ·  (1-hᵀ²)` to the chain.
That is exactly why gradients vanish — each time step multiplies by another number ≤ 1.

---

### CHAIN FROM L₁

L₁ only flows through t=1. It does not travel back in time because there is no earlier step.

#### PATH A — L₁ → raw₁ → Wy

```
∂L₁/∂Wy  =  (ŷ₁ - y₁) · h₁ᵀ
```

#### PATH B — L₁ → raw₁ → h₁

```
∂L₁/∂h₁  =  Wyᵀ · (ŷ₁ - y₁)
```

#### PATH C — L₁ → raw₁ → h₁ → z₁

```
∂L₁/∂z₁  =  ∂L₁/∂h₁  ⊙  (1 - h₁²)
```

#### PATH D — L₁ → ... → z₁ → Wₓ

```
∂L₁/∂Wₓ  =  ∂L₁/∂z₁ · x₁ᵀ
```

#### PATH E — L₁ → ... → z₁ → Wₕ

```
∂L₁/∂Wₕ  =  ∂L₁/∂z₁ · h₀ᵀ
```

#### PATH F — L₁ → ... → z₁ → b

```
∂L₁/∂b   =  ∂L₁/∂z₁ · 1  =  ∂L₁/∂z₁
```

---

## ∂z is the Hub — Everything Branches From It

Every time you arrive at ∂L/∂z at any time step, four things branch out:

```
∂L/∂zₜ
    │
    ├── × xₜᵀ       →  ∂L/∂Wₓ      input weight gradient
    │
    ├── × hₜ₋₁ᵀ    →  ∂L/∂Wₕ      hidden weight gradient
    │
    ├── × 1         →  ∂L/∂b       bias gradient
    │
    └── × Wₕᵀ      →  ∂L/∂hₜ₋₁   crosses time boundary
                                    → becomes ∂L/∂zₜ₋₁ after × (1 - hₜ₋₁²)
                                    → branches again at previous step
```

---

## Final Step — Add Everything Up

Because L = L₁ + L₂, every weight receives gradients from both losses:

```
∂L/∂Wy  =  ∂L₁/∂Wy               (PATH A)
          + ∂L₂/∂Wy               (PATH 1)

∂L/∂Wₓ  =  ∂L₁/∂Wₓ              (PATH D  — at t=1 from L₁)
          + ∂L₂/∂Wₓ(via z₂)      (PATH 4  — at t=2 from L₂)
          + ∂L₂/∂Wₓ(via z₁)      (PATH 9  — at t=1 from L₂ traveling back)

∂L/∂Wₕ  =  ∂L₁/∂Wₕ              (PATH E)
          + ∂L₂/∂Wₕ(via z₂)      (PATH 5)
          + ∂L₂/∂Wₕ(via z₁)      (PATH 10)

∂L/∂b   =  ∂L₁/∂b                (PATH F)
          + ∂L₂/∂b(via z₂)       (PATH 6)
          + ∂L₂/∂b(via z₁)       (PATH 11)
```

---

## How the Gradient Shrinks Going Back Through Time

Every time gradient crosses a time boundary it passes through two things:

```
1. Wₕᵀ           → can amplify or shrink
2. (1 - h²)      → tanh derivative, always between 0 and 1, always shrinks
```

So the gradient arriving at h₁ from L₂ is always smaller than the direct gradient at h₂:

```
At output t=2:              ŷ - y             (large, direct signal)
After Wyᵀ (at h₂):         smaller
After (1-h₂²) (at z₂):     smaller again     (tanh shrinks it)
After Wₕᵀ (arrives at h₁): smaller again     (crossed time boundary)
After (1-h₁²) (at z₁):     smaller again     (tanh shrinks it again)
```

With 10 time steps this happens 9 times going back. The gradient becomes nearly zero. That is the vanishing gradient problem.

---

## Key Rules to Remember

```
1. ∂L/∂raw  =  ŷ - y
   Always the starting point. Comes from combining cross entropy + softmax.

2. ∂L/∂h    =  Wyᵀ · ∂L/∂raw
   Gradient flows from output back to hidden state through Wyᵀ.

3. ∂L/∂z    =  ∂L/∂h  ⊙  (1 - h²)
   Gradient passes through tanh derivative. Always shrinks.

4. ∂L/∂Wₓ  =  ∂L/∂z · xᵀ
   Input weight gradient = z gradient × current input.

5. ∂L/∂Wₕ  =  ∂L/∂z · h_prevᵀ
   Hidden weight gradient = z gradient × previous hidden state.

6. ∂L/∂b   =  ∂L/∂z · 1  =  ∂L/∂z
   Bias gradient = z gradient directly. No multiplication needed.

7. ∂L/∂h_prev  =  Wₕᵀ · ∂L/∂z
   This is what crosses the time boundary. Gradient travels back one step.

8. Each weight accumulates gradients from ALL time steps it was used.
   Update only happens AFTER all gradients are collected.
```

---

## Weight Update — Creating the New Weights

Once all gradients are accumulated across all time steps, every weight matrix is updated using the same formula:

```
W_new  =  W_old  -  lr  ×  ∂L/∂W

Where:
  W_old   is the current weight value
  lr      is the learning rate (a small number you choose, e.g. 0.01)
  ∂L/∂W  is the accumulated gradient for that weight
  W_new   is the updated weight
```

### Applied to Every Weight

```
Wy_new  =  Wy_old  -  lr  ×  ∂L/∂Wy
Wₓ_new  =  Wₓ_old  -  lr  ×  ∂L/∂Wₓ
Wₕ_new  =  Wₕ_old  -  lr  ×  ∂L/∂Wₕ
b_new   =  b_old   -  lr  ×  ∂L/∂b
by_new  =  by_old  -  lr  ×  ∂L/∂by
```

All weights are updated **simultaneously** after all gradients are computed. Never one at a time.

---

### Why Subtract?

The gradient tells you which direction increases the loss.
You want to DECREASE the loss. So you go in the OPPOSITE direction.

```
Gradient positive  →  weight is pushing loss UP   →  subtract  →  weight goes DOWN
Gradient negative  →  weight is pushing loss DOWN →  subtract  →  weight goes UP
```

---

### Why Multiply by Learning Rate?

The gradient tells you the direction but not how big a step to take.

```
lr too large  →  you overshoot the minimum  →  loss bounces around, never settles
lr too small  →  you take tiny steps        →  learning takes forever
lr just right →  steady progress toward minimum
```

```
Loss
│
│ ×        ← lr too large, bouncing
│   ×  ×
│       ×
│──────────────────────→ steps

Loss
│
│×
│ ×
│  ×
│   ×
│    × × × × ×──────→   ← lr just right, steadily decreasing
```

Typical values: `lr = 0.01` or `lr = 0.001`

---

### Concrete Example From Our Network

Using lr = 0.1 and the gradients computed earlier:

```
BEFORE UPDATE:
  Wy = [[ 0.4,  0.3],
        [ 0.2,  0.6]]

Gradient:
  ∂L/∂Wy = [[-0.0104, -0.1469],
             [ 0.0104,  0.1469]]

UPDATE:
  Wy_new = Wy_old - 0.1 × ∂L/∂Wy

         = [[ 0.4,  0.3],    -   0.1 × [[-0.0104, -0.1469],
            [ 0.2,  0.6]]               [ 0.0104,  0.1469]]

         = [[ 0.4 - (-0.0010),   0.3 - (-0.0147)],
            [ 0.2 - ( 0.0010),   0.6 - ( 0.0147)]]

         = [[ 0.4010,  0.3147],
            [ 0.1990,  0.5853]]

Row 0 (weights for 'a') got slightly larger  → model will predict 'a' with higher score next time
Row 1 (weights for 'b') got slightly smaller → model will predict 'b' with lower score next time
```

---

### The Full Training Loop in One Picture

```
INITIALISE all weights randomly
         │
         ▼
┌─────────────────────────────────────────┐
│                                         │
│  FORWARD PASS                           │
│  x₁ → z₁ → h₁ → ŷ₁ → L₁             │
│  x₂ → z₂ → h₂ → ŷ₂ → L₂             │
│  L = L₁ + L₂                           │
│         │                               │
│  BACKWARD PASS (BPTT)                   │
│  compute ∂L/∂Wy                        │
│  compute ∂L/∂Wₓ  (all time steps)     │
│  compute ∂L/∂Wₕ  (all time steps)     │
│  compute ∂L/∂b   (all time steps)      │
│         │                               │
│  WEIGHT UPDATE                          │
│  Wy  = Wy  - lr × ∂L/∂Wy             │
│  Wₓ  = Wₓ  - lr × ∂L/∂Wₓ            │
│  Wₕ  = Wₕ  - lr × ∂L/∂Wₕ            │
│  b   = b   - lr × ∂L/∂b               │
│                                         │
│  repeat with next batch ────────────────┘
│
▼
after thousands of iterations → weights converge → loss is small → trained RNN ✅
```
