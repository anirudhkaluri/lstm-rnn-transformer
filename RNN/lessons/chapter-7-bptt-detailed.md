# Chapter 7: Backpropagation Through Time — Full Worked Example

---

## Setup — Everything Defined Upfront

### Network Sizes

```
vocab_size  (V) = 4       characters: h=0, e=1, l=2, o=3
hidden_size (H) = 2       two neurons in the RNN cell
time_steps  (T) = 2       we read "he", predict "el"
```

### All Matrices and Their Sizes

| Symbol | Size | What it is |
|---|---|---|
| `xₜ` | 4×1 | one-hot input at step t |
| `yₜ` | 4×1 | one-hot target at step t |
| `hₜ` | 2×1 | hidden state at step t |
| `h₀` | 2×1 | initial hidden state — all zeros |
| `zₜ` | 2×1 | pre-activation (before tanh) |
| `Wₓ` | 2×4 | input weight matrix |
| `Wₕ` | 2×2 | hidden weight matrix |
| `Wy` | 4×2 | output weight matrix |
| `b`  | 2×1 | hidden bias |
| `by` | 4×1 | output bias |
| `rawₜ`| 4×1 | raw scores before softmax |
| `ŷₜ` | 4×1 | predicted probabilities (after softmax) |
| `Lₜ` | scalar | loss at step t |
| `L`  | scalar | total loss = L1 + L2 |

---

### Initial Weight Values (chosen to keep math clean)

```
Wₓ = [[ 0.5,  0.2,  0.1,  0.1],    (2×4)
      [ 0.1,  0.5,  0.2,  0.1]]

Wₕ = [[ 0.3,  0.1],                 (2×2)
      [ 0.1,  0.3]]

Wy = [[ 0.1,  0.1],                 (4×2)
      [ 0.3,  0.1],
      [ 0.1,  0.3],
      [ 0.1,  0.1]]

b  = [0.0, 0.0]                      (2×1)
by = [0.0, 0.0, 0.0, 0.0]           (4×1)
h₀ = [0.0, 0.0]                      (2×1)
```

---

### The Sequence

```
t=1:  input x₁ = h = [1,0,0,0]    target y₁ = e = [0,1,0,0]
t=2:  input x₂ = e = [0,1,0,0]    target y₂ = l = [0,0,1,0]
```

---

## PART A — FORWARD PASS

### Time Step t=1 — Read "h", Predict "e"

**Step 1: Compute z₁**

```
z₁ = Wₓ·x₁  +  Wₕ·h₀  +  b

Wₓ·x₁:
  Row 1 of Wₓ = [0.5, 0.2, 0.1, 0.1]  ·  [1,0,0,0]  =  0.5
  Row 2 of Wₓ = [0.1, 0.5, 0.2, 0.1]  ·  [1,0,0,0]  =  0.1
  → Wₓ·x₁ = [0.5, 0.1]

Wₕ·h₀:
  h₀ = [0, 0]  →  Wₕ·h₀ = [0, 0]

z₁ = [0.5, 0.1] + [0, 0] + [0, 0] = [0.5, 0.1]
```

**Step 2: Compute h₁ = tanh(z₁)**

```
h₁ = [tanh(0.5), tanh(0.1)]
   = [0.4621,    0.0997]
```

**Step 3: Compute raw scores → raw₁ = Wy·h₁ + by**

```
Wy·h₁:
  Row 1: [0.1, 0.1] · [0.4621, 0.0997] = 0.0462 + 0.0100 = 0.0562   ← score for 'h'
  Row 2: [0.3, 0.1] · [0.4621, 0.0997] = 0.1386 + 0.0100 = 0.1486   ← score for 'e'
  Row 3: [0.1, 0.3] · [0.4621, 0.0997] = 0.0462 + 0.0299 = 0.0761   ← score for 'l'
  Row 4: [0.1, 0.1] · [0.4621, 0.0997] = 0.0462 + 0.0100 = 0.0562   ← score for 'o'

raw₁ = [0.0562, 0.1486, 0.0761, 0.0562]
```

**Step 4: Softmax → ŷ₁**

```
softmax(rawᵢ) = exp(rawᵢ) / Σ exp(rawⱼ)

exp([0.0562, 0.1486, 0.0761, 0.0562])
= [1.0578,  1.1603,  1.0791,  1.0578]

sum = 4.3550

ŷ₁ = [1.0578/4.3550, 1.1603/4.3550, 1.0791/4.3550, 1.0578/4.3550]
   = [0.2429,        0.2665,        0.2478,        0.2429]
```

**Step 5: Cross Entropy Loss L₁**

```
Target y₁ = [0, 1, 0, 0]  →  correct class is 'e' (index 1)
L₁ = -log(ŷ₁[1]) = -log(0.2665) = 1.3228
```

---

### Time Step t=2 — Read "e", Predict "l"

**Step 1: Compute z₂**

```
z₂ = Wₓ·x₂  +  Wₕ·h₁  +  b

Wₓ·x₂  (x₂ = [0,1,0,0]):
  Row 1: [0.5, 0.2, 0.1, 0.1] · [0,1,0,0] = 0.2
  Row 2: [0.1, 0.5, 0.2, 0.1] · [0,1,0,0] = 0.5
  → Wₓ·x₂ = [0.2, 0.5]

Wₕ·h₁  (h₁ = [0.4621, 0.0997]):
  Row 1: [0.3, 0.1] · [0.4621, 0.0997] = 0.1386 + 0.0100 = 0.1486
  Row 2: [0.1, 0.3] · [0.4621, 0.0997] = 0.0462 + 0.0299 = 0.0761
  → Wₕ·h₁ = [0.1486, 0.0761]

z₂ = [0.2 + 0.1486, 0.5 + 0.0761] = [0.3486, 0.5761]
```

**Step 2: Compute h₂ = tanh(z₂)**

```
h₂ = [tanh(0.3486), tanh(0.5761)]
   = [0.3364,       0.5194]

Notice: h₂ carries memory of BOTH "h" (from h₁) AND "e" (from x₂)
```

**Step 3: Compute raw₂ = Wy·h₂ + by**

```
Row 1: [0.1, 0.1] · [0.3364, 0.5194] = 0.0336 + 0.0519 = 0.0855   ← 'h'
Row 2: [0.3, 0.1] · [0.3364, 0.5194] = 0.1009 + 0.0519 = 0.1528   ← 'e'
Row 3: [0.1, 0.3] · [0.3364, 0.5194] = 0.0336 + 0.1558 = 0.1894   ← 'l'
Row 4: [0.1, 0.1] · [0.3364, 0.5194] = 0.0336 + 0.0519 = 0.0855   ← 'o'

raw₂ = [0.0855, 0.1528, 0.1894, 0.0855]
```

**Step 4: Softmax → ŷ₂**

```
exp([0.0855, 0.1528, 0.1894, 0.0855])
= [1.0893, 1.1651, 1.2087, 1.0893]

sum = 4.5524

ŷ₂ = [0.2393, 0.2559, 0.2655, 0.2393]
```

**Step 5: Cross Entropy Loss L₂**

```
Target y₂ = [0, 0, 1, 0]  →  correct class is 'l' (index 2)
L₂ = -log(ŷ₂[2]) = -log(0.2655) = 1.3266
```

---

### Total Loss

```
L = L₁ + L₂ = 1.3228 + 1.3266 = 2.6494
```

---

### Forward Pass Summary

```
                    x₁=[1,0,0,0]           x₂=[0,1,0,0]
                        ↓                       ↓
h₀=[0,0] ──→ [ RNN Cell ] ──h₁=[0.46,0.10]──→ [ RNN Cell ] ──h₂=[0.34,0.52]
                  ↓                                 ↓
              raw₁=[0.056,               raw₂=[0.086,
                    0.149,                     0.153,
                    0.076,                     0.189,
                    0.056]                     0.086]
                  ↓                                 ↓
              ŷ₁=[0.243,                ŷ₂=[0.239,
                  0.267,  ← 'e'             0.256,
                  0.248,                    0.266, ← 'l'
                  0.243]                    0.239]
                  ↓                                 ↓
              L₁=1.3228                    L₂=1.3266

                              L = 2.6494
```

---

## PART B — BACKWARD PASS (BPTT)

The goal is to compute how much each weight contributed to the total loss L.

We go **right to left** — start at t=2, then go back to t=1.

---

### Key Formula — Cross Entropy + Softmax Gradient

When you combine cross entropy loss and softmax, the gradient simplifies to:

```
dL/draw = ŷ - y

Predicted probabilities minus the one-hot target.
This is the starting point of all gradients.
```

---

### KEY CONCEPT — Two Sources of Gradient at h₁

This is the most important thing to understand in BPTT:

```
h₁ contributes to the loss in TWO ways:

1. DIRECTLY  → h₁ feeds into Wy at t=1 → produces ŷ₁ → L₁
2. INDIRECTLY → h₁ feeds into the RNN cell at t=2 → produces h₂ → ŷ₂ → L₂

So the total gradient at h₁ = direct gradient + gradient flowing back from t=2
```

This is why it is called Backpropagation THROUGH TIME — the gradient at early steps receives contributions from ALL future time steps.

---

### AT t=2 — Start Here

**B1: Gradient of L₂ w.r.t. raw scores (dL₂/draw₂)**

```
dL₂/draw₂ = ŷ₂ - y₂
           = [0.2393, 0.2559, 0.2655, 0.2393]
           - [0.0000, 0.0000, 1.0000, 0.0000]
           = [0.2393, 0.2559, -0.7345, 0.2393]   (4×1)

Intuition: 'l' got probability 0.2655 but should have gotten 1.0
           So the gradient for 'l' is -0.7345 (big negative → push score up)
           All others got some probability but should have gotten 0
           So their gradients are positive (push those scores down)
```

**B2: Gradient of L₂ w.r.t. Wy (dL₂/dWy)**

```
dL₂/dWy = dL₂/draw₂  ·  h₂ᵀ          ← outer product (4×1)·(1×2) = (4×2)

= [ 0.2393]                           [0.2393×0.3364, 0.2393×0.5194]
  [ 0.2559]  ·  [0.3364, 0.5194]  =  [0.2559×0.3364, 0.2559×0.5194]
  [-0.7345]                           [-0.7345×0.3364, -0.7345×0.5194]
  [ 0.2393]                           [0.2393×0.3364, 0.2393×0.5194]

dL₂/dWy = [[ 0.0805,  0.1243],
            [ 0.0861,  0.1329],
            [-0.2471, -0.3815],
            [ 0.0805,  0.1243]]   (4×2)
```

**B3: Gradient of L₂ w.r.t. h₂ (dL₂/dh₂)**

```
dL₂/dh₂ = Wyᵀ  ·  dL₂/draw₂          (2×4)·(4×1) = (2×1)

Wyᵀ = [[0.1, 0.3, 0.1, 0.1],
        [0.1, 0.1, 0.3, 0.1]]

= [0.1×0.2393 + 0.3×0.2559 + 0.1×(-0.7345) + 0.1×0.2393]
  [0.1×0.2393 + 0.1×0.2559 + 0.3×(-0.7345) + 0.1×0.2393]

= [0.0239 + 0.0768 - 0.0735 + 0.0239]
  [0.0239 + 0.0256 - 0.2204 + 0.0239]

dL₂/dh₂ = [0.0511, -0.1470]   (2×1)
```

**B4: Gradient through tanh at t=2 (dL₂/dz₂)**

```
tanh derivative:  d/dz [tanh(z)] = 1 - tanh²(z) = 1 - h²

1 - h₂² = [1 - 0.3364², 1 - 0.5194²]
         = [1 - 0.1132,  1 - 0.2698]
         = [0.8868,       0.7302]

dL₂/dz₂ = dL₂/dh₂  ⊙  (1 - h₂²)     ← element-wise multiplication

= [0.0511 × 0.8868,  -0.1470 × 0.7302]
= [0.0453,           -0.1073]   (2×1)

This is the gradient BEFORE the tanh squishing at t=2.
Notice it shrank slightly because (1-h²) ≤ 1 always.
```

**B5: Gradient of L₂ w.r.t. Wₓ at t=2 (dL₂/dWₓ)**

```
dL₂/dWₓ = dL₂/dz₂  ·  x₂ᵀ            (2×1)·(1×4) = (2×4)

= [ 0.0453]  ·  [0, 1, 0, 0]
  [-0.1073]

= [[ 0,      0.0453, 0, 0],
   [ 0,     -0.1073, 0, 0]]   (2×4)

Only column 1 (for 'e') is non-zero because x₂ is one-hot for 'e'.
The gradient only flows to weights connected to the current input.
```

**B6: Gradient of L₂ w.r.t. Wₕ at t=2 (dL₂/dWₕ)**

```
dL₂/dWₕ = dL₂/dz₂  ·  h₁ᵀ            (2×1)·(1×2) = (2×2)

= [ 0.0453]  ·  [0.4621, 0.0997]
  [-0.1073]

= [[ 0.0453×0.4621,  0.0453×0.0997],
   [-0.1073×0.4621, -0.1073×0.0997]]

= [[ 0.0209,  0.0045],
   [-0.0496, -0.0107]]   (2×2)
```

**B7: Gradient flows back to h₁ from t=2 (dL₂/dh₁)**

```
dL₂/dh₁ = Wₕᵀ  ·  dL₂/dz₂            (2×2)·(2×1) = (2×1)

Wₕᵀ = [[0.3, 0.1],
        [0.1, 0.3]]

= [0.3×0.0453 + 0.1×(-0.1073)]
  [0.1×0.0453 + 0.3×(-0.1073)]

= [0.0136 - 0.0107]
  [0.0045 - 0.0322]

dL₂/dh₁ = [0.0029, -0.0277]   (2×1)

This gradient has now TRAVELED BACK THROUGH TIME from t=2 to t=1.
It is the blame that t=2's mistake places on h₁.
Notice it is already small because it passed through Wₕ and tanh.
```

---

### AT t=1 — Going Further Back

At h₁, gradient arrives from TWO sources:

```
SOURCE 1: Direct from L₁ (h₁ → Wy → ŷ₁ → L₁)
SOURCE 2: From t=2 flowing back through time (computed above = [0.0029, -0.0277])
```

**B8: Direct gradient from L₁ through Wy (dL₁/dh₁)**

```
dL₁/draw₁ = ŷ₁ - y₁
           = [0.2429, 0.2665, 0.2478, 0.2429]
           - [0.0000, 1.0000, 0.0000, 0.0000]
           = [0.2429, -0.7335, 0.2478, 0.2429]

dL₁/dh₁ = Wyᵀ  ·  dL₁/draw₁

= [0.1×0.2429 + 0.3×(-0.7335) + 0.1×0.2478 + 0.1×0.2429]
  [0.1×0.2429 + 0.1×(-0.7335) + 0.3×0.2478 + 0.1×0.2429]

= [0.0243 - 0.2201 + 0.0248 + 0.0243]
  [0.0243 - 0.0734 + 0.0743 + 0.0243]

dL₁/dh₁_direct = [-0.1467, 0.0495]   (2×1)
```

**B9: Total gradient at h₁**

```
dL/dh₁ = dL₁/dh₁_direct  +  dL₂/dh₁_from_t2

        = [-0.1467,  0.0495]
        + [ 0.0029, -0.0277]

        = [-0.1438,  0.0218]   (2×1)
```

**B10: Gradient through tanh at t=1 (dL/dz₁)**

```
1 - h₁² = [1 - 0.4621², 1 - 0.0997²]
         = [1 - 0.2135,  1 - 0.0099]
         = [0.7865,       0.9901]

dL/dz₁ = dL/dh₁  ⊙  (1 - h₁²)

= [-0.1438 × 0.7865,  0.0218 × 0.9901]
= [-0.1131,            0.0216]   (2×1)

Notice: the gradient shrank again through the tanh derivative.
At t=1 it is already smaller than at t=2.
```

**B11: Gradient of L w.r.t. Wₓ at t=1 (dL/dWₓ at t=1)**

```
dL/dWₓ_t1 = dL/dz₁  ·  x₁ᵀ

= [-0.1131]  ·  [1, 0, 0, 0]
  [ 0.0216]

= [[-0.1131, 0, 0, 0],
   [ 0.0216, 0, 0, 0]]   (2×4)
```

**B12: Gradient of L w.r.t. Wₕ at t=1 (dL/dWₕ at t=1)**

```
dL/dWₕ_t1 = dL/dz₁  ·  h₀ᵀ

h₀ = [0, 0]  →  anything × [0,0] = all zeros

dL/dWₕ_t1 = [[0, 0],
              [0, 0]]   (2×2)

This makes sense — h₀ was all zeros, so Wₕ had no influence at t=1.
```

---

## PART C — ACCUMULATE GRADIENTS ACROSS ALL TIME STEPS

Each weight matrix accumulates gradients from every time step it was used.

**Total dL/dWy**

```
dL/dWy = dL₁/dWy + dL₂/dWy

dL₁/dWy = dL₁/draw₁ · h₁ᵀ
= [[ 0.2429×0.4621,  0.2429×0.0997],
   [-0.7335×0.4621, -0.7335×0.0997],
   [ 0.2478×0.4621,  0.2478×0.0997],
   [ 0.2429×0.4621,  0.2429×0.0997]]

= [[ 0.1122,  0.0242],
   [-0.3389, -0.0731],
   [ 0.1145,  0.0247],
   [ 0.1122,  0.0242]]

dL₂/dWy (from B2) = [[ 0.0805,  0.1243],
                      [ 0.0861,  0.1329],
                      [-0.2471, -0.3815],
                      [ 0.0805,  0.1243]]

Total dL/dWy = [[ 0.1927,  0.1485],
                [-0.2528,  0.0598],
                [-0.1326, -0.3568],
                [ 0.1927,  0.1485]]
```

**Total dL/dWₓ**

```
dL/dWₓ = dL/dWₓ_t1 + dL/dWₓ_t2

= [[-0.1131, 0,       0, 0],    +   [[ 0,       0.0453, 0, 0],
   [ 0.0216, 0,       0, 0]]        [ 0,       -0.1073, 0, 0]]

= [[-0.1131,  0.0453, 0, 0],
   [ 0.0216, -0.1073, 0, 0]]

Only columns 0 and 1 have gradients because we only used 'h' and 'e' as inputs.
Columns for 'l' and 'o' are zero — those characters were never seen.
```

**Total dL/dWₕ**

```
dL/dWₕ = dL/dWₕ_t1 + dL/dWₕ_t2

= [[0, 0],       +   [[ 0.0209,  0.0045],
   [0, 0]]           [-0.0496, -0.0107]]

= [[ 0.0209,  0.0045],
   [-0.0496, -0.0107]]
```

---

## PART D — WEIGHT UPDATE

Learning rate lr = 0.1

```
Wy_new = Wy - 0.1 × dL/dWy

Wy_old = [[ 0.1,  0.1],     dL/dWy = [[ 0.1927,  0.1485],
           [ 0.3,  0.1],               [-0.2528,  0.0598],
           [ 0.1,  0.3],               [-0.1326, -0.3568],
           [ 0.1,  0.1]]               [ 0.1927,  0.1485]]

Wy_new = [[ 0.1 - 0.0193,  0.1 - 0.0149],
           [ 0.3 + 0.0253,  0.1 - 0.0060],
           [ 0.1 + 0.0133,  0.3 + 0.0357],
           [ 0.1 - 0.0193,  0.1 - 0.0149]]

       = [[ 0.0807,  0.0851],
          [ 0.3253,  0.0940],
          [ 0.1133,  0.3357],
          [ 0.0807,  0.0851]]

Notice: Row 2 (for 'l') got the biggest update because 'l' was the target most often.
Row 3 (for 'o') also grew because 'o' follows 'l' in the sequence.
```

---

## PART E — THE VANISHING GRADIENT IN ACTION

Here is exactly where and why the gradient shrinks going back through time:

```
GRADIENT MAGNITUDE at each step:

At output (draw₂):  max gradient = 0.7345    (large — direct signal from loss)
At h₂:             max gradient = 0.1470    (shrank through Wy)
At z₂:             max gradient = 0.1073    (shrank through tanh: × 0.7302)
At h₁ from t=2:    max gradient = 0.0277    (shrank through Wₕ)

Compare to direct gradient at h₁:
dL₁/dh₁_direct:    max gradient = 0.1467    (much larger — direct path)
dL₂/dh₁_from_t2:   max gradient = 0.0277    (much smaller — traveled through time)
```

Visually:

```
t=2 Loss                              t=1 Loss
    │                                     │
    │ 0.7345                              │ 0.7335
    ↓                                     ↓
  draw₂                                 draw₁
    │ × Wyᵀ                              │ × Wyᵀ
    ↓                                     ↓
   h₂  → 0.1470                          h₁  ← receives 0.1467 from L₁ directly
    │ × (1-h₂²) = ×0.730                 │           + 0.0277 from t=2 (small!)
    ↓                                     │
   z₂  → 0.1073                          Total at h₁ = 0.1438
    │ × Wₕᵀ                              │ × (1-h₁²) = ×0.787
    ↓                                     ↓
  h₁ from t=2 → 0.0277 ─────────────→  z₁ → 0.1131
                                          │ × Wₓᵀ
                                          ↓
                                       no earlier step
```

If we had T=10 steps instead of 2, the gradient from step 10 arriving at step 1 would be:

```
0.7345 × (Wₕ slope)^9 × (tanh slope)^9

With tanh slope ≈ 0.5 each step:
= 0.7345 × 0.5^9 = 0.7345 × 0.002 = 0.0015  ← nearly zero

With tanh slope ≈ 0.25 (typical):
= 0.7345 × 0.25^9 = 0.7345 × 0.0000038 ≈ 0.000003  ← completely vanished
```

---

## PART F — FULL BPTT FLOW DIAGRAM

```
FORWARD (left to right):
                                                  
h₀ ──→ [Cell t=1] ──h₁──→ [Cell t=2] ──h₂──→ [Wy]──→ ŷ₂──→ L₂
  x₁↗       ↓          x₂↗       ↓                          
         [Wy]                  [Wy]                          
           ↓                    ↓                            
          ŷ₁                  ŷ₂                            
           ↓                    ↓                            
          L₁                  L₂                            

BACKWARD (right to left):

L₂──→ dL₂/draw₂ ──→ dL₂/dWy (accumulate)
            │
            ├──→ dL₂/dh₂
            │         │
            │    × (1-h₂²)  ← tanh derivative, shrinks gradient
            │         │
            │    = dL₂/dz₂
            │         │
            │    ├──→ dL₂/dWₓ (t=2, accumulate)
            │    ├──→ dL₂/dWₕ (t=2, accumulate)
            │    │
            │    × Wₕᵀ  ← gradient travels back through time
            │         │
            │    = dL₂/dh₁  [0.0029, -0.0277]  ← SMALL (traveled through time)
            │         │
L₁──→ dL₁/draw₁──→ dL₁/dWy (accumulate)
            │
            ├──→ dL₁/dh₁  [-0.1467, 0.0495]  ← LARGER (direct path)
            │
            + (add both gradients at h₁)
            │
       dL/dh₁ = [-0.1438, 0.0218]
            │
       × (1-h₁²)  ← tanh derivative, shrinks again
            │
       = dL/dz₁
            │
       ├──→ dL/dWₓ (t=1, accumulate)
       ├──→ dL/dWₕ (t=1, accumulate — zero because h₀=0)
       │
       × Wₕᵀ  ← would flow to h₀ but h₀ is fixed (all zeros, not trainable)
       STOP
```

---

## Summary — Key Rules of BPTT

```
1. Go backwards through time steps (t=T → t=1)

2. At each step, gradient at hₜ = 
   (direct gradient from Lₜ) + (gradient flowing back from t+1)

3. Gradient shrinks at each step back because of:
   - Multiplication by Wₕᵀ   (can amplify or shrink)
   - Multiplication by (1-hₜ²)  (always ≤ 1, always shrinks)

4. Each weight matrix accumulates gradients from ALL time steps it was used:
   dL/dWₓ = Σ dLₜ/dWₓ  (sum across t=1 to T)
   dL/dWₕ = Σ dLₜ/dWₕ  (sum across t=1 to T)
   dL/dWy = Σ dLₜ/dWy  (sum across t=1 to T)

5. After all gradients are accumulated → update all weights simultaneously
```
