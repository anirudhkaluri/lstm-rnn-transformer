# Chapter 4: How the RNN Learns — Backpropagation Through Time (BPTT)

---

## How Any Neural Network Learns

Think of throwing a ball at a target. You miss. You look at how far you missed and adjust your throw. You try again. Miss less. Adjust again. Eventually you hit it.

Neural networks learn the same way:

```
1. Make a prediction          (forward pass)
2. Measure how wrong it was   (loss)
3. Figure out who is to blame (backpropagation)
4. Nudge the weights to do better (gradient descent)
5. Repeat thousands of times
```

---

## What is Being Trained?

Training means finding the best values for ALL weight matrices and biases:

```
Wₓ    (128×27)    →   3,456 numbers
Wₕ    (128×128)   →  16,384 numbers
Wy    (27×128)    →   3,456 numbers   ← output weights (see below)
b     (128×1)     →     128 numbers
by    (27×1)      →      27 numbers

Total: 23,451 numbers — all start random, all need to be learned
```

---

## What is a Training Dataset?

Before training starts you need examples. For a character-level RNN predicting the next character:

```
Input:    "The dog ra"
Target:   "he dog ran"
           ↑
           shifted by one position
```

At every time step, the input is one character and the target is the NEXT character:

```
t=1:  input=T   →  target=h
t=2:  input=h   →  target=e
t=3:  input=e   →  target=" "
t=4:  input=" " →  target=d
...
```

The RNN is always trying to answer: **"given everything I have read so far, what comes next?"**

---

## The Three Stages of a Forward Pass

```
INPUT                  MEMORY                 PREDICTION
  xₜ                    hₜ                      ŷₜ
(27×1)    ────→       (128×1)    ────→         (27×1)

one-hot              internal                probability
character            memory of               over all
vector               everything              characters
                     seen so far
```

- `Wₓ` and `Wₕ` handle: input + memory → new memory (hₜ)
- `Wy` handles: memory → prediction (ŷₜ)

---

## What is Wy?

`Wy` is the **output weight matrix**. It converts the hidden state (128 numbers) into a prediction over the vocabulary (27 numbers).

```
hₜ is 128 numbers — the RNN's internal memory.
These are NOT a prediction yet.
They are just the network's compressed understanding of what it has read.

We need to go from 128 numbers → 27 probabilities (one per character).
That is what Wy does.
```

```
Wy        ·    hₜ      =   raw scores
(27×128)  ·  (128×1)  =   (27×1)

Then softmax converts those 27 raw scores → 27 probabilities
```

**Is Wy trainable?** Yes. It starts random and is updated via backpropagation just like Wₓ and Wₕ. It needs to learn which hidden neurons are useful signals for predicting which characters.

---

## What is Softmax?

Softmax is the output activation. It turns raw scores into **probabilities that add up to 1**.

```
Raw scores (from Wy·hₜ):   H=-1.2,  I=2.4,  !=0.8,  a=-0.3

After softmax:
  P(H) = 0.02   (2% chance next is H)
  P(I) = 0.71   (71% chance next is I)  ← highest → RNN picks this
  P(!) = 0.19   (19% chance next is !)
  P(a) = 0.05   (5% chance next is a)
  ─────────────
  Total = 1.00  ✅ always adds to 1
```

Formula:

```
softmax(zᵢ) = eᶻⁱ / Σ eᶻʲ

Each score gets exponentiated then divided by the sum of all exponentiated scores
```

tanh is used for the hidden state. Softmax is used for the final prediction. They serve different purposes.

---

## Step 1 — Measure the Mistake (Loss Function)

Our RNN reads "HI" and should predict "!" next.

```
RNN predicted:   ŷ = 0.3   (its confidence that next letter is "!")
Correct answer:  y  = 1.0   (yes it should be "!")

Loss = (y - ŷ)²  =  (1.0 - 0.3)²  =  0.49   (pretty wrong)
```

In practice we use **Cross Entropy Loss** (better than squared error for probabilities):

```
L = -Σ yᵢ · log(ŷᵢ)

If correct answer is "!" and RNN said P(!) = 0.71:
L = -log(0.71) = 0.34   (not too bad)

If correct answer is "!" and RNN said P(!) = 0.02:
L = -log(0.02) = 3.91   (very wrong, high loss)
```

The total loss across all time steps:

```
L = L₁ + L₂ + L₃ + ... + Lₜ
```

---

## Step 2 — Who is to Blame? (Backpropagation Through Time)

In a regular network you trace the error backwards through the layers.

In an RNN, time is also involved. The mistake at step 4 might have been caused by a bad weight at step 1. So we trace the error **backwards through time**:

```
FORWARD (making predictions):
  x₁ → h₁ → x₂ → h₂ → x₃ → h₃ → Wy → ŷ → LOSS

BACKWARD (finding blame):
  LOSS → Wy → h₃ → h₂ → h₁
```

Gradient flows through Wy first, then backwards through every hidden state, all the way to t=1.

---

## Step 3 — The Gradient

The gradient answers:
> "If I increase this weight by a tiny amount, does the loss go up or down?"

```
Gradient positive → weight is making things worse → decrease it
Gradient negative → weight is making things better → increase it
```

Formula for updating a weight:

```
new_weight = old_weight - (learning_rate × gradient)
```

`learning_rate` is your step size. Too big = overshoot. Too small = learn forever.

---

## Full BPTT Example

```
FORWARD PASS:
  t=1: read H  → h₁ = 0.46
  t=2: read I  → h₂ = 0.70
  t=3: read !  → h₃ = 0.89
               → Wy·h₃ → softmax → ŷ = [0.02, 0.71, 0.19, ...]
  
  Correct answer was "I" (t=2 predicts next char after H)
  Loss = -log(0.71) = 0.34

BACKWARD PASS:
  dL/dWy        ← how much did Wy cause the mistake?
  dL/dh₃        ← how much did h₃ cause the mistake?
  dL/dh₂        ← how much did h₂ cause the mistake?
  dL/dh₁        ← how much did h₁ cause the mistake?
  dL/dWₓ        ← how much did Wₓ cause the mistake?
  dL/dWₕ        ← how much did Wₕ cause the mistake?

UPDATE:
  Wy = Wy - 0.01 × dL/dWy
  Wₓ = Wₓ - 0.01 × dL/dWₓ
  Wₕ = Wₕ - 0.01 × dL/dWₕ
  b  = b  - 0.01 × dL/db
  by = by - 0.01 × dL/dby
```

---

## The Full Training Loop

```
STEP 1 — INITIALISE
━━━━━━━━━━━━━━━━━━━
Set Wₓ, Wₕ, Wy, b, by to small random numbers
Set h₀ = all zeros

STEP 2 — FORWARD PASS
━━━━━━━━━━━━━━━━━━━━━
For each time step t = 1 to T:
    hₜ = tanh( Wₓ·xₜ + Wₕ·hₜ₋₁ + b )
    ŷₜ = softmax( Wy·hₜ + by )

STEP 3 — CALCULATE LOSS
━━━━━━━━━━━━━━━━━━━━━━━
L = L₁ + L₂ + ... + Lₜ   (sum across all time steps)

STEP 4 — BACKWARD PASS (BPTT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Calculate gradients dL/dWy, dL/dWₓ, dL/dWₕ, dL/db, dL/dby

STEP 5 — UPDATE WEIGHTS
━━━━━━━━━━━━━━━━━━━━━━━
Wy = Wy - lr × dL/dWy
Wₓ = Wₓ - lr × dL/dWₓ
Wₕ = Wₕ - lr × dL/dWₕ
b  = b  - lr × dL/db
by = by - lr × dL/dby

STEP 6 — REPEAT
━━━━━━━━━━━━━━━
Repeat for every batch, every epoch, until loss is small
```

---

## What is an Epoch?

```
One epoch = the model has seen every training example once

Epoch 1:   Loss = 3.21   (guessing randomly)
Epoch 5:   Loss = 2.14   (learning some patterns)
Epoch 20:  Loss = 1.43   (getting better)
Epoch 50:  Loss = 0.87   (quite good)
Epoch 100: Loss = 0.54   (well trained)
```

---

## What is a Batch?

You do not feed the entire dataset at once. You feed it in small chunks called batches.

```
Dataset:    10,000 sentences
Batch size: 32 sentences

One epoch = 10,000 / 32 = 312 batches
Each batch does one forward pass + one backward pass + one weight update
```

Why batches?
- Full dataset at once = too much memory
- One example at a time = too slow and noisy
- Batches = best of both worlds

---

## Truncated BPTT — Practical Fix

In theory BPTT goes all the way back to t=1. In practice with long sequences this is too slow and the gradient vanishes anyway. So we use **Truncated BPTT** — only backpropagate through the last K steps:

```
Full sequence:     t=1 ... t=200
Truncated BPTT:   only backpropagate through last 20 steps

Gradient flows:   t=200 → t=199 → ... → t=180  (stop here)
```

---

## The Loss Curve

```
Loss
 │
3│ ×
 │  ×
2│    ×
 │      × ×
1│          × × × ×
 │                  × × × × × × ×
0│─────────────────────────────────→ Epoch
 0   10   20   30   40   50   60
```

Loss going down = network is learning.

---

## All Weight Matrices — Complete Picture

| Matrix | Size | Connects | Trainable |
|---|---|---|---|
| `Wₓ` | 128×27 | input xₜ → hidden hₜ | ✅ Yes |
| `Wₕ` | 128×128 | previous memory hₜ₋₁ → hidden hₜ | ✅ Yes |
| `Wy` | 27×128 | hidden hₜ → output prediction ŷₜ | ✅ Yes |
| `b` | 128×1 | bias for hidden neurons | ✅ Yes |
| `by` | 27×1 | bias for output layer | ✅ Yes |

---

## Total Parameters

```
Wₓ   =  128 × 27    =   3,456
Wₕ   =  128 × 128   =  16,384
Wy   =   27 × 128   =   3,456
b    =  128          =     128
by   =   27          =      27
─────────────────────────────
Total                =  23,451  numbers to learn
```

---

## Key Takeaway

> BPTT unrolls the RNN through time and traces the blame for the mistake backwards through every time step, through every weight matrix including Wy. All five parameter matrices (Wₓ, Wₕ, Wy, b, by) are updated every training step.
