# Chapter 3: The RNN Loop — Step by Step with Real Numbers

---

## Our Task

We will teach the RNN to read the word **"HI!"** one letter at a time.

```
Step 1: Read "H"
Step 2: Read "I"
Step 3: Read "!"
```

---

## Turning Letters into Numbers (One-Hot Encoding)

Computers cannot read letters, so we convert them into vectors:

```
Vocabulary = [H, I, !]   → size = 3

H = [1, 0, 0]
I = [0, 1, 0]
! = [0, 0, 1]
```

Only one position is "1" (hot) and the rest are "0". That is why it is called **one-hot encoding**.

---

## Setup — Starting Weights

We start with made-up weights (in real life these are learned through training):

```
Wₓ (input weight)   = [0.5, 0.5, 0.5]
Wₕ (memory weight)  = 0.8
b  (bias)           = 0
h₀ (starting memory) = 0    ← blank sticky note
```

---

## Step 1: Read "H" = [1, 0, 0]

```
hₜ = tanh( xₜ·Wₓ + hₜ₋₁·Wₕ + b )

h₁ = tanh( [1,0,0]·[0.5,0.5,0.5]  +  0 × 0.8  +  0 )
   = tanh( (1×0.5 + 0×0.5 + 0×0.5) + 0 + 0 )
   = tanh( 0.5 )
   = 0.46
```

Sticky note after Step 1: **h₁ = 0.46**

---

## Step 2: Read "I" = [0, 1, 0]

```
h₂ = tanh( [0,1,0]·[0.5,0.5,0.5]  +  0.46 × 0.8  +  0 )
   = tanh( (0×0.5 + 1×0.5 + 0×0.5) + 0.368 )
   = tanh( 0.5 + 0.368 )
   = tanh( 0.868 )
   = 0.70
```

Sticky note after Step 2: **h₂ = 0.70**

---

## What Just Happened?

```
      [H]──→[RNN]──→ h₁=0.46 ──→[RNN]──→ h₂=0.70
              ↑                    ↑
         memory=0             memory=0.46
         (blank)              (remembers H!)
```

After reading "I", the hidden state **0.70** contains information about BOTH "H" and "I". The network did not forget "H" when it read "I". That is memory.

---

## The Loop Unrolled

```
Time:      t=1          t=2          t=3
Input:      H     →      I     →      !
           ↓             ↓             ↓
Memory:   0→0.46      0.46→0.70    0.70→...
```

Same network. Same weights. Just loops over and over reading one new input each time.

---

## Key Takeaway

> The hidden state is updated at every step using the same formula. The memory from the previous step (hₜ₋₁) is always carried forward into the next step. The RNN never forgets — it just keeps updating its sticky note.
