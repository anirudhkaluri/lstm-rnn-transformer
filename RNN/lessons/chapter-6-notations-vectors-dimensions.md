# Chapter 6: Notations, Vectors, Dimensions — The Complete Reference

> This chapter answers every question about what each symbol means, what type it is (scalar / vector / matrix), what size it is, and what it is being multiplied with and why.

---

## What is a Neuron? (Building Block)

A single neuron does exactly 3 things:

```
1. Takes all inputs
2. Multiplies each input by its own weight and adds them up
3. Squishes the result through tanh
```

Example with 3 inputs:

```
inputs:   x₁=0.5,  x₂=0.8,  x₃=-0.3
weights:  w₁=0.4,  w₂=0.7,  w₃=0.2

sum  =  x₁×w₁ + x₂×w₂ + x₃×w₃
     =  0.5×0.4 + 0.8×0.7 + (-0.3)×0.2
     =  0.2 + 0.56 - 0.06
     =  0.70

output = tanh(0.70) = 0.604
```

Each neuron has its own set of weights. Each neuron produces **one output number**.

---

## Is an RNN Cell One Neuron?

**No.** An RNN cell is a **layer of many neurons** working together. The single box you see in diagrams is shorthand for all of them.

```
Diagram shorthand:          What is actually inside:
                            
┌──────────┐                x₁ ──→ [neuron 1]  ──→ h₁
xₜ → │ RNN Cell │ → hₜ     x₂ ──→ [neuron 2]  ──→ h₂
     └──────────┘           x₃ ──→ [neuron 3]  ──→ h₃
                            ...        ...          ...
                            x₂₇──→ [neuron 128] ──→ h₁₂₈
```

Every input connects to every neuron. This is called a **fully connected layer**.

---

## What is xₜ?

**xₜ is the input at the current time step t. It is always a vector.**

- It encodes whatever you are feeding in right now (a character or a word)
- We use **one-hot encoding** to convert it to numbers

### What is One-Hot Encoding?

One-hot encoding turns a character or word into a vector with one 1 and all 0s.

```
Vocabulary = [H, I, !]   → vocab size = 3

H  →  [1, 0, 0]   (position 0 is hot)
I  →  [0, 1, 0]   (position 1 is hot)
!  →  [0, 0, 1]   (position 2 is hot)
```

The size of xₜ always equals the vocabulary size.

---

## Letter Level vs Word Level Tokenization

The unit you encode is called a **token**. You decide what a token is. Both are valid choices.

| | Letter Level | Word Level |
|---|---|---|
| Tokens | H, I, !, a, b, c... | "the", "dog", "ran"... |
| Vocab size | ~27 to 100 | ~10,000 to 100,000 |
| xₜ size | ~27 | ~10,000 |
| Time steps for 70 chars | 70 | — |
| Time steps for 15 words | — | 15 |

**Important:** xₜ size = vocab size. Time steps = length of your input sequence. These are completely independent numbers.

---

## What is t?

`t` is just a **counter** for which position you are at in the sequence.

```
Sentence:   H        I        !
            ↑        ↑        ↑
           t=1      t=2      t=3

t=1 means "we are at step 1, reading H"
t=2 means "we are at step 2, reading I"
```

---

## What is h? What is hₜ?

`h` is the **hidden state** — the RNN's memory, like a sticky note.

- `hₜ` = the sticky note AFTER reading step t
- `hₜ₋₁` = the sticky note from the PREVIOUS step (before reading step t)
- `h₀` = the starting sticky note, all zeros, before reading anything

```
h₀ = [0.00, 0.00, 0.00, 0.00]   ← before reading anything
h₁ = [0.46, 0.31, 0.72, 0.11]   ← after reading H
h₂ = [0.70, 0.54, 0.83, 0.29]   ← after reading I (carries memory of H too)
```

**hₜ is a vector.** Its size = hidden size (a number YOU choose when designing the RNN).

```
Small RNN:   hidden size = 128   → h is a vector of 128 numbers
Medium RNN:  hidden size = 256   → h is a vector of 256 numbers
Large RNN:   hidden size = 1024  → h is a vector of 1024 numbers
```

---

## Time Steps vs Neurons — These are NOT the Same Thing

People confuse these constantly.

```
Time steps  =  how many times you RUN the cell   (depends on input length)
Neurons     =  how big the cell IS               (you decide this)
```

Example:

```
Sentence = 70 characters
Hidden size = 128 neurons

→ Same 128-neuron cell runs 70 times.
→ One cell. Used 70 times. Not 70 different cells.
```

Think of it like a washing machine. You do not buy a new machine for each piece of clothing. You run the same machine 70 times, once per item.

---

## What is Wₓ?

`Wₓ` is the **input weight matrix**. It contains the weights connecting every input to every neuron.

- Every neuron has one weight per input
- Every input connects to every neuron (fully connected)
- So total weights = neurons × inputs

```
Hidden size = 128 neurons
Vocab size  = 27 inputs

Wₓ size = 128 × 27 = 3,456 individual weights

Each row    = one neuron's weights for all 27 inputs
Each column = one input's weights going to all 128 neurons
```

### How Wₓ is Used

```
Wₓ        ·    xₜ       =   part1
(128×27)  ·   (27×1)    =   (128×1)

Each neuron's row [27 weights] · input vector [27 numbers] = one number per neuron
→ 128 neurons → 128 numbers → part1 is a 128×1 vector
```

---

## What is Wₕ?

`Wₕ` is the **hidden weight matrix**. It contains the weights connecting every previous memory value to every neuron.

The RNN has two inputs at every step:
1. `xₜ` — what we are reading RIGHT NOW
2. `hₜ₋₁` — the memory from the PREVIOUS step

Both need their own weight matrix.

```
Wₓ  →  weights for the CURRENT INPUT    xₜ
Wₕ  →  weights for the PREVIOUS MEMORY  hₜ₋₁
```

```
Hidden size = 128 neurons
Previous hidden state = 128 numbers

Wₕ size = 128 × 128 = 16,384 individual weights
```

### Why is Wₕ Square (128×128)?

Because the memory coming IN (`hₜ₋₁`) and the memory going OUT (`hₜ`) are both size 128.

```
hₜ₋₁  is 128 numbers   ← previous memory, coming in
hₜ    is 128 numbers   ← new memory, going out

So Wₕ must be 128×128 to transform one into the other
```

### How Wₕ is Used

```
Wₕ          ·   hₜ₋₁     =   part2
(128×128)   ·   (128×1)  =   (128×1)

Each neuron's row [128 weights] · memory vector [128 numbers] = one number per neuron
→ 128 neurons → 128 numbers → part2 is a 128×1 vector
```

---

## What is b (Bias)?

Each neuron has its own bias. It is a single number — the neuron's personal default before it sees any input.

```
Neuron 1   → b₁   = 0.5
Neuron 2   → b₂   = -1.2
Neuron 3   → b₃   = 0.0
...
Neuron 128 → b₁₂₈ = 0.3
```

When we write `b` in the formula we mean all 128 biases collected into one vector:

```
b = [b₁, b₂, b₃, ..., b₁₂₈]   →  size 128×1
```

### Why Bias Exists

Without bias, if all inputs are 0 then output = tanh(0) = 0 always. The neuron is stuck. Bias gives each neuron a starting personality — some naturally lean toward firing, some toward staying quiet.

---

## Matrix Multiplication Rule

Before going further, understand this rule:

```
Matrix A (m × n)  ·  Matrix B (n × p)  =  Result (m × p)

The INNER two dimensions must be equal.
The OUTER two dimensions become the result size.

(m × n) · (n × p) = (m × p)
         ↑↑ must match
```

Examples:

```
(128×27) · (27×1)  = (128×1)   ✅   inner: 27=27
(128×128)· (128×1) = (128×1)   ✅   inner: 128=128
(3×4)    · (4×2)   = (3×2)     ✅   inner: 4=4
(3×4)    · (5×2)   = ERROR     ✗    inner: 4≠5
```

---

## The Full Formula — Completely Annotated

```
hₜ  =  tanh(  Wₓ · xₜ   +   Wₕ · hₜ₋₁   +   b  )
```

### Step by Step with Dimensions

```
Step 1:   Wₓ       ·   xₜ        =   part1
          128×27   ·   27×1      =   128×1
          
          Each of 128 neurons gets:
          row[27 weights] · column[27 inputs] = one number
          
          Result: 128 numbers

──────────────────────────────────────────────────────

Step 2:   Wₕ          ·   hₜ₋₁     =   part2
          128×128      ·   128×1    =   128×1
          
          Each of 128 neurons gets:
          row[128 weights] · column[128 memory values] = one number
          
          Result: 128 numbers

──────────────────────────────────────────────────────

Step 3:   b   =   part3
          128×1
          
          128 individual bias numbers, one per neuron

──────────────────────────────────────────────────────

Step 4:   part1   +   part2   +   part3   =   z
          128×1   +   128×1   +   128×1   =   128×1
          
          Add element by element:
          Neuron 1:   part1₁ + part2₁ + b₁  = z₁
          Neuron 2:   part1₂ + part2₂ + b₂  = z₂
          ...
          Neuron 128: part1₁₂₈ + part2₁₂₈ + b₁₂₈ = z₁₂₈

──────────────────────────────────────────────────────

Step 5:   tanh( z )   =   hₜ
          tanh(128×1) =   128×1
          
          Apply tanh to every single number independently
          
          Result: hₜ = 128 numbers, all between -1 and +1  ✅
```

### Step 6 — Output Prediction (Wy)

After all time steps, hₜ is passed through Wy to make a prediction:

```
Step 6:   Wy        ·    hₜ       =   raw scores   →  softmax  →  ŷₜ
          27×128    ·    128×1    =   27×1                          27×1

          Each of 27 output neurons gets:
          row[128 weights] · hidden state[128 numbers] = one raw score per character

          softmax converts 27 raw scores → 27 probabilities that sum to 1
```

---

## What is Wy?

`Wy` is the **output weight matrix**. It connects the hidden state to the final prediction.

- hₜ (128 numbers) is the RNN's internal memory — NOT a prediction yet
- Wy converts it into 27 numbers (one per character)
- Softmax converts those 27 numbers into probabilities

```
Wy size = vocab_size × hidden_size = 27 × 128 = 3,456 weights
```

It is fully trainable — updated via backpropagation just like Wₓ and Wₕ.

---

## What is Softmax?

Softmax is the output activation. It turns raw scores into probabilities that sum to 1.

```
Raw scores:   H=-1.2,  I=2.4,  !=0.8,  a=-0.3

After softmax:
  P(H) = 0.02
  P(I) = 0.71   ← highest → RNN picks "I"
  P(!) = 0.19
  P(a) = 0.05
  ────────────
  Total = 1.00  ✅

Formula:  softmax(zᵢ) = eᶻⁱ / Σ eᶻʲ
```

tanh is used for the hidden state. Softmax is used for the final prediction. They serve different purposes.

---

## What Each Neuron Does — Fully Written Out

```
Neuron 1:
  z₁ = (x₁×Wₓ₁₁ + x₂×Wₓ₁₂ + ... + x₂₇×Wₓ₁₂₇)    ← from input
      +(h₁×Wₕ₁₁ + h₂×Wₕ₁₂ + ... + h₁₂₈×Wₕ₁₁₂₈)  ← from memory
      + b₁                                           ← bias
  h₁_new = tanh(z₁)

Neuron 2:
  z₂ = (x₁×Wₓ₂₁ + x₂×Wₓ₂₂ + ... + x₂₇×Wₓ₂₂₇)
      +(h₁×Wₕ₂₁ + h₂×Wₕ₂₂ + ... + h₁₂₈×Wₕ₂₁₂₈)
      + b₂
  h₂_new = tanh(z₂)

... same for all 128 neurons
```

All 128 neurons run this simultaneously. Their outputs form the new hidden state vector hₜ.

---

## Example with Real Numbers (Character Level)

```
Sentence:   "The dog ran fast because it was scared..."
Characters: 70 total (including spaces)
Vocabulary: a-z + space = 27 characters
Hidden size: 128 (your choice)
```

```
xₜ size      = 27×1      one-hot vector for current character
hₜ size      = 128×1     hidden state vector
h₀           = 128×1     all zeros at start
xₜ size      = 27×1       one-hot vector for current character
hₜ size      = 128×1     hidden state vector
h₀           = 128×1     all zeros at start
Wₓ size      = 128×27    3,456 weights
Wₕ size      = 128×128   16,384 weights
Wy size      = 27×128    3,456 weights
b size        = 128×1     128 bias values (one per hidden neuron)
by size       = 27×1      27 bias values (one per output character)
Time steps   = 70         cell runs 70 times (once per character)
Neurons      = 128        inside the one cell
```

---

## Example with Real Numbers (Word Level)

```
Sentence:   "The dog ran fast because it was scared of the loud thunder sound today"
Words:      15 total
Vocabulary: 10,000 English words
Hidden size: 256 (your choice)
```

```
xₜ size      = 10,000×1   one-hot vector for current word
hₜ size      = 256×1      hidden state vector
h₀           = 256×1      all zeros at start
xₜ size      = 10,000×1    one-hot vector for current word
hₜ size      = 256×1       hidden state vector
h₀           = 256×1       all zeros at start
Wₓ size      = 256×10,000  2,560,000 weights
Wₕ size      = 256×256     65,536 weights
Wy size      = 10,000×256  2,560,000 weights
b size        = 256×1       256 bias values (one per hidden neuron)
by size       = 10,000×1    10,000 bias values (one per output word)
Time steps   = 15           cell runs 15 times (once per word)
Neurons      = 256          inside the one cell
```

---

## Complete Notation Table

| Symbol | Name | Type | Size | Plain English |
|---|---|---|---|---|
| `t` | Time step | Scalar | 1 number | Which position in sequence (1, 2, 3...) |
| `xₜ` | Input at step t | Vector | vocab_size × 1 | Current token as one-hot vector |
| `hₜ` | Hidden state at step t | Vector | hidden_size × 1 | Memory after reading step t |
| `hₜ₋₁` | Previous hidden state | Vector | hidden_size × 1 | Memory from previous step |
| `h₀` | Initial hidden state | Vector | hidden_size × 1 | All zeros before reading anything |
| `Wₓ` | Input weight matrix | Matrix | hidden_size × vocab_size | Weights connecting input → hidden neurons |
| `Wₕ` | Hidden weight matrix | Matrix | hidden_size × hidden_size | Weights connecting previous memory → hidden neurons |
| `Wy` | Output weight matrix | Matrix | vocab_size × hidden_size | Weights connecting hidden neurons → prediction |
| `b` | Hidden bias vector | Vector | hidden_size × 1 | One bias number per hidden neuron |
| `by` | Output bias vector | Vector | vocab_size × 1 | One bias number per output character |
| `z` | Pre-activation (hidden) | Vector | hidden_size × 1 | Sum before tanh is applied |
| `ŷₜ` | Predicted output at step t | Vector | vocab_size × 1 | Probability over all characters |
| `yₜ` | True output at step t | Vector | vocab_size × 1 | The correct answer (one-hot) |
| `L` | Total loss | Scalar | 1 number | Sum of mistakes across all time steps |
| `Lₜ` | Loss at step t | Scalar | 1 number | How wrong the prediction was at step t |

---

## Total Parameters to Learn

### Character Level (vocab=27, hidden=128)
```
Wₓ   =  128 × 27    =   3,456
Wₕ   =  128 × 128   =  16,384
Wy   =   27 × 128   =   3,456
b    =  128          =     128
by   =   27          =      27
────────────────────────────────
Total                =  23,451  numbers to learn
```

### Word Level (vocab=10,000, hidden=256)
```
Wₓ   =  256 × 10,000   =  2,560,000
Wₕ   =  256 × 256      =     65,536
Wy   =  10,000 × 256   =  2,560,000
b    =  256             =        256
by   =  10,000          =     10,000
────────────────────────────────────
Total                   =  5,195,792  numbers to learn
```

---

## Key Rules to Remember

```
Wₓ columns   =  xₜ rows    =  vocab size    (must match for multiplication)
Wₕ columns   =  hₜ₋₁ rows  =  hidden size  (must match for multiplication)
Wₕ rows      =  hₜ rows    =  hidden size  (that is why Wₕ is square)
b rows        =  hₜ rows    =  hidden size  (one bias per neuron)

Input size  does NOT determine output size
Output size is determined by hidden size, which YOU choose
The weight matrix bridges the gap between them
```

---

## Q&A — Every Question That Was Asked

**Q: Is an RNN just one single neuron when not unrolled?**
No. An RNN cell is a full layer of many neurons (e.g. 128). The single box in diagrams is shorthand for all of them. Each neuron inside sees all inputs and all previous memory values.

**Q: What is xₜ — a vector or a single number?**
Always a vector. Size = vocab size. Never a single number, because we need to encode which character or word it is using one-hot encoding.

**Q: Is hₜ₋₁ a vector of the previous state?**
Yes. hₜ₋₁ is the complete hidden state vector from the previous time step. Size = hidden size. It contains one number per neuron from the previous step.

**Q: Are we encoding letters or words?**
You choose. The token (what you encode) can be a letter or a word. Both are valid. For letters, vocab size ≈ 27. For words, vocab size ≈ 10,000. We used letters in this course for simplicity.

**Q: If xₜ is size 27, why is the output 128?**
Because every input connects to every neuron (fully connected). Input size and neuron count are independent. Wₓ (128×27) bridges the gap. You choose 128 neurons. 27 is just the input size.

**Q: Are biases different for each neuron?**
Yes. Every neuron has its own bias — one single number. When written as `b` in the formula, it means all biases collected into a vector of size 128×1.

**Q: Is bias a 1×1 vector per neuron?**
No. The bias for one neuron is a plain scalar — just one number. The vector `b` in the formula is all 128 biases stacked together into a 128×1 vector so the matrix math works out.

**Q: What is Wₕ?**
Wₕ is the weight matrix for the previous hidden state hₜ₋₁. Just like Wₓ handles the new input, Wₕ handles the old memory. It is square (128×128) because both input and output hidden states are the same size (128).

**Q: Wₕ is 128×128 and hₜ₋₁ is 128×1 — are they compatible to multiply?**
Yes. Matrix multiplication rule: (m×n)·(n×p) = (m×p). Inner dimensions must match. (128×128)·(128×1) → inner dimensions are both 128 → compatible → result is 128×1.

**Q: Is Wₕ · hₜ₋₁ = hₜ?**
No. That is only one part of the formula. The full formula is:
hₜ = tanh( Wₓ·xₜ + Wₕ·hₜ₋₁ + b )
Wₕ·hₜ₋₁ alone is just part2. It must be added to part1 and b, then passed through tanh, to get hₜ.

**Q: Why is Wy needed? Why didn't you mention it earlier?**
Wy was not mentioned in early lessons because we were focused on understanding the memory mechanism (how hₜ updates). Wy only comes into play at the very end when converting memory into a prediction. hₜ (128 numbers) is the RNN's internal state — it is not a prediction. Wy (27×128) converts it into 27 scores, one per character, and softmax converts those into probabilities.

**Q: Is Wy trainable?**
Yes. Wy starts random and is updated via backpropagation at every training step, just like Wₓ and Wₕ. The gradient flows through Wy first (from loss → raw scores → Wy) before continuing backwards through the hidden states.

**Q: What is softmax and how is it different from tanh?**
tanh squishes one number to between -1 and +1. It is applied to each hidden neuron independently.
Softmax takes a whole vector of numbers and converts them into probabilities that sum to 1. It is applied to the output layer only. tanh is for internal memory. Softmax is for final predictions.

**Q: What is by?**
by is the output bias vector — one bias number per output character (size 27×1). Just like b gives each hidden neuron a default, by gives each output character score a default offset. It is also trainable.
