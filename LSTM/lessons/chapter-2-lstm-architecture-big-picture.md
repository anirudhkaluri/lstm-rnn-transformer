# Chapter 2 — LSTM Architecture: The Big Picture

---

## Two streams, not one

An RNN carries exactly one thing forward through time: the hidden state $h_t$.

An LSTM carries **two** things forward:

- $h_t \in \mathbb{R}^{H \times 1}$ — the **hidden state** (same as RNN, size H)
- $c_t \in \mathbb{R}^{H \times 1}$ — the **cell state** (new, also size H)

$H$ is the number of hidden units you choose — a hyperparameter. Same symbol as in the RNN. Both vectors have the same size.

---

## What each stream carries

**The cell state $c_t$ is long-term memory.**

It is the running record of everything important the network has decided to remember. It changes slowly and selectively. Information can survive in $c_t$ for many timesteps almost untouched if the network decides it is important.

**The hidden state $h_t$ is working memory.**

It is what the network actively uses to make a prediction right now. It is a filtered, processed version of the cell state — squeezed through a gate to produce something useful at this specific moment.

Every timestep, the network reads from $c_t$ to produce $h_t$. And $h_t$ is what gets sent to the output layer (just like in the RNN).

---

## Is $c_t$ specific to one neuron or shared across the whole network?

$c_t \in \mathbb{R}^{H \times 1}$ is a vector with $H$ numbers in it. Each position in that vector belongs to **one specific LSTM neuron**.

So if $H = 4$:

$$c_t = \begin{bmatrix} 0.9 \\ -0.3 \\ 0.1 \\ 0.7 \end{bmatrix}$$

- Neuron 1 has cell state $0.9$
- Neuron 2 has cell state $-0.3$
- Neuron 3 has cell state $0.1$
- Neuron 4 has cell state $0.7$

Every neuron has **its own private cell state value**. Neuron 1's memory is completely separate from Neuron 2's memory.

The vector $c_t$ is just a convenient way of writing all $H$ neurons' cell states stacked together. It is not one shared number — it is $H$ individual numbers, one per neuron, packed into a single vector so the math is clean.

The same is true for $h_t$, $f_t$, $i_t$, $o_t$ — all of them are vectors of size $H \times 1$, one value per neuron. When you see $f_t \odot c_{t-1}$, the elementwise multiplication is: neuron 1's forget gate value × neuron 1's cell state, neuron 2's forget gate value × neuron 2's cell state, and so on. Each neuron manages its own memory independently.

---

## The architecture as a pipeline

Here is what happens at every single timestep $t$. Two things come in from the previous step, two things go out to the next step.

```
         c_{t-1}                              c_t
 ═══════════════════════════════════════════════════════►  (cell state — long-term memory)
              │          │              │
              ▼          ▼              ▼
           FORGET      WRITE         READ
            GATE        GATE          GATE
              │          │              │
         ─────────────────────────────────
              │                          │
 ────────────►│                          │──────────────►
   h_{t-1}   │       LSTM CELL          │    h_t
              │                          │
 ────────────►│                          │
     x_t     └──────────────────────────┘
```

Three gates sit between the two streams and control information flow. Each gate is just a vector of numbers between 0 and 1 — computed fresh at every timestep from the current input and previous hidden state.

- **Forget gate** — looks at $c_{t-1}$ and decides what to erase
- **Write gate (input gate)** — decides what new information to add to $c_t$
- **Read gate (output gate)** — decides what part of $c_t$ to expose as $h_t$

---

## What a gate actually is

A gate is just a **vector of numbers between 0 and 1**.

If $H = 4$, a gate might look like:

$$f_t = \begin{bmatrix} 0.9 \\ 0.02 \\ 0.7 \\ 0.95 \end{bmatrix}$$

You multiply this gate elementwise ($\odot$) against the cell state:

$$f_t \odot c_{t-1} = \begin{bmatrix} 0.9 \\ 0.02 \\ 0.7 \\ 0.95 \end{bmatrix} \odot \begin{bmatrix} 1.2 \\ -0.8 \\ 0.3 \\ -0.5 \end{bmatrix} = \begin{bmatrix} 1.08 \\ -0.016 \\ 0.21 \\ -0.475 \end{bmatrix}$$

Dimension 2 of the cell state ($-0.8$) got multiplied by $0.02$ — nearly wiped out. Dimension 1 ($1.2$) got multiplied by $0.9$ — almost fully kept. The gate selectively erased some things and preserved others.

$\odot$ means **elementwise multiplication** — same-position numbers multiplied together. Both vectors must be the same size, and the output is the same size. No dot product, no matrix. Just position-by-position multiplication.

---

## Where do gate values come from?

The gates are **computed from the current input**. At each timestep, the network looks at $x_t$ (current input) and $h_{t-1}$ (previous hidden state) and computes each gate fresh. The gate values are different at every timestep — they adapt to what is being seen right now.

A sigmoid function produces the gate values, because sigmoid outputs numbers between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

At each timestep:
1. Feed $x_t$ and $h_{t-1}$ into a small linear layer
2. Pass the result through sigmoid
3. Get a gate vector full of numbers between 0 and 1
4. Use it to control what flows through

The exact equations are in Chapter 3.

---

## The cell state update

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

| Piece | What it means |
|---|---|
| $f_t \odot c_{t-1}$ | Take the old memory. Erase parts of it (forget gate) |
| $i_t \odot \tilde{c}_t$ | Take new candidate information. Let some of it through (input gate) |
| $+$ | Add them together to form the new memory |

$\tilde{c}_t \in \mathbb{R}^{H \times 1}$ is the **candidate cell state** — a vector of new information proposed by the current input. The input gate $i_t$ decides how much of it to actually write in.

Then the hidden state:

$$h_t = o_t \odot \tanh(c_t)$$

| Piece | What it means |
|---|---|
| $\tanh(c_t)$ | Squash the cell state into $[-1, 1]$ |
| $o_t \odot \tanh(c_t)$ | Let only some of the memory through to become the output |

$h_t$ is not all of memory — it is a filtered read of memory, controlled by the output gate $o_t$.

---

## Why the "+" in the cell update is everything

The cell state $c_t$ is formed by **addition**. When you differentiate through an addition during BPTT:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

The gradient of $c_t$ with respect to $c_{t-1}$ is just $f_t$ — the forget gate values. No $W_h$ matrix multiplication. No $\tanh$ squashing. Just elementwise multiplication by values close to 1 if the network decided to remember.

If $f_t \approx 1$, the gradient flows back through time almost unchanged. The vanishing problem is solved.

---

## The full picture at one timestep

**Inputs arriving at timestep $t$:**
- $x_t \in \mathbb{R}^{H \times 1}$ — current input (after embedding)
- $h_{t-1} \in \mathbb{R}^{H \times 1}$ — previous hidden state
- $c_{t-1} \in \mathbb{R}^{H \times 1}$ — previous cell state

**What the LSTM computes (full equations in Chapter 3):**
1. Forget gate $f_t$ — what to erase from $c_{t-1}$
2. Input gate $i_t$ — how much new info to write
3. Candidate $\tilde{c}_t$ — the new info being proposed
4. New cell state $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
5. Output gate $o_t$ — what to read from $c_t$
6. New hidden state $h_t = o_t \odot \tanh(c_t)$

**Outputs leaving timestep $t$:**
- $h_t$ — goes to output layer AND to next timestep
- $c_t$ — goes to next timestep only

---

## Summary

| Concept | What it is |
|---|---|
| $c_t \in \mathbb{R}^{H \times 1}$ | Long-term memory — one value per neuron, additively updated |
| $h_t \in \mathbb{R}^{H \times 1}$ | Working memory — filtered read of $c_t$, used for predictions |
| Gate | A vector in $\mathbb{R}^{H \times 1}$ with values between 0 and 1 |
| $\odot$ | Elementwise multiplication — same size in, same size out |
| Forget gate $f_t$ | Decides what to erase from $c_{t-1}$ |
| Input gate $i_t$ + candidate $\tilde{c}_t$ | Decides what new info to write into $c_t$ |
| Output gate $o_t$ | Decides what part of $c_t$ becomes $h_t$ |
| Why "+" fixes gradients | $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ — no matrix multiply, no $\tanh$ squash |
