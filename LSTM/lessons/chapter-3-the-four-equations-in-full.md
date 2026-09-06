# Chapter 3 — The Four Equations in Full

---

## What we are building

At every timestep $t$, the LSTM takes in three things:

- $x_t \in \mathbb{R}^{E \times 1}$ — the current input (word embedding), size $E$
- $h_{t-1} \in \mathbb{R}^{H \times 1}$ — the previous hidden state, size $H$
- $c_{t-1} \in \mathbb{R}^{H \times 1}$ — the previous cell state, size $H$

And it produces six things in order. We derive every single one below.

---

## How weight matrices are named

There are four computations inside an LSTM — forget gate, input gate, candidate, output gate. Each one needs its own set of weights. We name them with two subscripts:

- First subscript: which gate — $f$ (forget), $i$ (input), $c$ (candidate), $o$ (output)
- Second subscript: which input — $h$ (from hidden state) or $x$ (from input vector)

So $W_{fh}$ means: weight matrix for the forget gate, applied to $h_{t-1}$.

---

## A name for "the part before the activation": $z$

Every gate equation has the same shape: take a linear combination of $h_{t-1}$ and $x_t$, then apply an activation ($\sigma$ or $\tanh$). We give the linear part — **before** the activation is applied — its own name: $z$.

For the forget gate, the linear part is called $z_f$:

$$z_f = W_{fh}\, h_{t-1} + W_{fx}\, x_t + b_f$$

$$f_t = \sigma(z_f)$$

$z_f$ is called the **pre-activation**. Same pattern for every gate: $z_i$, $z_c$, $z_o$ are the pre-activations for the input gate, candidate, and output gate respectively. This naming will matter a lot in Chapter 6 (backpropagation) — the gradient with respect to $z_f$ is a distinct, named quantity in the chain rule.

---

## Equation 1 — The Forget Gate

**Intuition:** Look at what just happened ($h_{t-1}$) and what is happening now ($x_t$). Decide what to throw away from memory.

$$z_f = W_{fh}\, h_{t-1} + W_{fx}\, x_t + b_f \qquad f_t = \sigma(z_f)$$

Every symbol, sized:

| Symbol | Size | What it is |
|---|---|---|
| $W_{fh}$ | $H \times H$ | Weights connecting previous hidden state to forget gate |
| $h_{t-1}$ | $H \times 1$ | Previous hidden state |
| $W_{fx}$ | $H \times E$ | Weights connecting current input to forget gate |
| $x_t$ | $E \times 1$ | Current input embedding |
| $b_f$ | $H \times 1$ | Forget gate bias |
| $z_f$ | $H \times 1$ | Pre-activation — the value before $\sigma$ is applied |
| $f_t$ | $H \times 1$ | Forget gate output — values between 0 and 1 |

**Why the sizes work:**
- $W_{fh}\, h_{t-1}$: $(H \times H)(H \times 1) = H \times 1$ ✓
- $W_{fx}\, x_t$: $(H \times E)(E \times 1) = H \times 1$ ✓
- Add them + bias: $(H \times 1) + (H \times 1) + (H \times 1) = H \times 1 = z_f$ ✓
- $\sigma$ applied elementwise → $f_t \in \mathbb{R}^{H \times 1}$, values between 0 and 1 ✓

**Why sigmoid?** A forget gate must output values between 0 and 1. A 0 means "erase this completely." A 1 means "keep this completely." Sigmoid is the only standard activation that does exactly this.

---

## Equation 2 — The Input Gate

**Intuition:** Using the same information ($h_{t-1}$ and $x_t$), decide how much of the new candidate memory to actually write in.

$$z_i = W_{ih}\, h_{t-1} + W_{ix}\, x_t + b_i \qquad i_t = \sigma(z_i)$$

Every symbol, sized:

| Symbol | Size | What it is |
|---|---|---|
| $W_{ih}$ | $H \times H$ | Weights connecting previous hidden state to input gate |
| $W_{ix}$ | $H \times E$ | Weights connecting current input to input gate |
| $b_i$ | $H \times 1$ | Input gate bias |
| $z_i$ | $H \times 1$ | Pre-activation — the value before $\sigma$ is applied |
| $i_t$ | $H \times 1$ | Input gate output — values between 0 and 1 |

Structurally identical to the forget gate. Same sizes, same sigmoid. Different weight matrices, different learned behavior.

---

## Equation 3 — The Candidate Cell State

**Intuition:** Propose a completely new memory based on what just happened and what is happening now. This is not committed to memory yet — the input gate will decide how much of it to actually write.

$$z_c = W_{ch}\, h_{t-1} + W_{cx}\, x_t + b_c \qquad \tilde{c}_t = \tanh(z_c)$$

Every symbol, sized:

| Symbol | Size | What it is |
|---|---|---|
| $W_{ch}$ | $H \times H$ | Weights connecting previous hidden state to candidate |
| $W_{cx}$ | $H \times E$ | Weights connecting current input to candidate |
| $b_c$ | $H \times 1$ | Candidate bias |
| $z_c$ | $H \times 1$ | Pre-activation — the value before $\tanh$ is applied |
| $\tilde{c}_t$ | $H \times 1$ | Candidate cell state — values between $-1$ and $1$ |

**Why $\tanh$ here instead of sigmoid?** The candidate is a proposed memory value — it needs to be able to go positive or negative. $\tanh$ outputs values in $[-1, 1]$ centered at 0. Sigmoid only goes from 0 to 1 and can never propose "remember a negative value."

Gates use sigmoid. Content uses $\tanh$. That is the rule.

---

## The candidate is the RNN — exactly

Compare the two equations side by side:

**RNN hidden state update:**
$$h_t = \tanh(W_h\, h_{t-1} + W_x\, x_t + b)$$

**LSTM candidate cell state:**
$$\tilde{c}_t = \tanh(W_{ch}\, h_{t-1} + W_{cx}\, x_t + b_c)$$

Structurally identical. Same inputs ($h_{t-1}$ and $x_t$), same $\tanh$, same linear transformation. The candidate is exactly what an RNN neuron computes.

The difference is what happens **after**:

- In an RNN, that $\tanh$ output **directly becomes** $h_t$. It replaces everything. No negotiation.
- In an LSTM, that $\tanh$ output is just a **proposal**. It then passes through two more decisions:
    1. The input gate $i_t$ asks — *how much of this proposal do we actually write?*
    2. It gets **added** to what survived from $c_{t-1}$ — it doesn't replace, it amends.

LSTM took the RNN computation, demoted it from "the answer" to "a suggestion," and wrapped it with a negotiation process before anything gets committed to memory.

> The candidate proposes. The gates decide. The cell state records.
>
> In the RNN there are no gates — the candidate IS the answer. That is the entire architectural difference in one sentence.

---

## Equation 4 — Cell State Update

Now combine the first three equations into the new memory:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

Every symbol, sized:

| Symbol | Size | What it is |
|---|---|---|
| $f_t$ | $H \times 1$ | Forget gate — what fraction of old memory to keep |
| $c_{t-1}$ | $H \times 1$ | Old cell state |
| $i_t$ | $H \times 1$ | Input gate — what fraction of candidate to write |
| $\tilde{c}_t$ | $H \times 1$ | Candidate — proposed new memory |
| $c_t$ | $H \times 1$ | New cell state |

No new weight matrices here. Pure arithmetic on vectors already computed.

- $f_t \odot c_{t-1}$ — keep some fraction of old memory, per neuron
- $i_t \odot \tilde{c}_t$ — write some fraction of new candidate, per neuron
- Add them: new memory = surviving old memory + new writing

**Why do we need both $i_t$ and $\tilde{c}_t$?** They do different jobs. $\tilde{c}_t$ says *what* to write. $i_t$ says *how much* of it to write. The candidate can propose something large while the input gate says "not right now" ($i_t \approx 0$) and nothing gets written.

---

## Equation 5 — The Output Gate

**Intuition:** Now that the cell state is updated, decide which parts of it to expose as the hidden state — the thing used for prediction.

$$z_o = W_{oh}\, h_{t-1} + W_{ox}\, x_t + b_o \qquad o_t = \sigma(z_o)$$

Every symbol, sized:

| Symbol | Size | What it is |
|---|---|---|
| $W_{oh}$ | $H \times H$ | Weights connecting previous hidden state to output gate |
| $W_{ox}$ | $H \times E$ | Weights connecting current input to output gate |
| $b_o$ | $H \times 1$ | Output gate bias |
| $z_o$ | $H \times 1$ | Pre-activation — the value before $\sigma$ is applied |
| $o_t$ | $H \times 1$ | Output gate — values between 0 and 1 |

Sigmoid again — because this is a gate controlling how much flows through.

---

## Equation 6 — Hidden State Update

$$h_t = o_t \odot \tanh(c_t)$$

Every symbol, sized:

| Symbol | Size | What it is |
|---|---|---|
| $o_t$ | $H \times 1$ | Output gate |
| $c_t$ | $H \times 1$ | New cell state (just computed above) |
| $\tanh(c_t)$ | $H \times 1$ | Cell state squashed into $[-1, 1]$ |
| $h_t$ | $H \times 1$ | New hidden state |

**Why $\tanh(c_t)$ before multiplying?** The cell state $c_t$ is additive — it can grow large over many timesteps. The $\tanh$ squashes it back into $[-1, 1]$ so the hidden state stays well-behaved. Then the output gate masks it.

$h_t$ goes to the output layer for prediction. $c_t$ continues to the next timestep only — it never directly produces an output.

---

## All six equations in one place

$$z_f = W_{fh}\, h_{t-1} + W_{fx}\, x_t + b_f \qquad f_t = \sigma(z_f)$$

$$z_i = W_{ih}\, h_{t-1} + W_{ix}\, x_t + b_i \qquad i_t = \sigma(z_i)$$

$$z_c = W_{ch}\, h_{t-1} + W_{cx}\, x_t + b_c \qquad \tilde{c}_t = \tanh(z_c)$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$z_o = W_{oh}\, h_{t-1} + W_{ox}\, x_t + b_o \qquad o_t = \sigma(z_o)$$

$$h_t = o_t \odot \tanh(c_t)$$

Note the pattern: every gate and the candidate first compute a **pre-activation** $z_\bullet$ (a linear combination of $h_{t-1}$ and $x_t$, plus bias), then apply an activation ($\sigma$ for gates, $\tanh$ for the candidate). The four pre-activations $z_f, z_i, z_c, z_o$ are the quantities that Chapter 6's backpropagation differentiates through.

---

## Every learnable parameter in the LSTM

| Parameter | Size | Count (if $H=128$, $E=64$) |
|---|---|---|
| $W_{fh}$ | $H \times H$ | $16{,}384$ |
| $W_{fx}$ | $H \times E$ | $8{,}192$ |
| $b_f$ | $H \times 1$ | $128$ |
| $W_{ih}$ | $H \times H$ | $16{,}384$ |
| $W_{ix}$ | $H \times E$ | $8{,}192$ |
| $b_i$ | $H \times 1$ | $128$ |
| $W_{ch}$ | $H \times H$ | $16{,}384$ |
| $W_{cx}$ | $H \times E$ | $8{,}192$ |
| $b_c$ | $H \times 1$ | $128$ |
| $W_{oh}$ | $H \times H$ | $16{,}384$ |
| $W_{ox}$ | $H \times E$ | $8{,}192$ |
| $b_o$ | $H \times 1$ | $128$ |
| **Total** | | **~98,816** |

---

## The sigmoid vs tanh rule

> **Gates use $\sigma$** — output 0 to 1, act as valves.
> **Content uses $\tanh$** — output $-1$ to $1$, can store positive or negative values.

Gates: $f_t$, $i_t$, $o_t$ — all sigmoid.
Content: $\tilde{c}_t$, $\tanh(c_t)$ — all tanh.

---

## Summary

Six equations, executed in order, every timestep:

| Step | Equation | Role |
|---|---|---|
| 1 | $z_f = W_{fh} h_{t-1} + W_{fx} x_t + b_f$, then $f_t = \sigma(z_f)$ | Decide what to erase |
| 2 | $z_i = W_{ih} h_{t-1} + W_{ix} x_t + b_i$, then $i_t = \sigma(z_i)$ | Decide how much to write |
| 3 | $z_c = W_{ch} h_{t-1} + W_{cx} x_t + b_c$, then $\tilde{c}_t = \tanh(z_c)$ | Propose what to write (the RNN computation) |
| 4 | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | Update memory |
| 5 | $z_o = W_{oh} h_{t-1} + W_{ox} x_t + b_o$, then $o_t = \sigma(z_o)$ | Decide what to expose |
| 6 | $h_t = o_t \odot \tanh(c_t)$ | Produce output |

**The $z$ notation at a glance:**

| Pre-activation | Definition | Activation |
|---|---|---|
| $z_f$ | $W_{fh} h_{t-1} + W_{fx} x_t + b_f$ | $f_t = \sigma(z_f)$ |
| $z_i$ | $W_{ih} h_{t-1} + W_{ix} x_t + b_i$ | $i_t = \sigma(z_i)$ |
| $z_c$ | $W_{ch} h_{t-1} + W_{cx} x_t + b_c$ | $\tilde{c}_t = \tanh(z_c)$ |
| $z_o$ | $W_{oh} h_{t-1} + W_{ox} x_t + b_o$ | $o_t = \sigma(z_o)$ |

These four $z$'s are exactly the pre-activation gradients ($\partial L/\partial z_f$, etc.) you'll compute in Chapter 6.
