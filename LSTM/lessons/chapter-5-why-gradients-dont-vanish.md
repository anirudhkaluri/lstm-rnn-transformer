# Chapter 5 — Why Gradients Don't Vanish

---

## First, remember why the RNN gradient dies

In an RNN, the hidden state is:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

During BPTT, to get the gradient at timestep $t$, you need to push it back through every earlier step. Each step requires:

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(z_t)) \cdot W_h$$

where $z_t = W_h h_{t-1} + W_x x_t + b$ is the pre-activation, and $\tanh'(z) = 1 - \tanh^2(z)$.

$\tanh'(z)$ is always between 0 and 1 — it can never exceed 1. So every step, the gradient gets multiplied by $W_h$ and shrunk by $\tanh'$.

Over $T$ timesteps, the full gradient chain is:

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \text{diag}(\tanh'(z_k)) \cdot W_h$$

This is a product of $T - t$ matrices, each of which shrinks the gradient. With $T - t = 10$ steps, the gradient is essentially zero. The network cannot learn long-range dependencies.

---

## Why does $\frac{\partial h_t}{\partial h_{t-1}}$ even appear? (it is not a trainable parameter)

This is a natural question — $h_t$ is not a weight we update, so why differentiate with respect to it at all?

**You want to update $W_h$. That requires $\frac{\partial L}{\partial W_h}$.**

$W_h$ is **shared across every timestep** — the same matrix is used at $t=1$, $t=2$, $t=3$, every step. So the total gradient is a sum over all timesteps:

$$\frac{\partial L}{\partial W_h} = \sum_{\text{all } t} \text{(contribution from timestep } t\text{)}$$

Say you have 3 timesteps:

$$h_1 = \tanh(W_h h_0 + W_x x_1 + b)$$
$$h_2 = \tanh(W_h h_1 + W_x x_2 + b)$$
$$h_3 = \tanh(W_h h_2 + W_x x_3 + b)$$

The loss at $t=3$ is $L_3$, depending on $h_3$. Ask: how much does $W_h$ **at timestep 1** affect $L_3$?

The path is:

$$W_h \rightarrow h_1 \rightarrow h_2 \rightarrow h_3 \rightarrow L_3$$

By the chain rule:

$$\frac{\partial L_3}{\partial W_h^{(t=1)}} = \frac{\partial L_3}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_h}$$

The middle part — $\frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1}$ — **is exactly where $\frac{\partial h_t}{\partial h_{t-1}}$ comes from.**

You don't need it to "update $h$" — $h$ isn't a parameter. You need it to **route the gradient from a late loss back to an early $W_h$** through the chain of hidden states. Each crossing of one hidden state to the previous one costs one $\frac{\partial h_t}{\partial h_{t-1}}$.

---

## Expanding through the raw scores and $z_t$ explicitly

The full BPTT path to $W_h$, written through every intermediate variable (raw = pre-softmax scores, $z_t$ = pre-activation), looks like this for two timesteps' contributions:

$$\frac{\partial L}{\partial W_h} = \underbrace{\frac{\partial L}{\partial \text{raw}} \cdot \frac{\partial \text{raw}}{\partial h_t} \cdot \frac{\partial h_t}{\partial z_t} \cdot \frac{\partial z_t}{\partial W_h}}_{\text{timestep } t\text{'s direct contribution}} + \underbrace{\frac{\partial L}{\partial \text{raw}} \cdot \frac{\partial \text{raw}}{\partial h_t} \cdot \frac{\partial h_t}{\partial z_t} \cdot \frac{\partial z_t}{\partial h_{t-1}} \cdot \frac{\partial h_{t-1}}{\partial z_{t-1}} \cdot \frac{\partial z_{t-1}}{\partial W_h}}_{\text{timestep } t{-}1\text{'s contribution}}$$

Each piece evaluates to something concrete:

$$\frac{\partial z_t}{\partial W_h} = h_{t-1} \quad \text{(because } z_t = W_h h_{t-1} + W_x x_t + b\text{)}$$

$$\frac{\partial h_t}{\partial z_t} = \tanh'(z_t)$$

$$\frac{\partial z_t}{\partial h_{t-1}} = W_h \quad \text{(because } z_t = W_h h_{t-1} + \ldots\text{)}$$

$$\frac{\partial h_{t-1}}{\partial z_{t-1}} = \tanh'(z_{t-1})$$

So the second term contains:

$$\ldots \cdot \tanh'(z_t) \cdot W_h \cdot \tanh'(z_{t-1}) \cdot h_{t-2}$$

**This confirms $\frac{\partial h_t}{\partial h_{t-1}}$ is just two of these steps collapsed into one:**

$$\frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial h_t}{\partial z_t} \cdot \frac{\partial z_t}{\partial h_{t-1}} = \tanh'(z_t) \cdot W_h$$

Both forms are correct — the explicit form keeps $z_t$ visible, the collapsed form skips it.

**Why the gradient vanishes, in this notation:** for $k$ steps back, the chain picks up one $W_h$ and one $\tanh'$ per step:

$$\underbrace{\tanh'(z_t) \cdot W_h}_{\text{step 1 back}} \cdot \underbrace{\tanh'(z_{t-1}) \cdot W_h}_{\text{step 2 back}} \cdot \underbrace{\tanh'(z_{t-2}) \cdot W_h}_{\text{step 3 back}} \cdots$$

$\tanh'$ is always between 0 and 1, so every step back shrinks the gradient. By step 7 or 8, it is nearly zero.

---

## Now differentiate through the LSTM cell state

The cell state update is:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

We want $\frac{\partial c_t}{\partial c_{t-1}}$.

Write it out for a single element $j$:

$$c_{t,j} = f_{t,j} \cdot c_{t-1,j} + i_{t,j} \cdot \tilde{c}_{t,j}$$

Take the derivative with respect to $c_{t-1,j}$:

$$\frac{\partial c_{t,j}}{\partial c_{t-1,j}} = f_{t,j}$$

That's it. Just $f_{t,j}$.

The second term $i_{t,j} \cdot \tilde{c}_{t,j}$ vanishes in this derivative because it has no direct dependence on $c_{t-1,j}$ — it depends on $x_t$ and $h_{t-1}$, not $c_{t-1}$ directly.

Cross-terms ($\frac{\partial c_{t,j}}{\partial c_{t-1,k}}$ for $k \neq j$) are zero because $\odot$ is elementwise — position $j$ only touches position $j$.

Written as a full Jacobian matrix:

$$\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t)$$

A diagonal matrix with the forget gate values on the diagonal. No $W_h$. No $\tanh'$. Just $f_t$.

---

## Extend this over many timesteps

To push a gradient from timestep $T$ all the way back to timestep $t$ through the cell state, multiply these Jacobians together:

$$\frac{\partial c_T}{\partial c_t} = \prod_{k=t+1}^{T} \text{diag}(f_k)$$

Since these are all diagonal matrices, multiplying them together is just elementwise multiplication. For element $j$:

$$\frac{\partial c_{T,j}}{\partial c_{t,j}} = \prod_{k=t+1}^{T} f_{k,j}$$

---

## Side-by-side comparison

| | RNN | LSTM cell state path |
|---|---|---|
| One-step gradient | $\text{diag}(\tanh'(z_t)) \cdot W_h$ | $\text{diag}(f_t)$ |
| Multi-step gradient | $\prod_{k} \text{diag}(\tanh'(z_k)) \cdot W_h$ | $\prod_{k} f_{k,j}$ |
| Contains $W_h$? | Yes — full matrix multiply every step | No |
| Contains $\tanh'$? | Yes — always shrinks, always $< 1$ | No |
| Can the network control it? | No — $W_h$ serves too many purposes | Yes — network learns $f_t$ |

The RNN gradient involves a full matrix multiplication at every step. That matrix has to simultaneously (1) pass gradients back without vanishing, (2) transform the hidden state usefully, and (3) not explode. These goals conflict.

The LSTM cell state gradient is just a product of forget gate values — scalars between 0 and 1 — for each neuron independently.

---

## The numbers tell the story

Say we are pushing a gradient 10 timesteps back.

**RNN** — typical scenario where each step shrinks the gradient by ~0.25 (combining $\tanh'$ and $W_h$ effects):

$$0.25^{10} = 0.000001$$

One millionth of the original signal. Dead.

**LSTM** — network has learned to keep the forget gate at $f_{t,j} = 0.9$ for an important memory dimension:

$$0.9^{10} = 0.349$$

35% of the original signal survives. Meaningful.

**LSTM** — network sets forget gate to $f_{t,j} = 1.0$ for a critical long-range dependency:

$$1.0^{10} = 1.000$$

100% of the gradient survives. Perfect flow.

---

## The crucial insight — controllable vs uncontrollable

The forget gate $f_t$ is itself computed by the network:

$$f_t = \sigma(W_{fh} h_{t-1} + W_{fx} x_t + b_f)$$

The network learns $W_{fh}$, $W_{fx}$, and $b_f$. So the network is learning to control its own gradient flow.

When the network needs to remember something for a long time, it learns to produce $f_t \approx 1$ for that memory dimension. The gradient highway stays open.

When the network wants to forget, it learns $f_t \approx 0$. The gradient through that dimension is killed — but that is intentional. The network decided that information is no longer useful.

In an RNN, the gradient is controlled by $W_h$ — a single matrix serving every purpose at once (transform the state, pass gradients, mix information). It cannot independently decide "keep this dimension's gradient alive" without affecting everything else. The LSTM separates these concerns. The forget gate's only job is to control what survives.

---

## Why the cell state is called the "constant error carousel"

This term comes from the original 1997 LSTM paper by Hochreiter and Schmidhuber. "Constant" refers to the fact that when $f_t = 1$, the gradient flowing through the cell state is exactly 1 — constant — at every step. No decay, no explosion. A flat highway carrying the error signal from the loss at time $T$ all the way back to time $t=1$ unmodified.

The RNN has no such highway. Every path the gradient takes involves multiplying by $W_h$ and $\tanh'$. There is no bypass.

---

## Summary

| Question | Answer |
|---|---|
| Why does the RNN gradient vanish? | Products of $W_h \cdot \tanh'$ at every step — each shrinks the signal |
| Why does $\frac{\partial h_t}{\partial h_{t-1}}$ appear if $h$ isn't a parameter? | It's the bridge routing the gradient from a late loss back to an early $W_h$ |
| What is the LSTM cell state gradient? | $\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t)$ — just the forget gate values |
| Why is this better? | No full matrix multiply, no $\tanh'$, just scalar values the network controls |
| What does the network learn? | To set $f_t \approx 1$ when long-range memory is needed, $f_t \approx 0$ when not |
| Can the LSTM gradient still vanish? | Yes — if $f_t \approx 0$. But that is intentional forgetting, not architectural failure |
| What is the constant error carousel? | The cell state path where gradient is exactly 1 when $f_t = 1$ |
