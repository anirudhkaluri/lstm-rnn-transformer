# Chapter 4 — Forward Pass With Real Numbers

---

## Setup

We use a tiny LSTM with $H = 2$ hidden units and $E = 2$ embedding dimensions. This keeps every matrix small enough to compute by hand while showing every step exactly as it happens in a real network.

**Inputs at timestep $t$:**

$$x_t = \begin{bmatrix} 1 \\ 0 \end{bmatrix} \quad h_{t-1} = \begin{bmatrix} 0.5 \\ -0.3 \end{bmatrix} \quad c_{t-1} = \begin{bmatrix} 0.7 \\ -0.2 \end{bmatrix}$$

**All weight matrices** (these would be learned during training — we just pick small numbers here):

$$W_{fh} = \begin{bmatrix} 0.4 & 0.2 \\ -0.1 & 0.3 \end{bmatrix} \quad W_{fx} = \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & -0.4 \end{bmatrix} \quad b_f = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$$

$$W_{ih} = \begin{bmatrix} 0.3 & -0.2 \\ 0.5 & 0.1 \end{bmatrix} \quad W_{ix} = \begin{bmatrix} 0.4 & 0.2 \\ -0.3 & 0.6 \end{bmatrix} \quad b_i = \begin{bmatrix} 0.0 \\ 0.2 \end{bmatrix}$$

$$W_{ch} = \begin{bmatrix} 0.6 & 0.1 \\ -0.2 & 0.4 \end{bmatrix} \quad W_{cx} = \begin{bmatrix} 0.3 & -0.1 \\ 0.5 & 0.2 \end{bmatrix} \quad b_c = \begin{bmatrix} 0.0 \\ 0.1 \end{bmatrix}$$

$$W_{oh} = \begin{bmatrix} 0.2 & 0.5 \\ 0.3 & -0.1 \end{bmatrix} \quad W_{ox} = \begin{bmatrix} 0.6 & 0.1 \\ -0.2 & 0.4 \end{bmatrix} \quad b_o = \begin{bmatrix} 0.1 \\ 0.0 \end{bmatrix}$$

---

## Step 1 — Forget Gate

$$f_t = \sigma(W_{fh}\, h_{t-1} + W_{fx}\, x_t + b_f)$$

**Compute $W_{fh}\, h_{t-1}$:**

$$\begin{bmatrix} 0.4 & 0.2 \\ -0.1 & 0.3 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.3 \end{bmatrix} = \begin{bmatrix} (0.4)(0.5) + (0.2)(-0.3) \\ (-0.1)(0.5) + (0.3)(-0.3) \end{bmatrix} = \begin{bmatrix} 0.20 - 0.06 \\ -0.05 - 0.09 \end{bmatrix} = \begin{bmatrix} 0.14 \\ -0.14 \end{bmatrix}$$

**Compute $W_{fx}\, x_t$:**

$$\begin{bmatrix} 0.5 & 0.1 \\ 0.2 & -0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} (0.5)(1) + (0.1)(0) \\ (0.2)(1) + (-0.4)(0) \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}$$

**Add everything:**

$$z_f = \begin{bmatrix} 0.14 \\ -0.14 \end{bmatrix} + \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix} + \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix} = \begin{bmatrix} 0.74 \\ -0.04 \end{bmatrix}$$

**Apply sigmoid** — recall $\sigma(z) = \frac{1}{1 + e^{-z}}$:

$$\sigma(0.74) = \frac{1}{1 + e^{-0.74}} = \frac{1}{1 + 0.477} = \frac{1}{1.477} = 0.677$$

$$\sigma(-0.04) = \frac{1}{1 + e^{0.04}} = \frac{1}{1 + 1.041} = \frac{1}{2.041} = 0.490$$

$$\boxed{f_t = \begin{bmatrix} 0.677 \\ 0.490 \end{bmatrix}}$$

**What this means:** Neuron 1 will keep 67.7% of its old memory. Neuron 2 will keep 49.0% of its old memory.

---

## Step 2 — Input Gate

$$i_t = \sigma(W_{ih}\, h_{t-1} + W_{ix}\, x_t + b_i)$$

**Compute $W_{ih}\, h_{t-1}$:**

$$\begin{bmatrix} 0.3 & -0.2 \\ 0.5 & 0.1 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.3 \end{bmatrix} = \begin{bmatrix} (0.3)(0.5) + (-0.2)(-0.3) \\ (0.5)(0.5) + (0.1)(-0.3) \end{bmatrix} = \begin{bmatrix} 0.15 + 0.06 \\ 0.25 - 0.03 \end{bmatrix} = \begin{bmatrix} 0.21 \\ 0.22 \end{bmatrix}$$

**Compute $W_{ix}\, x_t$:**

$$\begin{bmatrix} 0.4 & 0.2 \\ -0.3 & 0.6 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.4 \\ -0.3 \end{bmatrix}$$

**Add everything:**

$$z_i = \begin{bmatrix} 0.21 \\ 0.22 \end{bmatrix} + \begin{bmatrix} 0.4 \\ -0.3 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.2 \end{bmatrix} = \begin{bmatrix} 0.61 \\ 0.12 \end{bmatrix}$$

**Apply sigmoid:**

$$\sigma(0.61) = \frac{1}{1 + e^{-0.61}} = \frac{1}{1 + 0.543} = \frac{1}{1.543} = 0.648$$

$$\sigma(0.12) = \frac{1}{1 + e^{-0.12}} = \frac{1}{1 + 0.887} = \frac{1}{1.887} = 0.530$$

$$\boxed{i_t = \begin{bmatrix} 0.648 \\ 0.530 \end{bmatrix}}$$

**What this means:** Neuron 1 will write 64.8% of the candidate into memory. Neuron 2 will write 53.0%.

---

## Step 3 — Candidate Cell State

$$\tilde{c}_t = \tanh(W_{ch}\, h_{t-1} + W_{cx}\, x_t + b_c)$$

**Compute $W_{ch}\, h_{t-1}$:**

$$\begin{bmatrix} 0.6 & 0.1 \\ -0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.3 \end{bmatrix} = \begin{bmatrix} (0.6)(0.5) + (0.1)(-0.3) \\ (-0.2)(0.5) + (0.4)(-0.3) \end{bmatrix} = \begin{bmatrix} 0.30 - 0.03 \\ -0.10 - 0.12 \end{bmatrix} = \begin{bmatrix} 0.27 \\ -0.22 \end{bmatrix}$$

**Compute $W_{cx}\, x_t$:**

$$\begin{bmatrix} 0.3 & -0.1 \\ 0.5 & 0.2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.3 \\ 0.5 \end{bmatrix}$$

**Add everything:**

$$z_c = \begin{bmatrix} 0.27 \\ -0.22 \end{bmatrix} + \begin{bmatrix} 0.3 \\ 0.5 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.1 \end{bmatrix} = \begin{bmatrix} 0.57 \\ 0.38 \end{bmatrix}$$

**Apply tanh:**

$$\tanh(0.57) \approx 0.515 \qquad \tanh(0.38) \approx 0.362$$

$$\boxed{\tilde{c}_t = \begin{bmatrix} 0.515 \\ 0.362 \end{bmatrix}}$$

**What this means:** The network proposes storing $+0.515$ in neuron 1 and $+0.362$ in neuron 2. These are just proposals — the input gate decides how much of each actually gets written. This computation is structurally identical to an RNN — same inputs, same $\tanh$. This is the RNN inside the LSTM.

---

## Step 4 — Cell State Update

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Compute $f_t \odot c_{t-1}$** — how much old memory survives:

$$\begin{bmatrix} 0.677 \\ 0.490 \end{bmatrix} \odot \begin{bmatrix} 0.7 \\ -0.2 \end{bmatrix} = \begin{bmatrix} (0.677)(0.7) \\ (0.490)(-0.2) \end{bmatrix} = \begin{bmatrix} 0.474 \\ -0.098 \end{bmatrix}$$

**Compute $i_t \odot \tilde{c}_t$** — how much new information gets written:

$$\begin{bmatrix} 0.648 \\ 0.530 \end{bmatrix} \odot \begin{bmatrix} 0.515 \\ 0.362 \end{bmatrix} = \begin{bmatrix} (0.648)(0.515) \\ (0.530)(0.362) \end{bmatrix} = \begin{bmatrix} 0.334 \\ 0.192 \end{bmatrix}$$

**Add them:**

$$c_t = \begin{bmatrix} 0.474 \\ -0.098 \end{bmatrix} + \begin{bmatrix} 0.334 \\ 0.192 \end{bmatrix} = \boxed{\begin{bmatrix} 0.808 \\ 0.094 \end{bmatrix}}$$

**What happened to each neuron:**

| | Neuron 1 | Neuron 2 |
|---|---|---|
| Old memory $c_{t-1}$ | $0.700$ | $-0.200$ |
| Surviving old memory | $0.474$ (kept 67.7%) | $-0.098$ (kept 49.0%) |
| New writing | $+0.334$ | $+0.192$ |
| New cell state $c_t$ | $0.808$ | $0.094$ |

Neuron 2 had negative memory ($-0.2$), kept about half of it ($-0.098$), then added positive new information ($+0.192$), and ended up slightly positive ($0.094$). The memory flipped sign.

---

## Step 5 — Output Gate

$$o_t = \sigma(W_{oh}\, h_{t-1} + W_{ox}\, x_t + b_o)$$

**Compute $W_{oh}\, h_{t-1}$:**

$$\begin{bmatrix} 0.2 & 0.5 \\ 0.3 & -0.1 \end{bmatrix} \begin{bmatrix} 0.5 \\ -0.3 \end{bmatrix} = \begin{bmatrix} (0.2)(0.5) + (0.5)(-0.3) \\ (0.3)(0.5) + (-0.1)(-0.3) \end{bmatrix} = \begin{bmatrix} 0.10 - 0.15 \\ 0.15 + 0.03 \end{bmatrix} = \begin{bmatrix} -0.05 \\ 0.18 \end{bmatrix}$$

**Compute $W_{ox}\, x_t$:**

$$\begin{bmatrix} 0.6 & 0.1 \\ -0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.6 \\ -0.2 \end{bmatrix}$$

**Add everything:**

$$z_o = \begin{bmatrix} -0.05 \\ 0.18 \end{bmatrix} + \begin{bmatrix} 0.6 \\ -0.2 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.65 \\ -0.02 \end{bmatrix}$$

**Apply sigmoid:**

$$\sigma(0.65) = \frac{1}{1 + e^{-0.65}} = \frac{1}{1 + 0.522} = \frac{1}{1.522} = 0.657$$

$$\sigma(-0.02) = \frac{1}{1 + e^{0.02}} = \frac{1}{1 + 1.020} = \frac{1}{2.020} = 0.495$$

$$\boxed{o_t = \begin{bmatrix} 0.657 \\ 0.495 \end{bmatrix}}$$

**What this means:** Expose 65.7% of neuron 1's memory as output. Expose 49.5% of neuron 2's memory as output.

---

## Step 6 — Hidden State

$$h_t = o_t \odot \tanh(c_t)$$

**Compute $\tanh(c_t)$:**

$$\tanh(0.808) \approx 0.668 \qquad \tanh(0.094) \approx 0.094$$

$$\tanh(c_t) = \begin{bmatrix} 0.668 \\ 0.094 \end{bmatrix}$$

**Apply output gate:**

$$h_t = \begin{bmatrix} 0.657 \\ 0.495 \end{bmatrix} \odot \begin{bmatrix} 0.668 \\ 0.094 \end{bmatrix} = \begin{bmatrix} (0.657)(0.668) \\ (0.495)(0.094) \end{bmatrix} = \boxed{\begin{bmatrix} 0.439 \\ 0.047 \end{bmatrix}}$$

---

## Everything in one place

| Step | Computation | Result |
|---|---|---|
| Forget gate | $\sigma(W_{fh} h_{t-1} + W_{fx} x_t + b_f)$ | $[0.677,\ 0.490]^T$ |
| Input gate | $\sigma(W_{ih} h_{t-1} + W_{ix} x_t + b_i)$ | $[0.648,\ 0.530]^T$ |
| Candidate | $\tanh(W_{ch} h_{t-1} + W_{cx} x_t + b_c)$ | $[0.515,\ 0.362]^T$ |
| Cell state | $f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | $[0.808,\ 0.094]^T$ |
| Output gate | $\sigma(W_{oh} h_{t-1} + W_{ox} x_t + b_o)$ | $[0.657,\ 0.495]^T$ |
| Hidden state | $o_t \odot \tanh(c_t)$ | $[0.439,\ 0.047]^T$ |

**What leaves this timestep:**
- $h_t = [0.439,\ 0.047]^T$ → goes to the output layer AND to the next timestep
- $c_t = [0.808,\ 0.094]^T$ → goes to the next timestep only

---

## The key observation

Look at the cell state before and after:

$$c_{t-1} = \begin{bmatrix} 0.7 \\ -0.2 \end{bmatrix} \longrightarrow c_t = \begin{bmatrix} 0.808 \\ 0.094 \end{bmatrix}$$

The values changed but not drastically. No $\tanh$ was applied to the cell state itself during the update — just an elementwise multiply and an add. This is what keeps the gradient alive during backpropagation. The cell state is a slow, controlled accumulation. The hidden state $h_t$ is the fast, filtered read of it.
