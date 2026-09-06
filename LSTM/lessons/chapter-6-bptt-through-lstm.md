# Chapter 6 — BPTT Through an LSTM

---

## The setup: two gradients flow backward, not one

In the RNN, BPTT pushes back a single gradient: ∂L/∂h_t

In the LSTM, **two** gradients flow backward through time, because there are two state vectors:

- dh_t = ∂L/∂h_t , size (H × 1) — gradient with respect to the hidden state
- dc_t = ∂L/∂c_t , size (H × 1) — gradient with respect to the cell state

At each timestep, going backward, you receive dh_t and dc_t (accumulated from the future), and your job is to:
1. Use them to compute gradients for all the weight matrices at this timestep
2. Produce dh_{t-1} and dc_{t-1} to pass further back

We will derive every one of these. Then we'll run the numbers using the exact values computed in Chapter 4.

---

## Step 1 — Backprop through h_t = o_t ⊙ tanh(c_t)

Two things depend on c_t and o_t: the hidden state h_t. We have an incoming gradient dh_t. Split it into its two paths.

**Gradient into the output gate:**

∂L/∂o_t = dh_t ⊙ tanh(c_t)

**Why:** h_t = o_t ⊙ tanh(c_t). Treating tanh(c_t) as a constant for this branch, ∂h_t/∂o_t = tanh(c_t). Multiply by the incoming gradient dh_t (chain rule, elementwise).

**Gradient into the cell state (additional contribution):**

dh_t → c_t contributes: dh_t ⊙ o_t ⊙ (1 − tanh²(c_t))

**Why:** ∂h_t/∂c_t = o_t ⊙ tanh'(c_t) = o_t ⊙ (1 − tanh²(c_t))

This gets **added** to whatever gradient c_t already received from c_{t+1} (computed in the previous backward step, since we move right-to-left). Call that incoming piece dc_t_future:

dc_t_total = dc_t_future + dh_t ⊙ o_t ⊙ (1 − tanh²(c_t))

This is the **single most important line** in this chapter — c_t collects gradient from two places: the future cell state c_{t+1}, and the current hidden state h_t.

---

## Step 2 — Backprop through c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

This single equation has **four** outputs to differentiate with respect to: f_t, c_{t-1}, i_t, c̃_t. Each is a simple elementwise product, so each derivative is just "the other factor."

∂L/∂f_t = dc_t_total ⊙ c_{t-1}

∂L/∂c_{t-1} = dc_t_total ⊙ f_t   ← this is dc_{t-1}_future for the next step back

∂L/∂i_t = dc_t_total ⊙ c̃_t

∂L/∂c̃_t = dc_t_total ⊙ i_t

This is exactly the ∂c_t/∂c_{t-1} = diag(f_t) result from Chapter 5 — here it is in context: ∂L/∂c_{t-1} = dc_t_total ⊙ f_t is literally dc_t_total multiplied by that Jacobian.

---

## Step 3 — Backprop through the activation functions

We now have gradients with respect to f_t, i_t, c̃_t, o_t — but these are **post-activation** values. To reach the weight matrices, we need **pre-activation** gradients (with respect to z_f, z_i, z_c, z_o, the values before σ/tanh were applied).

Recall: for f_t = σ(z_f), the derivative σ'(z_f) = f_t ⊙ (1 − f_t) — written in terms of the *output*, not the input. Same trick for tanh.

∂L/∂z_f = ∂L/∂f_t ⊙ f_t ⊙ (1 − f_t)

∂L/∂z_i = ∂L/∂i_t ⊙ i_t ⊙ (1 − i_t)

∂L/∂z_o = ∂L/∂o_t ⊙ o_t ⊙ (1 − o_t)

∂L/∂z_c = ∂L/∂c̃_t ⊙ (1 − c̃_t²)

---

## Step 4 — Gradients for the weight matrices

Recall z_f = W_fh · h_{t-1} + W_fx · x_t + b_f (and similarly for i, c, o). The gradient with respect to a weight matrix is an **outer product** between the pre-activation gradient and the input that was multiplied by that matrix:

∂L/∂W_fh = (∂L/∂z_f) · h_{t-1}ᵀ

∂L/∂W_fx = (∂L/∂z_f) · x_tᵀ

∂L/∂b_f = ∂L/∂z_f

Same pattern for W_ih, W_ix, b_i, for W_ch, W_cx, b_c, and for W_oh, W_ox, b_o.

**Sizes check:** ∂L/∂z_f ∈ ℝ^(H×1), h_{t-1}ᵀ ∈ ℝ^(1×H). Outer product gives H×H — matching W_fh. ✓

---

## Step 5 — Gradient with respect to h_{t-1} (to continue backward)

Each of the four pre-activation gradients (z_f, z_i, z_c, z_o) was computed using h_{t-1} via W_•h. All four paths contribute to dh_{t-1}:

∂L/∂h_{t-1} = W_fhᵀ·(∂L/∂z_f) + W_ihᵀ·(∂L/∂z_i) + W_chᵀ·(∂L/∂z_c) + W_ohᵀ·(∂L/∂z_o)

This becomes dh_{t-1} — combined with whatever direct gradient arrives at h_{t-1} from the output layer at timestep t-1 — for the next iteration of backprop.

Together with ∂L/∂c_{t-1} from Step 2, this gives the complete pair (dh_{t-1}, dc_{t-1}) needed to continue backward.

---

## Full numeric walkthrough — continuing Chapter 4's example

Recall from Chapter 4:

h_{t-1} = [0.5, -0.3]      x_t = [1, 0]      c_{t-1} = [0.7, -0.2]

f_t = [0.677, 0.490]
i_t = [0.648, 0.530]
c̃_t = [0.515, 0.362]
c_t = [0.808, 0.094]
o_t = [0.657, 0.495]
tanh(c_t) = [0.668, 0.094]

(all vectors written as [top, bottom] for a 2-element column)

**Assume** these gradients arrive from "later" (from the loss at this timestep, and from c_{t+1}):

dh_t = [0.3, -0.2]
dc_t_future = [0.1, 0.05]

---

### Step 1 — Output gate gradient and cell state contribution

∂L/∂o_t = dh_t ⊙ tanh(c_t)
        = [0.3 × 0.668, -0.2 × 0.094]
        = [0.200, -0.0188]

1 − tanh²(c_t) = [1 − 0.668², 1 − 0.094²] = [0.554, 0.991]

dh_t ⊙ o_t ⊙ (1 − tanh²(c_t))
  = [0.3 × 0.657 × 0.554, -0.2 × 0.495 × 0.991]
  = [0.109, -0.098]

dc_t_total = [0.1, 0.05] + [0.109, -0.098] = [0.209, -0.048]

---

### Step 2 — Gradients for f_t, c_{t-1}, i_t, c̃_t

∂L/∂f_t = dc_t_total ⊙ c_{t-1}
        = [0.209 × 0.7, -0.048 × (-0.2)]
        = [0.146, 0.0096]

∂L/∂c_{t-1} = dc_t_total ⊙ f_t
            = [0.209 × 0.677, -0.048 × 0.490]
            = [0.1415, -0.0235]

∂L/∂i_t = dc_t_total ⊙ c̃_t
        = [0.209 × 0.515, -0.048 × 0.362]
        = [0.1076, -0.0174]

∂L/∂c̃_t = dc_t_total ⊙ i_t
         = [0.209 × 0.648, -0.048 × 0.530]
         = [0.1354, -0.0254]

**∂L/∂c_{t-1} = [0.1415, -0.0235] is exactly dc_{t-1}_future for the next backward step.**

---

### Step 3 — Pre-activation gradients

∂L/∂z_f = ∂L/∂f_t ⊙ f_t ⊙ (1 − f_t)
        = [0.146 × 0.677 × 0.323, 0.0096 × 0.490 × 0.510]
        = [0.0319, 0.0024]

∂L/∂z_i = ∂L/∂i_t ⊙ i_t ⊙ (1 − i_t)
        = [0.1076 × 0.648 × 0.352, -0.0174 × 0.530 × 0.470]
        = [0.0245, -0.0043]

∂L/∂z_o = ∂L/∂o_t ⊙ o_t ⊙ (1 − o_t)
        = [0.200 × 0.657 × 0.343, -0.0188 × 0.495 × 0.505]
        = [0.0451, -0.0047]

∂L/∂z_c = ∂L/∂c̃_t ⊙ (1 − c̃_t²)
        = [0.1354 × (1 − 0.515²), -0.0254 × (1 − 0.362²)]
        = [0.0995, -0.0221]

---

### Step 4 — Example weight gradient (W_fh and b_f)

∂L/∂W_fh = (∂L/∂z_f) · h_{t-1}ᵀ

  = [0.0319, 0.0024] (as a column) times [0.5, -0.3] (as a row)

  =  | 0.0319 × 0.5    0.0319 × (-0.3) |     | 0.0160   -0.0096 |
     | 0.0024 × 0.5    0.0024 × (-0.3) |  =  | 0.0012   -0.0007 |

∂L/∂b_f = ∂L/∂z_f = [0.0319, 0.0024]

(Every other weight matrix's gradient — W_fx, W_ih, W_ix, W_ch, W_cx, W_oh, W_ox and their biases — follows the exact same outer-product pattern using ∂L/∂z_i, ∂L/∂z_c, ∂L/∂z_o respectively.)

---

### Step 5 — Gradient into h_{t-1}

Using W_fhᵀ, W_ihᵀ, W_chᵀ, W_ohᵀ from Chapter 4:

W_fhᵀ · (∂L/∂z_f) = [0.0125, 0.0071]
W_ihᵀ · (∂L/∂z_i) = [0.0052, -0.0053]
W_chᵀ · (∂L/∂z_c) = [0.0641, 0.0011]
W_ohᵀ · (∂L/∂z_o) = [0.0076, 0.0230]

∂L/∂h_{t-1} = sum of all four
            = [0.0125+0.0052+0.0641+0.0076,  0.0071-0.0053+0.0011+0.0230]
            = [0.0894, 0.0259]

---

## What just happened — the two values that travel backward

After processing timestep t, we hand off to timestep t-1:

dh_{t-1} = [0.0894, 0.0259]
dc_{t-1}_future = [0.1415, -0.0235]

At timestep t-1, this exact same six-step process repeats: split into output-gate and cell-state gradients, distribute through the four gate equations, push through activations, accumulate weight gradients, produce dh_{t-2} and dc_{t-2}.

**Note the asymmetry:** dh_{t-1} involves four matrix multiplications (W_fhᵀ, W_ihᵀ, W_chᵀ, W_ohᵀ) — this path can still shrink, similar to the RNN. But dc_{t-1} is just dc_t_total ⊙ f_t — the gradient highway from Chapter 5. Both paths exist; the cell-state path is the one that prevents total vanishing.

---

## Truncated BPTT

For long sequences (say 1000 timesteps), computing the full backward pass all the way to t=1 is too expensive — you'd need to store every intermediate value from every timestep.

**Truncated BPTT** caps how far back gradients are propagated — e.g., only 20 or 50 steps. The forward pass still runs over the whole sequence (carrying h_t and c_t forward), but when computing gradients, you stop accumulating after a fixed window and treat dh and dc as zero beyond it.

This is a practical compromise: you lose gradient signal for dependencies longer than the truncation window, but training becomes computationally feasible. The cell state's additive path means even truncated BPTT in an LSTM captures much longer dependencies than the equivalent truncation in a vanilla RNN.

---

## Summary — the full backward recipe per timestep

| Step | Equation |
|---|---|
| 1. Output gate grad | ∂L/∂o_t = dh_t ⊙ tanh(c_t) |
| 1. Cell state total | dc_t_total = dc_t_future + dh_t ⊙ o_t ⊙ (1 − tanh²(c_t)) |
| 2. Forget gate grad | ∂L/∂f_t = dc_t_total ⊙ c_{t-1} |
| 2. Prev cell state grad | ∂L/∂c_{t-1} = dc_t_total ⊙ f_t |
| 2. Input gate grad | ∂L/∂i_t = dc_t_total ⊙ c̃_t |
| 2. Candidate grad | ∂L/∂c̃_t = dc_t_total ⊙ i_t |
| 3. Pre-activations | ∂L/∂z_• = (∂L/∂•) ⊙ activation'(·) for each gate |
| 4. Weight grads | ∂L/∂W_•h = (∂L/∂z_•) · h_{t-1}ᵀ, etc. |
| 5. Prev hidden state grad | ∂L/∂h_{t-1} = Σ over • of W_•hᵀ · (∂L/∂z_•) |

Two values travel to the next step back: dh_{t-1} (four-matrix path, can shrink) and dc_{t-1} (single elementwise multiply by f_t, the gradient highway).
