# Chapter 1 — The Problem LSTM Solves

---

## What you already know

You finished the RNN. You know this equation drives everything:

h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)

And you know what breaks it: **vanishing gradients**.

When you do BPTT and push the gradient back through time, you multiply by W_h and by the derivative of tanh at every single timestep. The derivative of tanh is always between 0 and 1. Multiply enough numbers between 0 and 1 together and you get something so tiny it's basically zero.

After about 8–10 steps back, the gradient is dead. The early timesteps learn nothing.

---

## Why that's actually fatal — a concrete example

Say you're training a language model on this sentence:

> **"The cats that were sitting on the mat are hungry."**

The model is at the word **"are"** and needs to predict **"hungry"**. To do that correctly, it needs to know the subject of the sentence is **"cats"** — which is 7 words back.

```
t=1      t=2    t=3     t=4    t=5      t=6   t=7    t=8    t=9
"The"  "cats" "that" "were" "sitting" "on"  "the"  "mat"  "are"  → predict: "hungry"
```

The signal that **"cats"** is the subject needs to travel from t=2 all the way to t=9. That's 7 multiplication steps through W_h and tanh'. Each step shrinks the gradient. By the time you're back at t=2, the gradient is so small that the weights at that position don't update. The network **never learns** that "cats" controls "are".

This isn't a training trick problem. It's not "use a better learning rate." The architecture itself destroys the learning signal.

---

## The actual problems — precisely stated

### Does h_{t-1} flow forward in an RNN?

Yes. The equation explicitly includes it:

h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)

h_{t-1} is not thrown away. It is transformed and passed forward at every timestep.

---

### Problem 1 — Gradient vanishing during training

In theory, a perfect W_h could carry long-range information forward. But **gradient descent has to find that perfect W_h — and the gradient signal that teaches W_h what to do vanishes before it gets there.**

When you compute ∂L/∂W_h at timestep t=2 in a 9-step sequence, you multiply through 7 Jacobians:

∂h_t/∂h_{t-1} = W_hᵀ · diag(tanh'(z_t))

Each of those terms contains values smaller than 1. Multiplied together 7 times, the gradient at t=2 is essentially zero. So W_h receives no useful update signal for the long-range connection.

The "perfect W_h" exists in theory. But gradient descent is blind to it because the learning signal is dead by the time it reaches early timesteps.

---

### Problem 2 — W_h is static, but memory should be dynamic

W_h is a **single fixed matrix**, shared across every timestep. It applies the same transformation to h_{t-1} regardless of what the current input is.

But what you actually want is context-sensitive behavior:

- At t=5 (word "sitting"): ignore most of what's in memory — it's not relevant right now
- At t=2 (word "cats"): strongly preserve this — it's the subject of the sentence

A fixed W_h cannot do that. It cannot decide "at this particular timestep, keep this particular piece of information alive." It applies the same mixing rule everywhere.

**LSTM's gates are dynamic.** They are recomputed fresh at every timestep from the current input x_t and current hidden state h_{t-1}. So the network can learn to forget at t=5 and preserve at t=2 — based on what it is actually seeing.

---

## What LSTM does differently — the big idea

LSTM introduces a second vector alongside h_t: the **cell state**, written c_t.

- h_t ∈ ℝ^(H×1) — hidden state (same as RNN)
- c_t ∈ ℝ^(H×1) — cell state (new in LSTM)

The key is how c_t is updated. Instead of a full nonlinear rewrite, it is updated **additively**:

c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

Don't worry about what f_t, i_t, and c̃_t are yet — every symbol will be defined precisely in Chapter 3. Focus on the **shape** of this equation.

It says: new cell state = **some fraction of the old cell state** + **some new information**.

That **"+"** is everything. When you differentiate through an addition, the gradient passes straight through — it does not get multiplied by a weight matrix, it does not get squashed by tanh. It flows back through time like water through an open pipe.

This is called the **constant error carousel** — a gradient highway through which the signal can travel many timesteps without vanishing.

---

## The three questions LSTM answers at every timestep

At each timestep t, the LSTM asks:

1. **What should I forget** from what I remembered before? → **Forget gate** f_t
2. **What new thing should I remember** from the current input? → **Input gate** i_t + **candidate** c̃_t
3. **What should I output** right now based on my memory? → **Output gate** o_t

Each gate is a vector of numbers between 0 and 1, computed by a sigmoid. A 0 means "block everything." A 1 means "let everything through."

---

## Summary

| Question | Answer |
|---|---|
| Does h_{t-1} flow forward in an RNN? | **Yes** — through W_h |
| Can the perfect W_h carry long-range info? | **In theory yes, in practice no** — gradient vanishes before W_h can learn it |
| Is W_h flexible enough to selectively remember? | **No** — same matrix every step, no context-sensitivity |
| How does LSTM fix gradient vanishing? | Additive cell state update — gradient passes through "+" without shrinking |
| How does LSTM fix static memory? | Dynamic gates recomputed each step from current input |
