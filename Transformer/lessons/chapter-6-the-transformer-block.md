# Chapter 6 — The Transformer Block

---

## Step 1 — Recap: what we have so far

From Chapter 5c, we have **X ∈ ℝ^(T×d) = ℝ^(9×4)** — every word's embedding-plus-position, stacked as rows. From Chapter 4, we can compute **MultiHead(X) ∈ ℝ^(T×d)** — the same shape, but every row has now "looked around" the sentence and gathered relevant context via attention.

This chapter answers: what else needs to happen to MultiHead(X) before it's ready to be stacked into a deep network of many such blocks?

## Step 2 — Problem 1: attention mixes, but doesn't "think"

Look at what attention actually computes: output_i = Σ_j α(i,j)·v_j — a **weighted average** of other words' Value vectors. That's a powerful way to gather relevant information from across the sentence, but a weighted average is fundamentally a fairly simple operation — it can't, say, combine two features in a complicated new way, or apply a "rule" like "if this pattern is present, produce that pattern instead." Attention's whole job is mixing across positions; it was never built to do complex processing within a single position's content.

Let's make this precise with numbers.

### The precise mathematical fact: a weighted average is always trapped "between" its inputs

Recall attention's output: output_i = Σ_j α(i,j)·v_j, where the weights α(i,j) are all non-negative and sum to 1 (that's what softmax guarantees, from Chapter 2). This specific kind of combination — non-negative weights summing to 1 — is called a **convex combination**, and it has a hard mathematical property: **the result can never fall outside the range spanned by the values being combined.**

Concretely, for just two values a and b, and any valid weights α₁+α₂=1 (α₁,α₂≥0):

α₁·a + α₂·b always lands somewhere between the smaller of {a,b} and the larger of {a,b} — never below the smaller one, never above the larger one.

### A concrete numeric illustration

Suppose, for a toy example, we track just one number (one "slot") representing "how positive" a word's Value vector is:

- v("good") = 5 (strongly positive)
- v("not") = −1 (mildly negative)

If some word attends to both "not" and "good" with weights α(not)=α₁ and α(good)=1−α₁, the output is:

output = α₁×(−1) + (1−α₁)×5 = 5 − 6α₁

As α₁ ranges anywhere from 0 to 1, this output ranges from 5 down to −1 — that's the *entire* possible range, for any choice of attention weights. The output can never be, say, −5.

### Why that matters for "not good"

Semantically, "not good" should mean something close to "bad" — which isn't a blend or midpoint of "good" and "not"; it's more like the opposite of good, a genuinely new value lying outside the [−1, 5] range those two words' Values occupy. No matter how attention tunes its weights — 90% "good," 10% "not," or any other split — a weighted average of 5 and −1 can only ever land between them (the closest it can get to "bad" is landing near −1, "not"'s own value, never past it). **Attention alone, structurally, cannot flip a value to its opposite or amplify it beyond the range of what it's mixing.** That's what "can't apply a rule like 'if this pattern is present, produce that pattern instead'" means — negation is exactly that kind of rule (detect negation + positive-sentiment together → invert the sign), and a convex combination is mathematically incapable of producing it.

### Where the FFN comes in

The FFN, by contrast, is **not** a weighted average — it's W_2·ReLU(W_1·x+b_1)+b_2, involving matrix multiplications and a nonlinearity, which together can represent exactly this kind of rule: e.g., learn (via W_1) a slot that detects "negation signal AND positive signal both present," let ReLU threshold on that detection, and then (via W_2) route that detection into flipping the output strongly negative — something no amount of attention-weight-tuning could ever produce, because attention is fundamentally bounded by convex combination, and the FFN isn't.

**The fix, in short**: after attention gathers context, give each word's resulting vector a chance to be **processed independently** — a place for the model to "think about what it just heard," not just "listen to everyone." This is the job of the **feed-forward network (FFN)**.

## Step 3 — ReLU: a new nonlinearity

The FFN needs a nonlinearity (we'll see exactly why in Step 4). Instead of tanh or sigmoid (used throughout RNN/LSTM), Transformers standardly use **ReLU** ("Rectified Linear Unit"):

**ReLU(z) = max(0, z)**

For a single number z: if z is positive, ReLU leaves it unchanged; if z is zero or negative, ReLU outputs exactly 0. Applied to a vector, this happens **elementwise** — each slot independently.

Example: ReLU([2, −3, 0.5, −1]) = [2, 0, 0.5, 0].

**Why ReLU instead of tanh/sigmoid here?** Recall Chapter 1: tanh's derivative is always strictly between 0 and 1 — squashing gradients a little at every use, which is exactly what caused vanishing gradients in RNNs. ReLU's derivative is either **exactly 0** (for negative inputs) or **exactly 1** (for positive inputs) — never a fractional squashing value. For a network with many stacked layers (as Transformers will have), this makes gradients much less prone to shrinking away to nothing purely from the nonlinearity itself.

## Step 4 — The Feed-Forward Network (FFN), precisely

Applied to a single word's vector x ∈ ℝ^(d×1) (any row of MultiHead(X), after the steps in Step 6-8 below), the FFN is:

**FFN(x) = W_2 · ReLU(W_1·x + b_1) + b_2**

Defining every piece:
- **W_1 ∈ ℝ^(d_ff×d)** — the first layer's weight matrix. **d_ff** is a new symbol: the FFN's **internal, expanded dimension** — bigger than d, giving the network more "room" to combine features before compressing back down. Real models typically use d_ff ≈ 4×d; we'll use **d_ff = 8** (2×d, for hand-computable arithmetic).
- **b_1 ∈ ℝ^(d_ff×1)** — first layer's bias, added after the matrix multiply, same role as biases in RNN/LSTM's equations.
- **ReLU(...)** — applied elementwise to the d_ff-length result.
- **W_2 ∈ ℝ^(d×d_ff)** — the second layer, projecting back down from d_ff to d.
- **b_2 ∈ ℝ^(d×1)** — second layer's bias.

Shape check: W_1·x is (d_ff×d)·(d×1)=(d_ff×1); after ReLU, still (d_ff×1); W_2·(that) is (d×d_ff)·(d_ff×1)=(d×1); plus b_2, still (d×1). **FFN(x) ∈ ℝ^(d×1)** — same shape as the input, so it can slot back into the pipeline.

**Why is the ReLU essential — why not just W_2·(W_1·x+b_1)+b_2, two linear layers with nothing in between?** Expand it out: W_2·(W_1·x+b_1)+b_2 = (W_2·W_1)·x + (W_2·b_1+b_2). Since W_2·W_1 is *just some other matrix* (shape (d×d_ff)·(d_ff×d)=(d×d)) and W_2·b_1+b_2 is *just some other bias vector*, this whole expression is mathematically identical to a **single** linear layer, y = W'·x + b'. **Stacking two linear layers with no nonlinearity between them is no more expressive than one.** The ReLU is what breaks this collapse and gives the network genuinely new expressive power — it's not optional decoration. (This is the same structural gap Step 2 identified for attention — a convex combination alone can't implement a conditional "rule"; here, a linear map alone can't either, for a related reason. The nonlinearity, wherever it appears, is what unlocks rule-like behavior.)

**Applied per-word, independently**: the *same* W_1, b_1, W_2, b_2 are used for every word (shared, like W_Q from Chapter 3), but critically, the FFN **never mixes information across different words** — word 9's FFN output depends only on word 9's own vector, nothing from word 2 or word 5. This is the exact complementary role to attention: **attention mixes across positions** (but only via weighted averaging); **the FFN processes within a position** (but adds real nonlinear depth). Applied to the whole matrix at once — FFN(X) — this uses the same "row t of the output only depends on row t of the input" trick from Chapter 4, giving FFN(X) ∈ ℝ^(T×d), computed for all T words simultaneously.

## Step 5 — Problem 2: stacking many blocks causes gradient trouble

Real Transformers stack **many** of these blocks on top of each other — 6, 12, 24, even 96 layers deep. During training, gradients need to flow backward through every one of these blocks. Recall Chapter 1's vanishing gradient problem for RNNs: repeatedly multiplying by weight matrices and squashing derivatives across many steps caused gradients to shrink to nothing. **The exact same risk reappears here** — with dozens of stacked blocks, gradients flowing backward through that many multiplications can vanish or become unstable, making the network very hard to train.

## Step 6 — The fix: residual (skip) connections

Recall LSTM's fix for its own version of this problem: **c_t = f_t⊙c_{t-1} + i_t⊙c̃_t** — the "+" let gradients pass straight through addition, undiminished by any matrix multiplication or squashing function. Transformers use the **exact same trick**, called a **residual connection**:

Instead of replacing x with Sublayer(x) directly, compute:

**y = x + Sublayer(x)**

where Sublayer(x) could be MultiHeadAttention(x) or FFN(x) — whichever sub-computation we're wrapping.

**Why this helps gradients**: differentiating y = x + Sublayer(x) with respect to x gives ∂y/∂x = I + ∂Sublayer(x)/∂x, where **I** is the identity (the derivative of x with respect to itself). No matter how small or unstable ∂Sublayer(x)/∂x is, the **"+I" guarantees at least a direct, undiminished path** for the gradient to flow straight through — precisely the "constant error carousel" idea from LSTM, now applied across entire attention/FFN sublayers instead of just across timesteps.

("Residual" — the name comes from thinking of Sublayer(x) as only needing to learn the *residual*, i.e., the difference/remainder between x and whatever the ideal output should be, rather than reconstructing the whole thing including x itself from scratch. If the ideal transformation is close to "leave x mostly unchanged," Sublayer can learn something close to zero — often much easier than learning to faithfully reproduce x through a complicated nonlinear function.)

## Step 7 — Problem 3: values can drift in scale across many blocks

With many blocks, each doing additions (residual connections) and multiplications (attention, FFN), the raw *scale* of the numbers flowing through the network can drift — some values growing very large, others shrinking, block after block — making training unstable (similar in spirit to why Chapter 3 needed the √d_k scaling factor to keep attention scores well-behaved).

## Step 8 — The fix: LayerNorm

**LayerNorm** re-centers and re-scales a word's vector, keeping its numbers in a consistent, well-behaved range, without changing the essential pattern of relative values.

For a single word's vector x ∈ ℝ^(d×1) (the d numbers describing that one word):

**μ = (1/d) · Σ_j x[j]** — the **mean**: the average of that word's own d numbers (μ is the conventional letter for "mean").

**σ² = (1/d) · Σ_j (x[j] − μ)²** — the **variance**: how spread out that word's own d numbers are around their mean (σ, "sigma," is the conventional letter for standard deviation; σ² is variance).

**LayerNorm(x)[j] = (x[j] − μ) / √(σ² + ε)** — for each slot j: subtract the mean (recentering to 0), then divide by the standard deviation √σ² (rescaling to a consistent spread of ~1). **ε** (epsilon) is a tiny constant (like 0.00001) added purely to avoid dividing by zero if σ² happens to be extremely small — not a meaningful part of the math, just numerical safety.

**Crucial detail — this normalizes *across the d slots of one word*, not across different words or different sentences.** That's precisely why it's called **Layer** Norm (normalizing within one "layer" of features belonging to a single example) as opposed to Batch Norm (a different technique, common in other architectures, that normalizes across multiple examples instead — not used here).

**Concrete example**: take x = [10, 2, 6, 2] (d=4).
μ = (10+2+6+2)/4 = 5
Deviations: 5, −3, 1, −3. Squared: 25, 9, 1, 9. Sum=44. σ² = 44/4 = 11. σ = √11 ≈ 3.3166.
LayerNorm(x) = [5/3.3166, −3/3.3166, 1/3.3166, −3/3.3166] ≈ **[1.5076, −0.9045, 0.3015, −0.9045]**

Check: mean of the output ≈ (1.5076−0.9045+0.3015−0.9045)/4 ≈ 0 ✓ — no matter the original scale of the input numbers, LayerNorm's output is always centered at 0 with a standard spread of ~1.

In practice, this is followed by a small learned adjustment: **γ ⊙ LayerNorm(x) + β**, where **γ, β ∈ ℝ^(d×1)** are learned parameters (γ = "gamma," a per-slot scale; β = "beta," a per-slot shift) — letting the network learn to undo or adjust the normalization if that turns out to help, rather than being rigidly forced to exactly mean-0/spread-1 output.

## Step 9 — Assembling the full block: "Add & Norm"

The combination "add the residual, then LayerNorm" is standard enough to have its own name: **Add & Norm**. A full Transformer block applies this twice — once around attention, once around the FFN:

1. **X' = LayerNorm( X + MultiHeadAttention(X) )** — Add & Norm around attention
2. **X'' = LayerNorm( X' + FFN(X') )** — Add & Norm around the FFN

**X'' ∈ ℝ^(T×d)** — same shape as the original X, ready to either feed into another identical block, or move on to whatever comes next (Chapter 7).

## Step 10 — A full worked example, chained through both Add & Norm steps

Take word 9 ("are"), reusing **x_9 = [1, 0, 1, 0]** from Chapter 3. Say (illustratively — MultiHeadAttention's actual output depends on the whole sentence and the specific learned weights, which we're not re-deriving here) **MultiHeadAttention(x_9) = [0.7, 1.3, −0.5, 0.2]**.

**Add & Norm #1:**
Sum: x_9 + MultiHeadAttention(x_9) = [1.7, 1.3, 0.5, 0.2]
μ = 3.7/4 = 0.925; deviations [0.775, 0.375, −0.425, −0.725]; σ² = 1.4475/4 = 0.361875; σ ≈ 0.6016
**x'_9 = LayerNorm(...) ≈ [1.2885, 0.6234, −0.7067, −1.2052]**

**FFN(x'_9)** — using simple 0/1/−1-valued rows for W_1 (8×4) and W_2 (4×8), and bias b_1 with a +1 in its last slot only:

z = W_1·x'_9 + b_1 = [1.2885, 0.6234, −0.7067, −1.2052, 1.9119, 0.5818, −0.5818, 2.1636]
ReLU(z) = [1.2885, 0.6234, 0, 0, 1.9119, 0.5818, 0, 2.1636] (negative entries zeroed)
**FFN(x'_9) = W_2·ReLU(z) = [3.2004, 1.2052, 0, 2.1636]**

**Add & Norm #2:**
Sum: x'_9 + FFN(x'_9) = [4.4889, 1.8286, −0.7067, 0.9584]
μ ≈ 1.6423; deviations [2.8466, 0.1863, −2.3490, −0.6839]; σ² ≈ 3.5309; σ ≈ 1.8791
**X''(word 9) ≈ [1.5148, 0.0991, −1.2500, −0.3639]**

That final 4-number vector is word 9's representation after one complete Transformer block — having attended to context, been processed nonlinearly, and kept numerically well-behaved throughout.

## Step 11 — Stacking multiple blocks

Real Transformers stack many of these blocks — call the total count **L** (a new symbol: number of layers/blocks; e.g. L=6 in the original paper, L=96 in large modern models). Block 2 takes block 1's output X'' as its own input X, runs the identical two-step Add & Norm pipeline, and so on through all L blocks.

**Crucially: each block has its own, separate set of weights** — W_Q, W_K, W_V, W_O, W_1, b_1, W_2, b_2, γ, β are all **independently learned per block**, not shared across blocks (contrast with how, *within* one block, W_Q etc. *are* shared across all T words). Each block can therefore learn to specialize in progressively different kinds of processing as depth increases.

## Step 12 — Aside: Post-LN vs. Pre-LN

The version taught here — normalize *after* adding the residual (LayerNorm(X + Sublayer(X))) — is called **Post-LN**, and is what the original Transformer paper used. Many modern models (GPT-2 onward) instead use **Pre-LN** — normalize *before* the sublayer runs: X + Sublayer(LayerNorm(X)). Pre-LN tends to train more stably for very deep networks (many blocks). Both are used in practice; we're teaching Post-LN as the standard, foundational version.

---

## Summary

| Symbol | Shape | Plain meaning |
|---|---|---|
| ReLU(z) = max(0,z) | elementwise | zeroes out negatives, leaves positives unchanged |
| d_ff | scalar (=8 here) | FFN's internal expanded dimension, > d |
| W_1, b_1 | ℝ^(d_ff×d), ℝ^(d_ff×1) | FFN's first layer — expand |
| W_2, b_2 | ℝ^(d×d_ff), ℝ^(d×1) | FFN's second layer — contract back to d |
| FFN(x) = W_2·ReLU(W_1x+b_1)+b_2 | ℝ^(d×1) | per-word nonlinear processing; needs ReLU or it collapses to one linear layer |
| Residual: y = x + Sublayer(x) | same as x | gives gradients a direct "+I" path through many stacked blocks |
| μ, σ² | scalars, per word | mean and variance of one word's own d numbers |
| LayerNorm(x) | ℝ^(d×1) | recenters to mean 0, rescales to spread ~1, across a word's own slots |
| γ, β | ℝ^(d×1) each | learned scale/shift applied after normalizing |
| Add & Norm | — | the pattern LayerNorm(x + Sublayer(x)), applied around both attention and FFN |
| L | scalar | number of stacked Transformer blocks; each block has its own separate weights |

---

# Follow-up Q&A

## Q1 — "You mean we send MultiHead(X), which is T×d, into the FFN? Each row gets sent to FFN?"

**Yes, the FFN is applied row-by-row, one word at a time, using the same shared weights for every row** — that part is correct.

**One small correction on exactly what feeds in**: it's not literally MultiHead(X) that goes straight into the FFN. Look at the pipeline from Step 9:

1. **X' = LayerNorm( X + MultiHeadAttention(X) )** — the Add & Norm step happens *first*
2. **X'' = LayerNorm( X' + FFN(X') )** — the FFN operates on **X'** (the *post*-Add&Norm result), not on MultiHeadAttention(X) directly

So the actual input to FFN is **X' ∈ ℝ^(T×d)** — the residual-connected, normalized version of the attention output, not the raw attention output itself.

But the core idea — **each row sent to FFN independently** — is exactly right. For X' ∈ ℝ^(T×d), FFN gets applied to **each row independently**: row t (word t's own d-length vector) passes through FFN(row_t) = W_2·ReLU(W_1·row_t + b_1) + b_2, using the *identical* W_1, b_1, W_2, b_2 for every one of the T rows — no row ever sees or mixes with another row's content inside the FFN (that mixing-across-words job belongs entirely to attention, not the FFN).

Applied to the whole matrix at once — **FFN(X') ∈ ℝ^(T×d)** — this is exactly the same "row t of the output only depends on row t of the input" trick from Chapter 4 (where Q = X·W_Q^T worked the same way): one matrix operation computes all T rows' FFN outputs simultaneously, with each row's result depending only on its own row of X'.
