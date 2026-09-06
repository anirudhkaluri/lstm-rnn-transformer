# Chapter 3 — Where Q, K, V Come From, and the Full Formula

---

## Step 1 — Recap: the two loose threads from Chapter 2

At the end of Chapter 2, we had a working mechanism — score, softmax, weighted sum — but we *cheated* in one way: we just **handed you** q_9 = [1.0, 0.0], k_2 = [0.9, 0.1], v_2 = [5.0, 0.0], etc., as if they fell from the sky.

Two things left to explain:
1. **Where do q_t, k_t, v_t actually come from**, given that all we *really* have is x_t (the row of X)?
2. **The √d_k scaling factor** — the real formula divides scores by √d_k before softmax, and we skipped that.

This chapter answers both, then assembles everything into **one matrix formula** that computes self-attention for all 9 words at once.

---

## Step 2 — The idea: q_t, k_t, v_t are "extracted" from x_t

Recall **x_t ∈ ℝ^(d×1) = ℝ^(4×1)** — the row of X for word t: 4 numbers describing that word's raw, out-of-context meaning.

Here's the intuition: x_t contains *all sorts* of information about word t, mashed together into 4 numbers. But:
- The **Query** (q_t) only needs the part of that information relevant to "what am I looking for?"
- The **Key** (k_t) only needs the part relevant to "how should I advertise myself for matching?"
- The **Value** (v_t) only needs the part relevant to "what's the actual content I'll hand over?"

These are three different "lenses" looking at the same underlying x_t, each pulling out a different, smaller summary. **A "lens" here is just a matrix** — multiplying a vector by a matrix is exactly the operation that takes a list of numbers and produces a new (possibly different-length) list of numbers, by taking weighted combinations of the original numbers. You already used this idea in RNN/LSTM: h_t = tanh(W_x · x_t + ...) — W_x is exactly this kind of "lens," turning x_t into something the shape of h_t.

---

## Step 3 — W_Q: the "lens" that produces the Query

Define **W_Q** — a matrix, the **W** standing for **"weight"** (same meaning as W_x, W_h from RNN/LSTM — a matrix of learnable numbers), and the subscript **Q** meaning **"this is the weight matrix whose job is to produce Queries."**

The formula: **q_t = W_Q · x_t**

Now, what **shape** must W_Q be? We know:
- x_t ∈ ℝ^(4×1) — 4 numbers in, since d=4
- q_t ∈ ℝ^(2×1) — 2 numbers out, since d_k=2 (recall: d_k is named after Key, but it's also Query's size, from Chapter 2)

For matrix-vector multiplication (W_Q · x_t) to take a 4×1 vector and produce a 2×1 vector, W_Q must be **2 rows × 4 columns** — written **W_Q ∈ ℝ^(d_k × d) = ℝ^(2×4)**. (Rule: (rows × cols)·(cols × 1) → (rows × 1). The "cols" of W_Q must match the length of x_t (=4, the d), and the "rows" of W_Q determines the length of the output (=2, the d_k).)

### A concrete example — and a callback to Chapter 2

Let's pick numbers and actually *derive* q_9 = [1.0, 0.0] — the exact value Chapter 2 handed you — so you can see where it really comes from.

Say **x_9 ("are") = [1, 0, 1, 0]** (a toy 4-number embedding for "are" — just some numbers, in reality learned during training).

Let **W_Q** be the 2×4 matrix with:
- Row 1 = [1, 0, 0, 0]
- Row 2 = [0, 1, 0, -1]

To compute q_9 = W_Q · x_9, take the **dot product of each row of W_Q with x_9** (this is the same dot product from Chapter 2 — multiply matching positions, add them up):

- Row 1 · x_9 = (1)(1) + (0)(0) + (0)(1) + (0)(0) = **1**
- Row 2 · x_9 = (0)(1) + (1)(0) + (0)(1) + (-1)(0) = **0**

**q_9 = [1, 0]** — exactly the value Chapter 2 used! This is the mechanism that produces it: a learned 2×4 matrix, multiplied against "are"'s 4-number embedding.

---

## Step 4 — W_K and W_V: the same idea, for Key and Value

By identical reasoning:

- **W_K ∈ ℝ^(d_k × d) = ℝ^(2×4)** — "the weight matrix whose job is to produce Keys." Formula: **k_t = W_K · x_t**. Same shape as W_Q (since Key and Query share the size d_k), but a **different matrix** — it gets to learn a completely different "lens."
- **W_V ∈ ℝ^(d_v × d) = ℝ^(2×4)** — "the weight matrix whose job is to produce Values." Formula: **v_t = W_V · x_t**. Its row-count is d_v (=2 here) instead of d_k — in general d_v could differ from d_k, so W_V's shape could differ from W_Q/W_K's, even though in our toy example both happen to be 2×4.

So every word t gets **three** numbers-from-numbers extractions, all starting from the same x_t:

q_t = W_Q · x_t,  k_t = W_K · x_t,  v_t = W_V · x_t

---

## Step 5 — These weights are learned, and shared across all words

Crucially: **W_Q, W_K, W_V are each a single matrix, used for every word t** (just like W_x and W_h in your RNN/LSTM were single matrices reused at every timestep). What differs from word to word is **x_t** (each word's own embedding) — not the matrices.

These matrices start out as random numbers and get adjusted via gradient descent during training — exactly like every other weight matrix you've encountered. Training is what teaches W_Q to extract "things I need to know about" and W_K to extract "things I can offer for matching," in whatever way actually helps the model predict the next word correctly.

---

## Step 6 — The scaling factor: why divide by √d_k?

Here's the piece we skipped in Chapter 2. The real formula isn't just score(i,j) = q_i · k_j — it's:

**scaled_score(i,j) = (q_i · k_j) / √d_k**

where **√d_k** means "the square root of d_k" — and remember, **d_k is the length of the Query/Key vectors** (=2 in our toy example, but typically 64 or more in real models).

### Why does this matter?

Think about what a dot product does: q_i · k_j = (sum of d_k separate products, one per position). **The more positions you're summing over, the bigger that sum tends to get** — even if each individual product is just a small, "typical-sized" number.

Concretely: suppose every entry of q and k is roughly around 1 in size (could be positive or negative). Each individual product (one entry of q times the matching entry of k) is then also roughly around 1 in size. If d_k = 2, you're adding **2** such terms — the sum is roughly around 1-2 in size. But if d_k = 64, you're adding **64** such terms — the sum could be roughly around 8 in size (it grows like √d_k for "typical" random numbers, since positive and negative terms partly cancel — but it still grows).

**Why is a big score a problem?** Recall softmax: softmax(z)_j = exp(z_j) / Σ exp(z_j'). If one score is, say, 8, and another is 6, then exp(8) ≈ 2981 and exp(6) ≈ 403 — softmax gives the first almost ALL the weight (≈88%) and crushes the second to almost nothing, **even though 8 vs 6 isn't actually that different in relative terms**. The softmax becomes too "spiky" / saturated, attention collapses to picking just one token almost exclusively, and — more importantly for training — the gradient through a saturated softmax becomes tiny (it's flat almost everywhere except right at the spike), so learning slows down.

**The fix**: divide every score by √d_k *before* the softmax. This rescales scores back down to a "typical size around 1," regardless of how large d_k is — keeping softmax in its well-behaved, gradient-friendly range. With d_k = 2, √d_k = √2 ≈ 1.414 — a small correction. With d_k = 64, √d_k = 8 — a much bigger correction, exactly compensating for the bigger sums.

So the formula becomes:

**α(i,j) = softmax_j( (q_i · k_j) / √d_k )**

— same α(i,j) from Chapter 2 ("the fraction of word i's attention given to word j"), just computed from the *scaled* score instead of the raw one.

---

## Step 7 — Doing this for all 9 words at once: stack into matrices

In Chapter 2, we computed q_9, and compared it against k_2, k_5, k_8 individually — one word at a time. But **every word** needs to do this. Rather than writing 9 separate sets of equations, we stack things into matrices — exactly the same trick that turned individual x_t vectors into the big matrix X.

- **Q ∈ ℝ^(T×d_k) = ℝ^(9×2)** — the matrix you get by stacking q_1, q_2, ..., q_9 as rows. Row t of Q is word t's Query.
- **K ∈ ℝ^(T×d_k) = ℝ^(9×2)** — stack k_1, ..., k_9 as rows.
- **V ∈ ℝ^(T×d_v) = ℝ^(9×2)** — stack v_1, ..., v_9 as rows.

These are computed as: **Q = X · W_Qᵀ**, **K = X · W_Kᵀ**, **V = X · W_Vᵀ** (the "ᵀ" here means transpose, needed to make the matrix shapes line up for matrix-matrix multiplication — X is 9×4, W_Q is 2×4, so W_Q transposed is 4×2, and (9×4)·(4×2) = 9×2 ✓). See the **Follow-up Q&A** section at the end for exactly *why* this stacking trick works, derived directly from the per-word equation.

---

## Step 8 — A quick refresher: matrix-matrix multiplication

You've used matrix-**vector** multiplication (W_x · x_t — a matrix times a single column). Now we need matrix-**matrix** multiplication (one matrix times another matrix). The rule is the same idea, just repeated for every column of the second matrix:

**If A is (m×n) and B is (n×p), then A·B is (m×p), where entry (i,j) of A·B = (row i of A) · (column j of B)** — the same dot product as before, just computed once for every (row, column) pair.

We'll use exactly this in the next step.

---

## Step 9 — QK^T: all scores, computed in one shot

⚠️ **Notation alert, before we go further**: We've been using **T = 9** to mean "the number of words in our sentence" (Chapter 1). We're about to write **K^T** — here, the superscript **T means "transpose"** (flip a matrix so its rows become columns and vice versa). **These are two completely unrelated uses of the letter T that happen to collide** — one is a count (sequence length), the other is an operation (transpose). Context tells you which is meant: a superscript T right after a matrix name = transpose; a standalone T = sequence length.

Now: **K ∈ ℝ^(T×d_k) = ℝ^(9×2)**. Its transpose, **K^T ∈ ℝ^(d_k×T) = ℝ^(2×9)**, is the same numbers with rows and columns swapped — so K^T has 2 rows and 9 columns. Column j of K^T is exactly k_j (word j's Key), just written vertically instead of horizontally.

Now compute **Q · K^T**: Q is (9×2), K^T is (2×9), so Q·K^T is **(9×9)**. Using the matrix-multiplication rule from Step 8:

**entry (i,j) of Q·K^T = (row i of Q) · (column j of K^T) = q_i · k_j = score(i,j)**

**Q·K^T is a 9×9 matrix containing EVERY score(i,j) for every pair of words (i,j) — all 81 scores — computed via a single matrix multiplication.** Row i of this matrix is exactly "word i's scores against every other word" — the same numbers we computed by hand one at a time in Chapter 2, now produced all at once.

---

## Step 10 — The full formula

Putting every piece together:

**Attention(Q, K, V) = softmax( Q·K^T / √d_k ) · V**

Reading this left to right, as a sequence of operations on whole matrices:

1. **Q·K^T** → a (9×9) matrix of raw scores (every word vs. every word)
2. **÷ √d_k** → divide every entry by √d_k, to keep softmax well-behaved (Step 6)
3. **softmax(...)** → applied **row by row** — each row gets turned into a "recipe" that sums to 1 (each row i becomes [α(i,1), α(i,2), ..., α(i,9)])
4. **· V** → multiply the resulting (9×9) "recipe matrix" by V (9×2). By the matrix-multiplication rule (Step 8): entry (i, *) of the result = (row i of the recipe matrix) · (columns of V) = Σ_j α(i,j)·v_j = **output_i** — exactly the Chapter 2 formula!

The final result is a **(9×2) matrix** — row i is output_i, word i's new context-aware representation. **One formula, all 9 words processed simultaneously — this is why Transformers parallelize so well** (recall Chapter 1's complaint about LSTM being forced sequential).

---

## Step 11 — A small concrete walk-through of the matrix formula

To see this with actual numbers, let's use a fresh tiny example: **T = 3** words (call them A, B, C), **d_k = 2**, **d_v = 2**.

**Q (3×2)** — rows are q_A, q_B, q_C:
```
Q = [ 1  0 ]   ← q_A
    [ 0  1 ]   ← q_B
    [ 1  1 ]   ← q_C
```

**K (3×2)** — rows are k_A, k_B, k_C (in this example, happens to equal Q):
```
K = [ 1  0 ]   ← k_A
    [ 0  1 ]   ← k_B
    [ 1  1 ]   ← k_C
```

### Compute Q·K^T (3×3) — entry (i,j) = q_i · k_j

- Row A: q_A·k_A = (1)(1)+(0)(0) = 1; q_A·k_B = (1)(0)+(0)(1) = 0; q_A·k_C = (1)(1)+(0)(1) = 1 → **[1, 0, 1]**
- Row B: q_B·k_A = 0; q_B·k_B = 1; q_B·k_C = 1 → **[0, 1, 1]**
- Row C: q_C·k_A = 1; q_C·k_B = 1; q_C·k_C = (1)(1)+(1)(1) = 2 → **[1, 1, 2]**

### Scale by 1/√d_k = 1/√2 ≈ 0.7071

- Row A: [0.7071, 0, 0.7071]
- Row B: [0, 0.7071, 0.7071]
- Row C: [0.7071, 0.7071, 1.4142]

### Softmax each row (turn into a "recipe" summing to 1)

Row A: exp(0.7071)=2.0281, exp(0)=1.0000, exp(0.7071)=2.0281 → sum=5.0562
→ α_A = [0.4011, 0.1978, 0.4011]

Row B (same numbers, different order): → α_B = [0.1978, 0.4011, 0.4011]

Row C: exp(0.7071)=2.0281, exp(0.7071)=2.0281, exp(1.4142)=4.1133 → sum=8.1695
→ α_C = [0.2482, 0.2482, 0.5036]

### V (3×2) — rows are v_A, v_B, v_C:
```
V = [ 1  0 ]   ← v_A
    [ 0  1 ]   ← v_B
    [ 2  2 ]   ← v_C
```

### Multiply: output = (α matrix) · V

output_A = 0.4011·[1,0] + 0.1978·[0,1] + 0.4011·[2,2]
= [0.4011, 0] + [0, 0.1978] + [0.8022, 0.8022] = **[1.2033, 1.0000]**

output_B = 0.1978·[1,0] + 0.4011·[0,1] + 0.4011·[2,2]
= [0.1978, 0] + [0, 0.4011] + [0.8022, 0.8022] = **[1.0000, 1.2033]**

output_C = 0.2482·[1,0] + 0.2482·[0,1] + 0.5036·[2,2]
= [0.2482, 0] + [0, 0.2482] + [1.0072, 1.0072] = **[1.2554, 1.2554]**

**Result — the output matrix (3×2)**:
```
[ 1.2033  1.0000 ]   ← output_A
[ 1.0000  1.2033 ]   ← output_B
[ 1.2554  1.2554 ]   ← output_C
```

Each row is a "blended" version of A, B, C's Values, with the blend determined entirely by how similar each word's Query was to each word's Key — computed for all 3 words in one pass.

---

## Step 12 — Summary

| Symbol | Shape | Plain meaning |
|---|---|---|
| x_t | ℝ^(d×1) = ℝ^(4×1) | word t's raw embedding (recap from Ch.1) |
| W_Q | ℝ^(d_k×d) = ℝ^(2×4) | learned "lens" that extracts word t's Query from x_t |
| W_K | ℝ^(d_k×d) = ℝ^(2×4) | learned "lens" that extracts word t's Key from x_t |
| W_V | ℝ^(d_v×d) = ℝ^(2×4) | learned "lens" that extracts word t's Value from x_t |
| q_t = W_Q·x_t | ℝ^(d_k×1) | word t's Query (Ch.2) |
| k_t = W_K·x_t | ℝ^(d_k×1) | word t's Key (Ch.2) |
| v_t = W_V·x_t | ℝ^(d_v×1) | word t's Value (Ch.2) |
| Q, K, V | ℝ^(T×d_k), ℝ^(T×d_k), ℝ^(T×d_v) | all 9 words' q/k/v vectors, stacked as rows |
| K^T | ℝ^(d_k×T) | K with rows/columns swapped ("T" here = transpose, NOT seq length!) |
| Q·K^T | ℝ^(T×T) | every score(i,j), all pairs, in one matrix |
| √d_k | scalar | normalizes score size so softmax doesn't saturate as d_k grows |
| Attention(Q,K,V) = softmax(QK^T/√d_k)·V | ℝ^(T×d_v) | every word's output, computed simultaneously |

---

# Follow-up Q&A

## Q1 — "So Row i of Attention(Q,K,V) is a row of size d_v, which is a context-aware representation of word i. Am I right?"

Yes, exactly. Row i of the output is a vector of **d_v** numbers, and it's "context-aware" in a precise sense: it's a **weighted blend of every word's Value vector**, where the weights (the α(i,j)'s) were determined by how well word i's Query matched each word's Key.

Unlike row i of X (which is just "word i's raw embedding, the same numbers regardless of what sentence it appears in"), row i of the output **knows about the whole sentence** — it has "looked around" and mixed in information from whichever words were most relevant to word i. E.g. output_9 was dominated by v_2 ("cats"), because "are"'s Query matched "cats"'s Key best.

## Q2 — "What does V[i,j] represent? I want to understand the significance of multiplying a row of softmax with a column of V."

Row i of the softmax matrix is **[α(i,1), α(i,2), ..., α(i,T)]** — word i's attention weights over every word.

Column j of V is **[V[1,j], V[2,j], ..., V[T,j]]** — the j-th content slot of **every** word's Value vector.

Their dot product: α(i,1)·V[1,j] + α(i,2)·V[2,j] + ... + α(i,T)·V[T,j]

In plain English: **"take the j-th content slot from every word, and mix them together according to how much word i attends to each word."**

Concretely: word 9 ("are") had weights [0.54, 0.22, 0.24] for cats/sitting/mat. Column 1 of V = [5.0, 0.0, 1.0] (the first content slot of each). Dot product = 0.54×5.0 + 0.22×0.0 + 0.24×1.0 = 2.94 — "are"'s first content slot after mixing, dominated by cats's 5.0 because "are" pays 54% attention to cats.

So every entry Attention(Q,K,V)[i,j] answers: **"what number does word i end up with in its j-th content slot, after borrowing from all other words proportionally to how much it attends to them?"** The full output row i = all d_v such numbers assembled = word i's complete context-aware representation.

## Q3 — "So basically the context-aware representation of word i in the j-th slot is based on the values of all other words in the j-th slot?"

Exactly. The j-th slot of word i's output = a weighted average of the j-th slot of **every** word's Value, weighted by how much word i attends to each word.

Every word contributes its own j-th slot value, and word i decides how much to take from each one via its attention weights. Words with high α(i, ·) contribute a lot to word i's j-th slot; words with low α(i, ·) contribute almost nothing.

So the j-th slot of the output isn't "word i's own j-th value" — it's a **blend of the entire sentence's j-th values**, filtered through word i's perspective. That's exactly what "context-aware" means — the same j-th slot would be a completely different number for word i in a different sentence, because the other words (and therefore what gets blended in) would be different.

## Q4 — "q_t = W_Q·x_t — this represents one word, right, where W_Q is (d_k×d) so it can multiply x_t (d×1)? Now how are the computations done for all words at the same time?"

Yes — **q_t = W_Q·x_t** is a **per-word** equation, computed once for each individual word t:
- x_t ∈ ℝ^(d×1) — word t's embedding, a column of d numbers
- W_Q ∈ ℝ^(d_k×d) — d_k rows (output size), d columns (input size, must match x_t's length)
- q_t = W_Q·x_t ∈ ℝ^(d_k×1) — shape check: (d_k×d)·(d×1) = (d_k×1) ✓

Here's how it becomes "all words at once":

**Step A — Transpose both sides.** q_t^T = (W_Q·x_t)^T = x_t^T·W_Q^T (using the rule (A·B)^T = B^T·A^T).

**Step B — Recognize x_t^T and q_t^T as rows.** x_t^T (x_t written as a row instead of a column) is exactly **row t of X**, by definition of how X was built (stacking every word's embedding as a row). Likewise, if Q is built by stacking every word's Query as a row, then q_t^T = **row t of Q**. So the flipped equation says:

**(row t of Q) = (row t of X) · W_Q^T**

**Step C — Recall what matrix multiplication already means.** For any matrices A·B, the definition is (A·B)[row t, col j] = Σ_s A[row t, s]×B[s, col j] — this **only uses row t of A**, never any other row. So **(row t of A·B) = (row t of A)·B**, for every row t, automatically, just from the definition of matrix multiplication — no extra rule needed.

**Step D — Substitute A=X, B=W_Q^T.** (row t of X·W_Q^T) = (row t of X)·W_Q^T = x_t^T·W_Q^T = q_t^T = (row t of Q). This holds for every t=1,...,T simultaneously, because matrix multiplication computes every row this way in one operation. So:

**Q = X · W_Q^T**, shape (T×d)·(d×d_k) = (T×d_k) ✓

**The key insight**: the per-word equation and the whole-sentence equation are not two different computations — they're the *same* computation, written two ways. W_Q never changes between words (shared, like W_x in RNN/LSTM); only x_t (which row of X) changes. And because row t of Q depends *only* on row t of X (never on any other row), there's no sequential dependency at all — a GPU can compute every row in parallel. This is the direct payoff of Chapter 1's motivation: no c_t-needs-c_{t-1} chain here.

Same logic applies identically to K = X·W_K^T and V = X·W_V^T.
