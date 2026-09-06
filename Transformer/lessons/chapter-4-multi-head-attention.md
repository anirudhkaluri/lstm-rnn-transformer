# Chapter 4 — Multi-Head Attention

---

## Step 1 — Why isn't one head enough?

Chapter 3 gave us one attention computation per sentence — one W_Q, one W_K, one W_V, learning to track one type of relationship at a time. Real language has many relationships happening simultaneously (subject-verb agreement, adjective-noun pairing, positional adjacency...). One lens can't specialize in all of them at once.

**Fix**: run several independent attention computations in parallel, each with its own weights, each free to specialize. Each one is called a **head**.

---

## Step 2 — Defining h and m

**h** = the **number of heads** — a plain count, chosen by whoever designs the network (like choosing how many layers to use). It's a design decision, not derived from a formula. Common real-world choices are 8, 12, or 16; we'll use **h = 2** for hand-computable examples.

**m** = the **index of a specific head**, ranging from 1 to h. So "head m" means "the m-th of our h parallel attention computations." When h=2, m takes the values 1 or 2.

---

## Step 3 — What size should each head's Query/Key/Value be? (deriving d_k = d/h)

Each head needs its own Query size and Value size. Name them:
- **d_k** — the Query/Key size **used by each individual head**. (The **k** stands for **Key** — Query is forced to match Key's size so they can be dot-producted, so one name, d_k, covers both — established in Ch.2.)
- **d_v** — the Value size **used by each individual head**. (**v** stands for **Value**.)

**Reframing from Chapter 3**: there, with only one head, d_k was just "whatever size we felt like choosing." Now that we have h heads running side by side, d_k needs to be chosen with the *total* cost of all h heads combined in mind.

### Deriving d_k = d/h from a parameter-budget argument

Each head m needs its own weight matrix **W_Q^(m)**, compressing a word's full embedding (length d) into that head's Query (length d_k). For any matrix, **the number of individual numbers inside it = rows × columns**. W_Q^(m) has shape (d_k rows × d columns), so it contains **d_k × d** numbers.

Since there are **h** heads, each with its own such matrix, the **total** number of Query-related parameters, summed across all heads, is:

**h × (d_k × d)**

**The design choice**: if every head kept the *same* d_k a single head would use (i.e., d_k = d, no shrinking), h heads would cost **h times more** parameters and compute than one head — purely from adding heads.

To avoid this, the standard convention shrinks each head's size in proportion to how many heads there are:

**d_k = d / h**, and by the same reasoning, **d_v = d / h**

Check: total Query parameters across all h heads = h × (d/h) × d = **d²** — identical to a single full-size head, no matter how many heads you use. The h cancels out. This is provable algebra, not a guess.

(Whether using multiple smaller heads actually *performs better* than one big head is a separate, **empirical** claim — confirmed by ablation experiments in the original Transformer paper and every major model since, not something algebra alone proves.)

---

## Step 4 — The actual equation: computing Q^(m) for ALL words at once, directly from X

You have **X ∈ ℝ^(T×d)** — every word's embedding, stacked as rows, all at once. You want **Q^(m)** — every word's head-m Query, also stacked as rows, all at once, computed in **one single equation**:

**Q^(m) = X · (W_Q^(m))^T**

Deriving the required shape of W_Q^(m), starting from what we know and what we want:

- **X is known**: (T × d)
- **We want Q^(m) to be**: (T × d_k) — **T rows**, because Q^(m) still holds one Query per word, and there are T words. **d_k columns**, the per-head Query size from Step 3.

For X · (something) to take a (T×d) matrix and produce a (T×d_k) result, that "something" must be **(d × d_k)** — rule: (T×d)·(d×d_k) = (T×d_k), inner dimensions (d, d) must match.

So: **(W_Q^(m))^T ∈ ℝ^(d × d_k)**, meaning, undoing the transpose: **W_Q^(m) ∈ ℝ^(d_k × d)**.

**What do the rows/columns of W_Q^(m) mean, in terms of this equation?** Look at (W_Q^(m))^T — it has d rows and d_k columns. Each **column** acts like a "detector": take one word's full row from X (its complete d-length embedding), dot it against that column (also d entries) — this uses **all d numbers** of the word's embedding, producing **one output number**. There are **d_k** such detectors, so applying all of them to one word gives d_k output numbers — that word's full head-m Query. Applying this to **every row of X simultaneously** — exactly what X · (W_Q^(m))^T does in one shot — produces the entire Q^(m) matrix for all T words at once.

(Row j of W_Q^(m) = column j of (W_Q^(m))^T = the j-th detector, just written horizontally instead of vertically.)

**Aside, connecting back to Chapter 3**: isolating just row t of this equation recovers exactly Chapter 3's per-word formula, q_t^(m) = W_Q^(m) · x_t — that was always just "one row's worth" of the full equation above.

---

## Step 5 — K^(m), by identical reasoning

**K^(m) = X · (W_K^(m))^T**

X is (T×d), we want K^(m) to be (T×d_k). So (W_K^(m))^T must be (d×d_k), meaning **W_K^(m) ∈ ℝ^(d_k × d)**. Same interpretation: each column of (W_K^(m))^T is a detector reading all d numbers of a word's embedding, producing one Key slot; d_k such detectors give the full Key per word.

---

## Step 6 — V^(m), by identical reasoning

**V^(m) = X · (W_V^(m))^T**

X is (T×d), we want V^(m) to be (T×d_v). So (W_V^(m))^T must be (d×d_v), meaning **W_V^(m) ∈ ℝ^(d_v × d)**.

---

## Step 7 — head_m: from Q^(m), K^(m), V^(m) to one head's output

**head_m = softmax( Q^(m) · (K^(m))^T / √d_k ) · V^(m)**

Tracking shapes:
- Q^(m): (T × d_k)
- (K^(m))^T: (d_k × T)
- Q^(m)·(K^(m))^T: (T×d_k)·(d_k×T) = **(T × T)** — one score per pair of words, specific to head m
- ÷ √d_k, then softmax row-by-row: shape unchanged, **(T × T)**
- multiply by V^(m) (T×d_v): (T×T)·(T×d_v) = **(T × d_v)**

**head_m ∈ ℝ^(T × d_v)** — T rows (one per word), d_v columns (head m's Value size). True for every head, m = 1 to h.

---

## Step 8 — Concat and W_O

**Concat(head_1, ..., head_h) ∈ ℝ^(T × h·d_v)** — glue all h heads' outputs side by side along columns; T (word count) unchanged, columns add up across heads.

Substituting d_v = d/h: h · d_v = h · (d/h) = **d**. So Concat(...) ∈ ℝ^(T × d) — back to exactly the original width.

**W_O ∈ ℝ^(d × d)** — the mixing matrix (**O** for **Output**). Rows = d, must match Concat's column count (d) for the multiplication to be valid. Columns = d, chosen so the final result matches X's original width.

**MultiHead(X) = Concat(head_1, ..., head_h) · W_O**

shape: (T×d) = (T×d)·(d×d)

**MultiHead(X) ∈ ℝ^(T × d)** — same shape as X, always.

### A concrete numeric walkthrough (h=2, T=3, d=4)

Say head 1 (using its own W_Q^(1), W_K^(1), W_V^(1)) produces:

```
head_1 =  [ 1.20   1.00 ]   ← word A's output from head 1
          [ 1.00   1.20 ]   ← word B
          [ 1.26   1.26 ]   ← word C
```

And head 2 (using its own, completely different W_Q^(2), W_K^(2), W_V^(2)) produces:

```
head_2 =  [ 0.50   0.30 ]   ← word A's output from head 2
          [ 0.40   0.60 ]   ← word B
          [ 0.50   0.50 ]   ← word C
```

**Concatenate** — stick them side by side, row by row:

```
Concat =  [ 1.20   1.00   0.50   0.30 ]   ← word A
          [ 1.00   1.20   0.40   0.60 ]   ← word B
          [ 1.26   1.26   0.50   0.50 ]   ← word C
```

Shape (3×4). First 2 columns = head_1, last 2 columns = head_2.

**Multiply by W_O (4×4)** — say W_O is:

```
W_O = [ 1  0  1  0 ]
      [ 0  1  0  1 ]
      [ 1  0  0  1 ]
      [ 0  1  1  0 ]
```

Take word A's row [1.20, 1.00, 0.50, 0.30] and multiply against W_O:
- Output slot 1: (1.20)(1) + (1.00)(0) + (0.50)(1) + (0.30)(0) = **1.70**
- Output slot 2: (1.20)(0) + (1.00)(1) + (0.50)(0) + (0.30)(1) = **1.30**
- Output slot 3: (1.20)(1) + (1.00)(0) + (0.50)(0) + (0.30)(1) = **1.50**
- Output slot 4: (1.20)(0) + (1.00)(1) + (0.50)(1) + (0.30)(0) = **1.50**

Word A's final output: **[1.70, 1.30, 1.50, 1.50]** — a 4-number vector that blends information from both heads together. Full output shape: (3×4), matching X.

---

## Step 9 — Plugging in our running numbers (pure substitution, sanity check)

Everything above holds for any valid d and h. Substituting our running numbers: **d = 4**, **T = 9**, **h = 2**.

- d_k = d/h = 4/2 = **2**
- d_v = d/h = 4/2 = **2**

This is exactly why Chapters 2–3 used d_k = 2 as their toy Query/Key size — it's precisely what d/h gives with d=4, h=2. Chapter 3, with no multi-head splitting, was implicitly the h=1 case.

- W_Q^(m), W_K^(m), W_V^(m) ∈ ℝ^(2×4), for each of the h=2 heads
- head_m ∈ ℝ^(9×2), for each head
- Concat(head_1, head_2) ∈ ℝ^(9×4) — since h·d_v = 2×2 = 4 = d ✓
- W_O ∈ ℝ^(4×4)
- MultiHead(X) ∈ ℝ^(9×4) — same shape as X ✓

---

## Summary

| Equation | Shape derivation |
|---|---|
| Q^(m) = X·(W_Q^(m))^T | (T×d)·(d×d_k) = (T×d_k) → forces W_Q^(m) ∈ ℝ^(d_k×d) |
| K^(m) = X·(W_K^(m))^T | (T×d)·(d×d_k) = (T×d_k) → forces W_K^(m) ∈ ℝ^(d_k×d) |
| V^(m) = X·(W_V^(m))^T | (T×d)·(d×d_v) = (T×d_v) → forces W_V^(m) ∈ ℝ^(d_v×d) |
| head_m = softmax(Q^(m)(K^(m))^T/√d_k)·V^(m) | (T×d_k)(d_k×T)→(T×T); (T×T)(T×d_v)→(T×d_v) |
| Concat(head_1..h) | (T × h·d_v) = (T×d), since h·d_v=d |
| MultiHead(X) = Concat(...)·W_O | (T×d)·(d×d) = (T×d) |

---

# Follow-up Q&A

## Q1 — "Why are you calling the per-head Query/Key size d_k? Shouldn't it be d_k^(m) or something, since it's per-head?"

**d_k is a *size*, not a *value*. By design, every head uses the exact same size** — that's why it doesn't need an (m).

Compare the two different things that could vary per head:
- **The size** of each head's Query/Key vectors — a **design choice**, made once, uniformly, when the architecture is built. All h heads are deliberately built to use the same size, d_k = d/h.
- **The actual numbers** inside each head's weight matrix — this **does** vary per head. W_Q^(1) contains different learned numbers than W_Q^(2), even though both matrices have the identical shape (d_k × d). That's why W_Q^(m) carries the superscript (m) — it points at "which specific set of learned numbers," not "which size."

Rule: **symbols describing "how big" don't get an instance index; symbols describing "what specific content" do.**

You've already seen this pattern one level down, in Chapter 2: q_t and k_t (word index t) — every word's Query has size d_k, but we never wrote "d_k(t)," because d_k is the same size for word 2 as for word 9; only the *content* of q_2 vs q_9 differs. Multi-head repeats the identical pattern one level up: d_k is the same size for every head, so no (m); W_Q^(m), Q^(m), K^(m), V^(m), head_m all vary in content across heads, so they all carry (m).

(Nothing in the math forbids designing an architecture with head-varying sizes — you'd write d_k^(m) if you genuinely wanted that. But standard Transformers use uniform sizing, so a single shared d_k is both the standard choice and the simpler one.)

## Q2 — "So all heads will have the same size of d_k?"

Yes — to be crystal clear: **d_k does NOT change with the head. Every single head uses the exact same d_k.**

If h=2 and d=4, then d_k = d/h = 2 for **head 1** and d_k = 2 for **head 2** — the identical number, not two different numbers. There is only **one d_k** for the whole architecture, shared by all h heads simultaneously.

What changes from head to head is **only the content** — the actual learned numbers sitting inside W_Q^(1) vs W_Q^(2) — never the size/shape of those matrices.

Analogy: two employees given identical-sized notepads (same number of pages, same page size — that's d_k) but each writes different notes on their own notepad (that's the (m)-indexed content). The notepad size is a shared spec handed to everyone; what gets written on it is individual.

## Q3 — "How did you arrive at total cost h×d²? You didn't explain that right."

**The building block**: for any matrix of shape (rows × columns), the number of individual entries is **rows × columns** — one number per grid position.

**Applying it to one head's W_Q matrix, hypothetically unshrunk**: recall W_Q^(m) ∈ ℝ^(d_k × d) — rows = d_k (output size), columns = d (input size, unchanged — every head always reads the full d-length embedding). Imagine, hypothetically, we did *not* shrink d_k — let it stay equal to d (same as a single, non-multi-head attention would use). Then W_Q^(m)'s shape becomes (d × d), and the number of individual numbers inside it = **d × d = d²**.

**Accounting for h separate heads**: each of the h heads needs its own independent W_Q matrix — head 1's and head 2's don't share any numbers. So the total, summed across all h heads = (numbers in one head's matrix) × (number of heads) = **d² × h = h·d²**.

That's the full derivation: d² comes from "rows × columns" of one unshrunk head's matrix; the extra factor of h comes from having h separate copies, one per head, with no sharing.

**The shrinking fix**: if instead each head's size is shrunk to d_k = d/h, each head's matrix becomes (d/h × d), so numbers per head = d²/h. Total across h heads = h × (d²/h) = **d²** — the h cancels, matching a single full-size head's cost.

## Q4 — "Why are we multiplying Concat with W_O? What is W_O and why is it needed? Concat already has context-aware representations from each head, right? Why do we need W_O?"

Excellent question — and you're right that Concat already contains context-aware info from every head. The problem is **how that information is arranged**, not whether it exists.

### The problem: each head's info sits in its own separate, untouched lane

Look at what Concat actually looks like, using the numeric example from Step 8 — word A's row after concatenation:

**[ 1.20, 1.00, 0.50, 0.30 ]**

The first 2 numbers (1.20, 1.00) came entirely from head_1. The last 2 numbers (0.50, 0.30) came entirely from head_2. These two pairs have never been mathematically combined — they were computed completely independently (different W_Q^(m), W_K^(m), W_V^(m) for each head) and then just placed side by side. Nothing about slot 1 knows anything about slot 3. Nothing about head_2's findings has touched head_1's findings.

If we stopped here and called Concat the final output, every downstream number would be forced to be "purely head 1's opinion" or "purely head 2's opinion" — never a genuine blend of both. If head_1 specialized in subject-verb agreement and head_2 specialized in adjective-noun pairing, the model would have no way, at this point, to produce something like "a combined signal that reflects both relationships together" — those two pieces of information just sit next to each other, unmixed.

### What W_O does: it lets heads finally "confer"

W_O is a learned matrix (its actual numbers are trained, exactly like every other W matrix in this course) whose entire job is to take that side-by-side arrangement and produce genuine combinations across head boundaries — new numbers that are weighted mixes of multiple heads' findings, not just a copy of one head's output.

### Concrete proof: compare with vs. without W_O

Without any mixing (imagine W_O were the identity matrix — does nothing): the output for word A would just be [1.20, 1.00, 0.50, 0.30], unchanged. Slot 1 is purely head_1's slot 1. Slot 3 is purely head_2's slot 1. No cross-talk, ever.

With the actual learned W_O from the Step 8 example:

```
W_O = [ 1  0  1  0 ]
      [ 0  1  0  1 ]
      [ 1  0  0  1 ]
      [ 0  1  1  0 ]
```

We computed output slot 1 = (1.20)(1) + (1.00)(0) + (0.50)(1) + (0.30)(0) = 1.70.

Look at what that actually is: **1.20 (from head_1's slot 1) + 0.50 (from head_2's slot 1)** — a genuine sum combining a piece of head_1's finding with a piece of head_2's finding. This number never existed in either head's own output — it's a brand-new value that only exists because W_O deliberately mixed the two.

Similarly, output slot 4 = (1.20)(0)+(1.00)(1)+(0.50)(1)+(0.30)(0) = 1.50, blending head_1's slot 2 with head_2's slot 1 — again, a new combination, not something either head computed alone.

### Why this matters, and why it's learned

Without W_O, the model would be stuck with rigid, pre-decided "lanes" — head 1's output always occupies the same fixed columns, head 2's always occupies different fixed columns, forever. With W_O, the network can learn, through training, exactly how much of each head's finding should blend into each final output slot — maybe some slots end up being 90% head_1 and 10% head_2, others an even 50/50 mix, others ignore one head almost entirely. That flexibility is learned, not hand-designed, and it's what gives the model the chance to combine what different heads noticed into unified, richer features.

The name fits: **O = Output** — it produces the single, final, unified output of the whole multi-head attention block, out of pieces that had been computed in isolation from each other.
