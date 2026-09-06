# Chapter 5c — Decoding the Positional Encoding Formula

---

## Step 1 — The formula, in full

**PE(pos, 2i) = sin( pos / 10000^(2i/d) )**
**PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )**

Every symbol gets defined precisely below — including the parts that look like bookkeeping but actually aren't.

## Step 2 — pos: the position, 0-indexed

**pos** is the position, using **0-indexed** counting (first word = position 0) — the standard convention for this formula. Throughout this course we've used **t** as a **1-indexed** word position (t=1 for "The," ..., t=9 for "are"). To use the formula, convert with **pos = t − 1** (word 1 → pos=0, word 9 → pos=8). Just a counting-start difference, but needs to be explicit so numbers line up.

## Step 3 — i: which pair, not which slot

We know from Chapter 5b that we need **pairs** of sin/cos, each pair spinning at a different speed. With **d** total slots in PE_t, and each pair using 2 slots, the number of pairs available is **d/2**.

**i** is the label for **which pair** — "pair number i." Since there are d/2 pairs, **i ranges from 0 to (d/2 − 1)**. With our running example d=4, that's d/2=2 pairs, so **i ∈ {0, 1}**.

## Step 4 — 2i and 2i+1: the actual slot addresses

**2i and 2i+1 answer a different question from i**: not "which pair," but "**which of the d boxes in the PE vector does this pair's value get written into?**"

Picture d boxes in a row, numbered 0 to d−1. Pairs claim 2 consecutive boxes each, one after another: pair 0 gets the first 2, pair 1 the next 2, and so on. Before pair i starts, all **i** earlier pairs have already claimed 2 boxes each — **2i** boxes used up. So pair i's first box is at position **2i**, and its second box, right after, is at position **2i+1**.

With d=4 (pairs i=0,1):
- Pair i=0: boxes 2(0)=**0** and 2(0)+1=**1**
- Pair i=1: boxes 2(1)=**2** and 2(1)+1=**3**

All 4 boxes used exactly once — no gaps, no overlaps. (Same bookkeeping as numbering a row of shoe-pairs: pair i's left shoe is always at position 2i, right shoe at 2i+1.)

## Step 5 — Why sin lives at slot 2i and cos at slot 2i+1

Honestly: **this is a convention**, not a mathematical necessity. There's no deep reason sin must come before cos rather than the reverse — someone had to pick an order in the original paper, and "sin first (even slot), cos second (odd slot)" is what stuck. What matters is that it's applied **consistently** across every pair.

**Important consequence**: this means the PE vector stores each pair as **[sin θ, cos θ]** — sin first — which is the *reverse* of the natural (x,y) = (cos θ, sin θ) geometric order used to originally *define* sin/cos in Chapter 5b. Not a contradiction — just two different, independently-motivated conventions (one geometric, one about laying out a flat vector) that happen to disagree on order.

**Re-deriving the rotation matrix for this actual storage order** (as a direct callback to Chapter 5b Step 9, adapted to match): starting from the same two identities,

sin(θ+k) = sin(θ)cos(k) + cos(θ)sin(k)
cos(θ+k) = cos(θ)cos(k) − sin(θ)sin(k)

regrouped to match input order [sin θ, cos θ]:

```
[ sin(θ+k) ]   [  cos k   sin k ] [ sin θ ]
[ cos(θ+k) ] = [ −sin k   cos k ] [ cos θ ]
```

Check with pos=0 (θ=0, so [sinθ,cosθ]=[0,1]) shifting by k=1:

sin(0+1) = cos(1)×0 + sin(1)×1 = 0.8415
cos(0+1) = −sin(1)×0 + cos(1)×1 = 0.5403

Result **[0.8415, 0.5403]** — matches PE_2's fast pair [sin(1), cos(1)] = [0.8415, 0.5403] from Chapter 5, entry-for-entry. The rotation property survives the reordering; only the matrix's entries needed to be arranged to match.

## Step 6 — Why the exponent must be shared between sin and cos in a pair (not tracking the individual slot)

Here's the sharpest question: **why does the cos formula (living at slot 2i+1) reuse the exponent "2i/d" instead of using its own slot number, "(2i+1)/d"?**

Recall: sin(θ) and cos(θ) are the two coordinates of **one single point**, at **one single angle θ**, on **one single circle**. For the rotation-as-linear-shift property (Step 5) to hold, **the sin and cos within a pair must be evaluated at the exact same angle θ** — they're not two independent numbers, they're two views of one rotating point.

If the cos formula used a *different* exponent (based on its own slot number, 2i+1, instead of reusing the pair's shared index i), the sin part and cos part of that "pair" would be sampled from **two different circles spinning at two different speeds** — no longer the (x,y) coordinates of a single point, and the rotation-matrix relative-position property would break down completely (cos(θ_A+k) and sin(θ_B+k), with θ_A ≠ θ_B, do not combine via a clean matrix).

So the exponent is keyed to **which pair** (i), never to **which individual slot** — that's why the identical "2i/d" appears in both formulas.

## Step 7 — Why "2i/d" and not just "i/d"? The rescaling reason

Separate question: given the exponent must be shared, why is it written as **2i/d** specifically?

Goal: the exponent should sweep smoothly from **0** (divisor 10000⁰=1, fastest pair) up to **(nearly) 1** (divisor 10000¹=10000, slowest pair), as i sweeps across all pairs, i=0 to i=d/2−1.

**If we used exponent = i/d** (no doubling): since i only reaches up to d/2−1 (i counts *pairs* — half as many as there are slots), the largest exponent would be roughly (d/2)/d = **1/2** — only halfway. Even the slowest pair would only reach divisor 10000^0.5=100, never the full 10000. Half the intended frequency range wasted.

**The fix**: multiply i by 2 first. Since i's natural range (0 to d/2) is exactly half of d's range (0 to d), doubling i rescales it to match: **2i/d** now sweeps 0 to (nearly) 1 properly. **The "×2" exists purely to correct for i counting pairs (half as many as slots) while d counts individual slots.**

Verify with d=4 (only 2 pairs — small, won't reach the full range, but mechanism is identical):
- i=0: exponent = 2(0)/4 = 0 → divisor = 10000⁰ = **1**
- i=1 (last pair here): exponent = 2(1)/4 = 0.5 → divisor = 10000^0.5 = **100**

A real model with larger d (say d=512, 256 pairs) would have i reach 255, giving exponent 2(255)/512≈0.996, divisor≈9770 — nearly the full 10000. Our toy d=4 is just too small to see the whole range play out, but the construction is identical.

## Step 8 — The two roles of "2i," side by side

| Where it appears | What it means | Why "2i" specifically |
|---|---|---|
| PE(pos, **2i**) — argument | Which **slot** this pair's sin value is written to | 2i, because i earlier pairs each used 2 slots (Step 4) |
| 10000^(**2i**/d) — exponent | Which **frequency** this whole pair shares | 2i, to rescale i's range (0 to d/2) to match d's range (0 to d) (Step 7) |

Two genuinely different quantities, written with the same expression, for two independent reasons — that overlap is exactly what made the formula look redundant at first glance.

## Step 9 — Worked example: computing PE_t by hand, for our running sentence

With d=4: i=0 gives exponent 0 (divisor 1); i=1 gives exponent 0.5 (divisor 100). So:

PE(pos,0)=sin(pos), PE(pos,1)=cos(pos), PE(pos,2)=sin(pos/100), PE(pos,3)=cos(pos/100)

**pos=0 (t=1, "The"):** [sin(0), cos(0), sin(0), cos(0)] = **[0.0000, 1.0000, 0.0000, 1.0000]**

**pos=1 (t=2, "cats"):** [sin(1), cos(1), sin(0.01), cos(0.01)] ≈ **[0.8415, 0.5403, 0.0100, 0.9999]**

**pos=8 (t=9, "are"):** [sin(8), cos(8), sin(0.08), cos(0.08)] ≈ **[0.9894, −0.1455, 0.0799, 0.9968]**

Comparing pos=0 to pos=1 (adjacent): the fast pair (slots 0,1) swings dramatically (0.00→0.84, 1.00→0.54); the slow pair (slots 2,3) barely moves (0.00→0.01, 1.00→0.9999). Comparing pos=0 to pos=8 (7 apart): the slow pair has now moved further too (0.00→0.08, 1.00→0.9968) — still far less than the fast pair already showed at just 1 step, exactly the "multiple clock hands" behavior from Chapter 5.

## Step 10 — Learned vs. fixed positional encodings

This sinusoidal formula produces **fixed** encodings — computed once, never updated during training, identical for every sentence (PE for position 1 is always PE for position 1, whatever word occupies it). This was the original Transformer paper's choice.

Some later models (e.g. BERT) instead use **learned** positional encodings — PE_t as free parameters, initialized randomly and updated via gradient descent like any weight matrix. Both are used in practice; we stick with the fixed sinusoidal version here since it needs no training and is fully hand-computable.

## Step 11 — Where this fits into the pipeline

1. Look up each word's content embedding: **e_t ∈ ℝ^(d×1)**
2. Compute its positional encoding: **PE_t ∈ ℝ^(d×1)**, via Steps 1–9 above
3. Add: **x_t = e_t + PE_t ∈ ℝ^(d×1)** — the actual input Chapters 2–4 were calling "word t's embedding"
4. Stack into **X ∈ ℝ^(T×d)**, exactly as before
5. Feed into W_Q, W_K, W_V, self-attention, multi-head attention — Chapters 2–4, unchanged

Positional encoding happens once, at the very start, before any attention computation.

---

## Summary

| Symbol | Meaning |
|---|---|
| pos | 0-indexed position; pos = t − 1 |
| i | pair index, 0 to d/2−1 — which sin/cos pair |
| 2i, 2i+1 | slot addresses in the d-length PE vector — where this pair's sin/cos values are written |
| 10000^(2i/d) | the shared divisor/frequency for pair i — same for both sin and cos in that pair, since they represent one rotating point |
| Why sin↔2i, cos↔2i+1 | convention (arbitrary but consistent), reversing the natural (cos,sin)=(x,y) order from Ch.5b |
| Why shared exponent | sin and cos in a pair must share one angle θ, or the rotation/linear-shift property breaks |
| Why "×2" in 2i/d | rescales i's range (0 to d/2) to match d's range (0 to d), so the exponent spans ~0 to ~1 |
