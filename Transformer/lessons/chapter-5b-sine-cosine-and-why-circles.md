# Chapter 5b — Sine, Cosine, and Why Circles Encode Position

---

## Part A — Sine and cosine, from scratch

### Step 1 — The picture behind everything: a point moving around a circle

Imagine a circle centered at the origin (0,0), with a **radius of exactly 1** — every point on it is exactly 1 unit from the center. This specific circle is called the **unit circle** ("unit" = 1, same sense as a "unit vector" being a vector of length 1).

Picture a single point, starting at the circle's rightmost edge — position (1, 0) — then **moving counterclockwise** around the circle's edge. Sine and cosine are both just descriptions of **where this point is** as it travels.

### Step 2 — Measuring how far the point has traveled: radians

You know **degrees** (360° = one full trip) — a human convention, not derived from the circle itself. Math instead uses **radians**, defined directly from the circle's geometry: **1 radian = the angle swept when the arc length traveled equals the radius.**

A circle's full circumference = 2π × radius (π ≈ 3.14159). For our radius-1 circle, circumference = 2π ≈ 6.2832. Since a full trip covers that whole arc length, **a full trip around the circle = 2π radians** — matching 360°. So: **360° = 2π radians**, **180° = π radians**, **90° = π/2 radians**.

**Why radians specifically, for our formula?** They tie the angle directly to physical distance traveled around the circle — no extra conversion factor needed. That's why the positional encoding formula can plug a raw position number straight in as an angle, with no rescaling required for the geometry to work.

### Step 3 — Defining sin(θ) and cos(θ)

Name the swept angle **θ** (theta) — the conventional label mathematicians use for "the angle," same role as t for "word position" or d for "dimension" elsewhere in this course.

As the point travels to angle θ, it sits at some (x, y) location:

**cos(θ) = the x-coordinate of the point, at angle θ**
**sin(θ) = the y-coordinate of the point, at angle θ**

That's the entire definition: the horizontal and vertical position of a point walking around a radius-1 circle.

### Step 4 — Concrete checkpoints around the circle

- **θ = 0**: point at (1, 0) → cos(0) = **1**, sin(0) = **0**
- **θ = π/2 ≈ 1.5708** (90°): point at (0, 1) → cos(π/2) = **0**, sin(π/2) = **1**
- **θ = π ≈ 3.1416** (180°): point at (−1, 0) → cos(π) = **−1**, sin(π) = **0**
- **θ = 3π/2 ≈ 4.7124** (270°): point at (0, −1) → cos(3π/2) = **0**, sin(3π/2) = **−1**
- **θ = 2π ≈ 6.2832** (360°): back to (1, 0) → cos(2π) = **1**, sin(2π) = **0**, identical to θ=0

### Step 5 — Why sin/cos never leave [−1, 1]

The point can never leave the circle, and the circle never extends farther than 1 unit from center in any direction. Since cos(θ) and sin(θ) are just that point's x and y coordinates, they **can never** exceed 1 or drop below −1, for any θ. This is exactly the "bounded" requirement from Chapter 5, now grounded in *why* it's guaranteed.

### Step 6 — Why sin/cos repeat forever (periodicity)

Once the point completes a full loop (θ=2π), it's back exactly where it started at θ=0, and the journey repeats identically. This is **periodicity**; the distance you travel before the pattern exactly repeats is the **period** — 2π, for both sin and cos.

### Step 7 — Why we need BOTH sin and cos together, not just one alone

**sin alone can't always tell you where the point is.** sin(30°) = 0.5, but sin(150°) also = 0.5 — two different angles, identical sin value. **cos** resolves the ambiguity: cos(30°) ≈ 0.866, cos(150°) ≈ −0.866 — clearly different. Looking at sin **and** cos together, as a pair, uniquely pins down the angle — the ambiguity either one alone has gets resolved once you have both. This is exactly why positional encoding always generates sin and cos **in pairs**, one pair per frequency.

---

## Part B — Why circles specifically, for encoding word position?

### Step 8 — Why naive alternatives fail

- **Raw position number** (PE_500 = [500,500,...]): unbounded — swamps content embeddings for long sequences.
- **One-hot vector** (all zeros, a single 1 in slot t): only works up to d positions; and has **no** notion of closeness — position 5's vector is exactly as different from position 4's as from position 500's.
- **Random fixed vectors per position**: unique, but no structure — nearby positions wouldn't look similar, and there'd be no way to compute a vector for a position never explicitly assigned one.

None of these give boundedness **and** smoothness **and** a cheap way to reason about relative offsets, all at once.

### Step 9 — The real payoff: shifting position by a fixed amount is a *linear* operation

Circles give boundedness (Step 5) and smoothness (nearby angles → nearby coordinates) for free. But there's a third, more powerful reason circles specifically were chosen, involving the **angle addition formulas**:

**cos(θ+k) = cos(θ)cos(k) − sin(θ)sin(k)**
**sin(θ+k) = sin(θ)cos(k) + cos(θ)sin(k)**

The new coordinates (cos(θ+k), sin(θ+k)) are a **weighted sum** of the old coordinates (cos(θ), sin(θ)), using weights (cos k, sin k) that depend **only on the offset k**, never on the starting angle θ. Written as a matrix (in the natural x-then-y order established in Step 3):

```
[ cos(θ+k) ]   [ cos k   −sin k ] [ cos θ ]
[ sin(θ+k) ] = [ sin k    cos k ] [ sin θ ]
```

**Verify with concrete numbers.** Take θ=1, k=1 (so θ+k=2): cos(1)≈0.5403, sin(1)≈0.8415.

```
[ cos(2) ]   [ cos(1)   −sin(1) ] [ cos(1) ]   [ 0.5403×0.5403 − 0.8415×0.8415 ]   [ −0.4162 ]
[ sin(2) ] = [ sin(1)    cos(1) ] [ sin(1) ] = [ 0.8415×0.5403 + 0.5403×0.8415 ] = [  0.9093 ]
```

Matches the true values cos(2)≈−0.4161, sin(2)≈0.9093 (tiny rounding). **Moving forward by k=1 was literally just multiplying by a fixed 2×2 matrix.**

**Why this matters for a Transformer specifically:** everything else in the architecture — W_Q, W_K, W_V, W_O — is built from linear operations (matrix multiplications). By encoding position with circles, "shift by k positions" becomes yet another linear operation, exactly the kind of thing the network's existing machinery (its Q/K/V projections) can learn to exploit. Had we picked an encoding where "shift by k" required some complicated nonlinear transformation, the network's linear machinery would have a much harder time learning relative-position patterns like "attend to whatever is 3 words back." Circles were chosen because rotation — their one defining operation — happens to be linear, meshing perfectly with an architecture built entirely from linear pieces.

### Step 10 — Why one circle isn't enough: the aliasing problem

A single sin/cos pair repeats every 2π ≈ 6.2832 radians. Even within our small T=9 example, this shows up: compare position 0 (angle 0, coordinates [cos,sin]=[1, 0]) to position 6 (angle 6, just 0.283 radians short of a full lap — coordinates [cos(6), sin(6)] ≈ [0.960, −0.279]). Position 6 is noticeably *closer* to position 0's coordinates than, say, position 3 is (coordinates [cos(3),sin(3)] ≈ [−0.990, 0.141], wildly different) — purely because position 6 is nearly back around to where it started, **despite being farther away in the actual sentence than position 3 is.**

For a real document with thousands of words, this wraparound effect gets far more severe: many pairs of positions, however far apart in the actual text, could produce nearly identical fast-pair readings, purely by coincidence of the wraparound.

**Why not just use a slow pair instead, to avoid wrapping around?** Because a slow pair (like the divide-by-100 pair from Chapter 5c) barely moves between *adjacent* positions — recall sin(0)=0.0000 vs sin(0.01)≈0.0100 — nearly indistinguishable in practice.

**Neither extreme works alone.** The fix: use **many** circles simultaneously, spinning at many different speeds — exactly like **place-value digits in a number**. The ones-digit alone repeats every 10 and can't distinguish 3 from 13 or 23; combined with a tens-digit, a hundreds-digit, and so on, the full combination uniquely and smoothly identifies numbers across a huge range — no single digit needs impossible precision, because each one only resolves its own scale. The multiple frequency-pairs in positional encoding play exactly this role.

---

## Summary

| Term | Meaning |
|---|---|
| Unit circle | radius-1 circle centered at origin — the geometric object sin/cos are defined from |
| θ (theta) | the angle swept, in radians |
| Radian | angle where arc length traveled = radius; full circle = 2π radians |
| cos(θ), sin(θ) | x-coordinate, y-coordinate of the point at angle θ |
| Bounded, periodic | guaranteed by the circle's geometry (Steps 5–6) |
| Why sin+cos together | resolves the ambiguity either one alone has (Step 7) |
| Why circles for position | bounded + smooth + shifting by k is a **linear** (matrix) operation — matches a Transformer's all-linear architecture |
| Why multiple frequencies | one circle aliases (wraps around, confusing distant positions); many speeds together, like digits of a number, resolve position at every scale |
