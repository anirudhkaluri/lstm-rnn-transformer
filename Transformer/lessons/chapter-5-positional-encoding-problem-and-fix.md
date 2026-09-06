# Chapter 5 — Positional Encoding: The Problem and The Fix

---

## Step 1 — The problem: self-attention has no idea what order the words are in

Look precisely at every equation built in Chapters 2–4:

q_t = W_Q·x_t
k_t = W_K·x_t
v_t = W_V·x_t
score(i,j) = q_i · k_j

**The number t itself never appears anywhere as a value being computed with.** t only ever shows up as a *label* — "which row of X are we talking about." The actual arithmetic only ever touches the **content** sitting in that row — never the row's position in the stack.

Precisely stated: **if you shuffled the 9 rows of X into a different order, self-attention would compute the exact same set of relationships between word-contents — just relabeled to match the new order.** The mechanism has no built-in sense of "this word came before that word," "these two words are adjacent," or "these two are far apart." It only measures "how similar is this word's content to that word's content."

## Step 2 — Why this actually breaks things: a concrete example

> "The cats chased the dog."
> "The dog chased the cats."

Same five words, different order, opposite meaning — in the first, cats is the chaser; in the second, cats is chased. Each word's embedding is identical across both sentences ("cats" always gets the same embedding vector). **The only thing distinguishing these two sentences is word order** — and self-attention, as built so far, cannot see order at all.

Compare this to RNN/LSTM (Chapter 1): order was automatic there, because h_t could only ever be computed *after* h_{t-1} — the sequential recurrence itself encoded "which word came first." Self-attention deliberately threw away that dependency (that's exactly why it parallelizes so well), but as a side effect, it threw away order information too. We need to add it back, some other way.

## Step 3 — The fix, at a high level: give each position its own vector, and add it in

Create a second vector, alongside each word's content, that encodes purely *where* that word sits — nothing about what the word means. Combine the two before self-attention ever runs.

New names (this also retroactively clarifies a gap left open in Chapters 2–4, where "x_t" was called simply "word t's embedding" without addressing position):

- **e_t ∈ ℝ^(d×1)** — word t's **pure content embedding**: describes only what the word *means*, nothing about where it sits. (This is what we were informally calling x_t before.)
- **PE_t ∈ ℝ^(d×1)** — word t's **positional encoding**: describes only *where* position t is, nothing about which word occupies it. Same size d as e_t — required, since we're about to add them together, and addition only works between vectors of matching length.

And now, precisely, what Chapters 2–4 called **x_t all along is actually**:

**x_t = e_t + PE_t**

— elementwise addition. This combined vector, carrying both "what" and "where" baked into the same d numbers, is what actually feeds into W_Q, W_K, W_V.

**Why addition, not concatenation (sticking PE_t onto the end, making a longer vector)?** Two reasons: (1) addition keeps the input exactly d-dimensional, so nothing about W_Q/W_K/W_V's shapes needs to change. (2) It lets every one of the d slots potentially carry a mix of content and position — the network, through its learned weights, can decide how much of its attention patterns should be driven by content versus position, using the whole space flexibly, rather than being forced into separate walled-off sections.

## Step 4 — What should PE_t actually look like? (requirements, before any formula)

Before reaching for a specific formula, it's worth being explicit about what we actually need PE_t to satisfy:

1. **Bounded** — shouldn't blow up in size for long sentences. If PE_500 were something like [500,500,...], that single huge number would completely dominate the small, carefully-tuned content numbers in e_t once added together.
2. **Unique per position** — PE_2 must look different from PE_5, or the model can't tell those positions apart.
3. **Smooth / distance-aware** — nearby positions (like 2 and 3) should look more similar to each other than far-apart positions (like 2 and 9) — giving the model a notion of "adjacent" vs. "far away," not just "different."
4. **Ideally, cheap to shift** — the network should be able to reason about *relative* position ("3 words back"), not just memorize each absolute position independently.

A rough intuition for what could satisfy all of this: imagine a **clock with multiple hands** — second hand, minute hand, hour hand — moving at different speeds. Looking at all the hands together uniquely identifies any moment across a wide span, because the fast hand distinguishes nearby moments while the slow hand distinguishes far-apart moments — no single hand alone could do the whole job.

The next two chapters (5b, 5c) build the actual mathematical tool that satisfies all four requirements — sine and cosine, arranged in pairs at different speeds — and then decode the exact formula, symbol by symbol.

---

## Summary

| Symbol | Shape | Plain meaning |
|---|---|---|
| e_t | ℝ^(d×1) | word t's pure content embedding — "what" (no position info) |
| PE_t | ℝ^(d×1) | word t's positional encoding — "where" (no content info) |
| x_t = e_t + PE_t | ℝ^(d×1) | the actual input to attention — content and position combined |
