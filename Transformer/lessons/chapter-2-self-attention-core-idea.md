# Chapter 2 — Self-Attention: The Core Idea

---

## Step 1 — Recap: what are X, T, d?

**X** is a spreadsheet: one row per word in our sentence, one column per "meaning number" describing that word.

- **T** = number of rows = number of words = **9**, for "The cats that were sitting on the mat are hungry" (the input words — "hungry" is what we're trying to predict, so it's not a row of X).
- **d** = number of columns = how many numbers describe each word's meaning = **4** (small, so we can do arithmetic by hand).

So **X ∈ ℝ^(T×d) = ℝ^(9×4)**. Row 2 of X is the 4 numbers describing "cats". Row 9 is the 4 numbers describing "are".

---

## Step 2 — The big question

Row 9 ("are") only contains "are"'s generic, out-of-context meaning. But "are" vs "is" depends on whether its subject is plural — and that fact lives in row 2 ("cats"), 7 rows away.

**Question this chapter answers**: how does row 9 reach over to row 2, and pull in exactly the relevant part of it — directly, in one step, without passing through rows 3–8?

---

## Step 3 — The analogy: meet Query, Key, AND Value together, with no math yet

Picture this: word 9 ("are") stands in a room with 8 mailboxes — one per other word (positions 1–8). Every mailbox has **two things**: a **label** on the front, and **contents** inside. Word 9 is holding **one note**.

Three roles, all introduced right now, together:

- **The note word 9 holds** = its way of saying *"here's what I need."* "are" needs to know "is my subject singular or plural?" → This is called the **Query**.
- **The label on a mailbox** = that word's way of saying *"here's what kind of thing I am, so your note can be checked against me."* "cats"'s label might say "plural noun, possible subject." → This is called the **Key**.
- **The contents inside a mailbox** = the actual substance you'd receive if that mailbox were chosen. → This is called the **Value**.

The process: word 9 holds its **note (Query)** up against every mailbox's **label (Key)**, finds the best matches, then mixes together the **contents (Values)** — mostly from the best matches, a little from the rest.

**Checkpoint**: Query, Key, and Value all now exist as plain-English concepts, with zero numbers attached. Nothing below uses any of these three words without you already knowing what they mean.

---

## Step 4 — Query and Key, formalized together (because their sizes are linked)

The note (Query) and every label (Key) need to become **lists of numbers** — because in a moment we'll **compare** the note against each label, and a numeric comparison only works if both lists have the **same number of entries**.

- **q_t** = word t's Query — its note, written as a list of numbers.
- **k_t** = word t's Key — its label, written as a list of numbers.

Because q_t and k_t must have the same length to be compared, that shared length gets **one name**. Here's where the name comes from:

- The shared length is written **d_k**.
- **d** = "dimension" (a count of numbers) — same role as the d in X ∈ ℝ^(T×d).
- **k** = stands for **Key** — the Key concept from Step 3 (the mailbox label). The shared length is named after Key.
- Query's vector q_t is *forced* to be this same length (so it can be compared to k_t) — rather than inventing a separate name for Query's length, the field just reuses **d_k** for both.

For our example: **d_k = 2**. So **q_t ∈ ℝ^(2×1)** and **k_t ∈ ℝ^(2×1)** — each is a list of 2 numbers.

(Quick note on how this relates to d from Step 1: d = 4 is the size of the *original* word descriptions in X. d_k = 2 is the size of the *Query/Key* lists — a smaller, different number, chosen by us for easy arithmetic. *How* a 4-number word description becomes a 2-number Query or Key is Chapter 3's job. For now, just notice d and d_k are allowed to be different numbers measuring different things.)

---

## Step 5 — Value, formalized

The contents (Value) have a different job from Query/Key — they're not for comparing, they're the actual payload delivered once a match is found. So there's no requirement that Value's list be the same length as q_t/k_t's.

- **v_t** = word t's Value — its mailbox contents, written as a list of numbers.
- **v_t ∈ ℝ^(d_v×1)**, where **d_v** is named the same way d_k was: **d** = dimension (count of numbers), **v** = stands for **Value** (the concept from Step 3).
- For our example: **d_v = 2** (this didn't have to equal d_k — it's just a coincidence of our chosen numbers).

**Before computing**: in a real Transformer, q_t, k_t, v_t are *computed* from x_t (a row of X) — that's Chapter 3. For this chapter, we're given their values directly so we can focus on what attention *does* with them, not where they come from.

---

## Step 6 — Worked example: word 9 ("are") checks words 2, 5, 8

To keep the arithmetic small, "are" (word 9) will compare itself against just 3 candidates: "cats" (word 2), "sitting" (word 5), "mat" (word 8). The real model checks all 8 — same math, more terms.

### 6a — "are"'s Query (its note)

**q_9 = [1.0, 0.0]** — a list of 2 numbers (since d_k=2). This is "are"'s note: "I'm looking for something that scores high in slot 1." (What "slot 1" means semantically is something the network learns during training — we're tracking pure arithmetic here.)

### 6b — Keys for the candidates (their labels)

- **k_2 ("cats") = [0.9, 0.1]** — mostly "slot 1"
- **k_5 ("sitting") = [0.0, 1.0]** — entirely "slot 2"
- **k_8 ("mat") = [0.1, 0.9]** — mostly "slot 2"

### 6c — Comparing note against label: the score

To compare two lists of numbers, we use the **dot product**: multiply each pair of matching positions together, then add up the results.

We give this comparison a name: **score(i,j)**, where:
- **i** = the row number of the word doing the asking (here, i=9, "are")
- **j** = the row number of the candidate word being checked (here, j=2, 5, or 8)
- **score(i,j) = q_i · k_j** = "how well word j's label matches word i's note" — a single number.

- score(9,2) = q_9 · k_2 = (1.0)(0.9) + (0.0)(0.1) = **0.9**
- score(9,5) = q_9 · k_5 = (1.0)(0.0) + (0.0)(1.0) = **0.0**
- score(9,8) = q_9 · k_8 = (1.0)(0.1) + (0.0)(0.9) = **0.1**

q_9 = [1.0, 0.0] only "cares about" slot 1. The dot product is asking each label "how much do you have in slot 1?" — "cats" answers 0.9, "sitting" answers 0.0, "mat" answers 0.1. "cats" wins — correctly, since "cats" is "are"'s subject.

### 6d — Turning scores into a "recipe": softmax

The raw scores [0.9, 0.0, 0.1] aren't yet usable as mixing proportions — we need positive numbers that add up to exactly 1 (a recipe: "54% of this, 22% of that..."). **Softmax** does this: exponentiate every number (makes everything positive, and stretches out the differences), then divide each by the total of all of them.

- exp(0.9) = 2.4596, exp(0.0) = 1.0000, exp(0.1) = 1.1052 → total = **4.5648**

The results are called **α(i,j)** — same i and j as before (i = the word asking, j = the candidate) — meaning *"the fraction of word i's attention given to word j"*:

- α(9,2) = 2.4596 / 4.5648 = **0.5387** (≈54%)
- α(9,5) = 1.0000 / 4.5648 = **0.2191** (≈22%)
- α(9,8) = 1.1052 / 4.5648 = **0.2422** (≈24%)

Check: 0.5387 + 0.2191 + 0.2422 = 1.0000 ✓

### 6e — The Values (the actual contents)

- v_2 ("cats") = [5.0, 0.0]
- v_5 ("sitting") = [0.0, 3.0]
- v_8 ("mat") = [1.0, 1.0]

### 6f — Mixing: the output

We define **output_i** = "word i's new, context-aware representation" as:

**output_i = Σ_j α(i,j) · v_j**

Read the Σ_j as: "go through every candidate j, take its Value v_j, scale it down by the attention weight α(i,j) it received, and add all those scaled Values together."

output_9 = 0.5387×[5.0,0.0] + 0.2191×[0.0,3.0] + 0.2422×[1.0,1.0]

- 0.5387×[5.0,0.0] = [2.6935, 0.0000]
- 0.2191×[0.0,3.0] = [0.0000, 0.6573]
- 0.2422×[1.0,1.0] = [0.2422, 0.2422]

Adding slot by slot: **output_9 = [2.9357, 0.8995] ≈ [2.94, 0.90]**

---

## Step 7 — What does this mean?

output_9 ≈ [2.94, 0.90] vs. the raw Values:
- v_2 ("cats") = [5.0, 0.0]
- v_5 ("sitting") = [0.0, 3.0]
- v_8 ("mat") = [1.0, 1.0]

output_9's first number (2.94) is close to 54% of "cats"'s 5.0 — output_9 is **mostly "cats"**, with small contributions from "sitting" and "mat". "are" reached across the sentence and absorbed "cats"'s information **directly — one dot product, one softmax, one weighted sum**. No 7-step chain like in LSTM; distance never entered the computation.

---

## Step 8 — Every word does this at once — this is "self-attention"

All 9 words run steps 6a–6f simultaneously, each with its own Query, producing a new 9-row spreadsheet of outputs. The name **self-attention** comes from: the Queries, Keys, AND Values **all come from the same sentence X** — every word is simultaneously a questioner (via its own Query) and a candidate answer for everyone else (via its Key and Value).

(Chapter 7 will cover **cross-attention** — where Queries come from one sequence but Keys/Values come from a different one.)

---

## Step 9 — What's left for Chapter 3

1. Where q_t, k_t, v_t actually come from (matrices W_Q, W_K, W_V applied to x_t).
2. The √d_k scaling factor applied before softmax, and why it matters when d_k is larger than our toy value of 2.

---

## Summary

| Concept | Symbol | Size | Plain meaning |
|---|---|---|---|
| Sequence matrix | X | ℝ^(T×d) = ℝ^(9×4) | spreadsheet: T=9 words (rows), d=4 meaning-numbers each (columns) |
| Query | q_t | ℝ^(d_k×1) | word t's "note" — what it's looking for |
| Key | k_t | ℝ^(d_k×1) | word t's "label" — what it advertises, for matching |
| Value | v_t | ℝ^(d_v×1) | word t's "contents" — actual info it offers |
| d_k | — | scalar (=2 here) | shared size of Query & Key, named after Key |
| d_v | — | scalar (=2 here) | size of Value, named after Value, can differ from d_k |
| Score | score(i,j) = q_i · k_j | scalar | how well word j's Key matches word i's Query |
| Attention weight | α(i,j) = softmax_j(score(i,j)) | scalar, Σ_j α(i,j)=1 | normalized relevance — the "recipe" |
| Output | output_i = Σ_j α(i,j)·v_j | ℝ^(d_v×1) | word i's new, context-aware representation |
