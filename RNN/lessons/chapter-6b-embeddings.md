# Chapter 6b: Embeddings — How Words Become Vectors

---

## Why One-Hot Encoding Fails for Words

For characters (27 tokens) one-hot encoding is fine. Small and manageable.

For words it breaks down completely.

```
Vocabulary = 50,000 words

"the" = [1, 0, 0, 0, 0, ... 0]   ← 50,000 numbers
"dog" = [0, 1, 0, 0, 0, ... 0]   ← 50,000 numbers
"ran" = [0, 0, 1, 0, 0, ... 0]   ← 50,000 numbers
```

99.998% of every vector is zeros. This is called a **sparse** representation.

### Problem 1 — Memory and Compute

```
Wₓ = hidden_size × vocab_size
   = 256 × 50,000
   = 12,800,000 weights

Just to process ONE word at ONE time step. Enormous.
```

### Problem 2 — Meaning is Completely Lost

```
"dog" = [0, 1, 0, 0, 0, ...]
"cat" = [0, 0, 1, 0, 0, ...]
"car" = [0, 0, 0, 1, 0, ...]

Distance between "dog" and "cat"  =  same as
Distance between "dog" and "car"

But dog and cat are both animals — the network has no idea.
Every word looks equally different from every other word.
```

---

## The Fix — Word Embeddings

Instead of a 50,000-sized sparse vector, map every word to a **small dense vector** of say 300 numbers:

```
One-hot:                          Embedding:

"dog" = [0,1,0,...,0]    →    [0.2,  0.8, -0.3, 0.5, ...]
         50,000 numbers              300 numbers

"cat" = [0,0,1,...,0]    →    [0.3,  0.7, -0.1, 0.4, ...]
         50,000 numbers              300 numbers

"car" = [0,0,0,...,0]    →    [-0.5, 0.1,  0.9, 0.2, ...]
         50,000 numbers              300 numbers
```

Now:
```
"dog" and "cat" vectors are CLOSE to each other   ✅
"dog" and "car" vectors are FAR from each other   ✅
```

The network understands similarity through geometry.

---

## The Famous Example

```
king  -  man  +  woman  ≈  queen
```

This works because embeddings capture meaning as directions in space:

```
king  = [0.8, 0.3, 0.7, ...]
man   = [0.7, 0.1, 0.2, ...]
woman = [0.6, 0.2, 0.1, ...]

king - man + woman  ≈  queen vector  🎉
```

You can do arithmetic on meaning. One-hot could never do this.

---

## How Embeddings Fit into the RNN

Instead of feeding xₜ (50,000 numbers) directly into the RNN, you add an embedding layer first:

```
BEFORE (one-hot):
  word → [0,0,1,0,...,0]   (50,000)  →  RNN
                ↑
           wasteful, no meaning

AFTER (embedding):
  word → [0,0,1,...,0]  →  Embedding Layer E  →  [0.2, 0.8, ...]  →  RNN
          (50,000)              (lookup)               (300)
```

The embedding layer E is just a matrix:

```
E = vocab_size × embedding_size
  = 50,000 × 300
  = one row per word
  = 15,000,000 numbers total
```

Looking up a word's embedding = just selecting one row from E.

And now Wₓ becomes much smaller:

```
WITHOUT embeddings:   Wₓ = 256 × 50,000  =  12,800,000  weights
WITH embeddings:      Wₓ = 256 × 300     =      76,800  weights  ✅
```

---

## How Are Embeddings Created?

There are two ways:

```
1. Learned during training        ← simplest, most common
2. Pre-trained embeddings         ← Word2Vec, GloVe, BERT, Jina
```

---

## Method 1 — Learned During Training

The embedding matrix E starts completely random and gets trained via backpropagation along with everything else (Wₓ, Wₕ, Wy, etc.).

```
START of training:
  E["dog"] = [0.13, -0.42, 0.87, ...]   ← random garbage
  E["cat"] = [0.55,  0.21, -0.34, ...]  ← random garbage

AFTER training on millions of sentences:
  E["dog"] = [0.2,  0.8, -0.3, 0.5, ...]   ← meaningful
  E["cat"] = [0.3,  0.7, -0.1, 0.4, ...]   ← close to dog ✅
  E["car"] = [-0.5, 0.1,  0.9, 0.2, ...]   ← far from dog ✅
```

The network figured out similarity entirely on its own just by reading text and predicting the next word. Nobody told it that dog and cat are similar.

### How Does Backprop Update E?

E is treated just like any other weight matrix. Gradients flow all the way back into it:

```
Loss
  ↓
dL/dWy   → update Wy
  ↓
dL/dWₕ  → update Wₕ
  ↓
dL/dWₓ  → update Wₓ
  ↓
dL/dE   → update E   ← embedding matrix gets gradients too
```

---

## Method 2 — Pre-trained Embeddings

These are embedding matrices trained **separately** on massive text datasets, then reused. You download them and plug them into your model.

### Word2Vec (Google, 2013)

Trained on one simple task:

> **"Given a word, predict the words around it"**

```
Sentence: "The dog barked at the cat"

Task: given "barked" → predict ["dog", "at"]
Task: given "dog"    → predict ["The", "barked"]
```

That is it. Just predict neighbouring words. After training on billions of sentences, words appearing in similar contexts get similar vectors.

```
"dog" and "cat" both appear near: "barked", "pet", "furry", "owner"
→ their vectors become similar  ✅

"dog" and "quantum" never appear in the same context
→ their vectors stay far apart  ✅
```

### GloVe (Stanford, 2014)

Uses a different approach — counts how often words appear **together** across the whole dataset:

```
"dog"  + "bone"    appear together 15,432 times  → vectors should be close
"dog"  + "galaxy"  appear together 3 times        → vectors should be far
```

Builds a co-occurrence matrix and factorises it mathematically into embeddings.

### FastText (Facebook, 2016)

Like Word2Vec but also breaks words into subwords:

```
"running" = "run" + "ing"
"runner"  = "run" + "ner"
```

Can handle words it has never seen before by combining subword embeddings.

---

## Are Embeddings Deterministic?

This is an important distinction between two phases:

```
PHASE 1 — TRAINING (not deterministic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Random weight initialisation
- Shuffled data order
- Run training twice → slightly different final weights each time
- The exact numbers differ but the relationships are stable
  (dog and cat will always end up close regardless)

PHASE 2 — INFERENCE (fully deterministic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Weights are frozen (saved to a file after training)
- Same input → same output. Always.
- send "dog" → always get [0.2, 0.8, -0.3, 0.5, ...]
- send "dog" → always get [0.2, 0.8, -0.3, 0.5, ...]
```

When you use BERT or Jina at work, you are always in **Phase 2**. The model was trained once. Weights are frozen. Same text in = same vector out. Every time.

---

## Real World Usage — Lexis Nexis / Snowflake Example

```
Text: "The court dismissed the appeal"
         ↓
  Jina / BERT embedding model
  (fixed frozen weights — Phase 2)
         ↓
  [0.23, -0.41, 0.87, 0.12, ...]   ← always identical for same input
         ↓
  Store in Snowflake vector database
         ↓
  Query: "find similar legal documents"
         ↓
  Cosine similarity search
         ↓
  Returns nearest vectors (most semantically similar documents)
```

The non-determinism only matters to the team who originally **trained** the model (Google for BERT, Jina AI for Jina). You are always consuming the frozen result of that training.

---

## When to Use Which

| | Characters | Words |
|---|---|---|
| One-hot | ✅ Fine (27-100 tokens) | ❌ Too large (50,000+ tokens) |
| Embeddings | Overkill | ✅ Essential |

| Embedding Type | Who Made It | Dimensions | Best For |
|---|---|---|---|
| Random init (trained with model) | You | Your choice | Small datasets, custom vocab |
| Word2Vec | Google (2013) | 300 | General NLP |
| GloVe | Stanford (2014) | 50 / 100 / 300 | General NLP |
| FastText | Facebook (2016) | 300 | Handles unknown words |
| BERT | Google (2018) | 768 | Deep contextual meaning |
| Jina | Jina AI | 768+ | Semantic search, RAG |

---

## Key Rules to Remember

```
Embedding matrix E:
  Size     =  vocab_size × embedding_size   (e.g. 50,000 × 300)
  One row per word in vocabulary
  Lookup   =  just pick the row for that word
  Trainable = yes (either from scratch or fine-tuned)

After embedding:
  xₜ goes from  50,000 × 1   →   300 × 1
  Wₓ goes from  256 × 50,000 →   256 × 300   (much smaller!)

Inference is always deterministic.
Training is not (but the relationships learned are stable).
```

---

## Key Takeaway

> One-hot encoding for words is wasteful and meaningless. Embeddings compress each word into a small dense vector that captures meaning through geometry. They are either learned during training (just another weight matrix updated by backprop) or downloaded as pre-trained models (Word2Vec, GloVe, BERT, Jina). When you use them at inference time they are completely deterministic — same text always gives the same vector.
