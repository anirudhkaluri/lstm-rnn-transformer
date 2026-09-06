# Chapter 1: What is an RNN and What Problem Does It Solve?

---

## The Problem — Computers Have No Memory

Imagine you are reading this sentence:

> "The dog chased the cat because **it** was scared."

What does **"it"** refer to? The cat. You knew that because you **remembered** the earlier part of the sentence. Your brain keeps context as you read forward.

Now imagine a regular neural network reading the same sentence. It reads one word and **completely forgets it** before reading the next. It has no memory. It is like having amnesia after every single word.

**That is the problem RNNs solve.** They give computers a way to remember what came before.

---

## Where RNNs Are Used

- Autocomplete on your phone (predicts the next word)
- Google Translate (understands a full sentence, not just one word)
- Siri / Alexa (understands speech over time)
- Stock price prediction (patterns over time)
- Music generation

---

## The Core Idea

```
Normal Network:    Input → [Brain] → Output
                   (forgets everything each time)

RNN:               Input → [Brain] → Output
                              ↑   |
                              |___|
                         (passes memory to itself!)
```

The RNN has a **loop** — it feeds its own memory back into itself at every step. That loop is the entire secret.

---

## Key Takeaway

> A regular neural network is like a vending machine — it takes your input, gives an output, and remembers nothing. An RNN is like a conversation — it remembers everything said before and uses it to respond intelligently.
