# ForgeAgent Learning Notes

This file is for concise, durable notes learned while building ForgeAgent.

It should not become a transcript of documentation or a general AI encyclopedia.

Prefer notes that answer:

- What did I learn?
- What misconception did I correct?
- What tradeoff became clearer?
- What would I want to recall in an interview six months from now?

## Format

Use short entries like:

### YYYY-MM-DD — Topic

**Concept**

Concise explanation.

**Why it matters**

Connection to ForgeAgent, production systems, or interviews.

**Evidence / experiment**

What was observed in code, benchmarks, or tests.

**Open question**

Anything still unclear.

---

## Layer 0 topics to capture

As experiments begin, capture concise notes on:

- tokenization;
- context windows;
- temperature / top-p;
- streaming;
- KV cache;
- quantization;
- local inference;
- Ollama;
- MLX / MLX-LM;
- llama.cpp;
- structured outputs;
- embeddings vs generative models.

## Retrieval practice

Periodically answer these without looking:

- What does quantization trade away?
- Why does KV caching improve decoding performance?
- What consumes context-window capacity?
- What is the difference between model inference and an agent runtime?
- When would local inference be preferable to a hosted model?

