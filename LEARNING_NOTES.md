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

### 2026-10-07 — Local inference, prefill, context, and KV-cache reuse

**Concept**

- An LLM inference request has two major phases: **prefill**, which processes the input/context tokens, and **decode**, which generates output tokens autoregressively.
- The inference runtime owns tokenization, model execution, and the KV cache. The agent runtime sits above it and owns higher-level concerns such as context construction, tool use, state, and the agent loop.
- Prompt text is still sent on later requests, but when a long prefix is unchanged the inference runtime can reuse previously computed KV state for that prefix instead of recomputing all of its prefill work.
- The configured context window is a runtime constraint. Larger context also increases runtime memory requirements.

**Evidence / experiment**

- With Qwen2.5-Coder 14B on the local Mac, decode throughput stayed roughly stable at about 13–14 output tokens/s while prefill time varied substantially with prompt length and cache reuse.
- In the controlled prefix-cache experiment, prompts A and C had similarly sized but different long prefixes and each required about 11–12 s of fresh prefill. Prompt B reused A's long prefix and required only about 0.33 s of prefill, directly demonstrating prefix/KV-cache reuse.
- Increasing Ollama's configured context from 4096 to 8192 increased the loaded model footprint and changed the observed oversized-prompt truncation point from about 2050 to about 4098 input tokens.
- An unloaded model does not appear in `ollama ps`; downloaded model weights, a loaded model runner, and per-prompt KV state are separate concepts.

**Why it matters**

For an agent, repeatedly rebuilding and sending context is unavoidable at the application level, but inference-runtime prefix caching can make repeated long prefixes much cheaper. This affects agent prompt design, context ordering, latency, and cost.

**Open questions**

- How does tokenization differ across natural language, code, and non-English text?
- How do context length and KV-cache size affect memory in more detail?
- Which cache behaviors are Ollama-specific versus inherited from llama.cpp?


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

