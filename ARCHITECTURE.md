# ForgeAgent Architecture

This document tracks the architecture as ForgeAgent evolves.

The goal is to keep the design grounded in what has actually been implemented. Avoid documenting speculative production architecture too early.

## Current stage

**Layer 0 — Local LLM fundamentals**

There is not yet an agent runtime.

Current focus:

```text
Python client
    ↓
local inference runtime
    ↓
local model
```

Initial concerns:

- model invocation;
- streaming;
- token accounting;
- latency measurement;
- structured output.

## Architectural principles

1. **Build the important abstractions directly before adopting an agent framework.**
2. **Keep model responsibilities separate from runtime responsibilities.**
3. **Prefer small, measurable components over speculative complexity.**
4. **Add distributed-system machinery only when a concrete requirement justifies it.**
5. **Treat evaluation as part of the architecture, not as an afterthought.**

## Expected evolution

The architecture is expected to grow roughly along this path:

```text
local model client
      ↓
minimal agent loop
      ↓
tool registry + execution
      ↓
isolated coding workspace
      ↓
code retrieval/index
      ↓
planning/test/revision loop
      ↓
eval harness
      ↓
repo-specific memory/skills
      ↓
multi-user distributed service
```

## Model vs runtime

A core principle for later stages:

> The model proposes actions; the runtime validates, enforces, executes, persists, and recovers them.

The runtime will eventually own concerns such as:

- tool registration;
- schema validation;
- authN/authZ;
- sandbox access;
- retries/timeouts;
- persistence;
- observability;
- resource limits.

## Decisions log

Record significant architecture decisions here as they become real.

| Decision | Status | Rationale |
|---|---|---|
| Start with local inference | Planned | Reduce iteration cost and expose lower-level model/runtime behavior |
| Build the initial agent loop without LangGraph/Strands | Planned | Learn the underlying abstractions before delegating them to a framework |
| Keep evals as a first-class subsystem | Planned | Enable empirical improvement rather than prompt changes by intuition |

