# ForgeAgent Roadmap

ForgeAgent is a coding-agent systems laboratory built from first principles.

The goal is not only to build a useful coding agent, but to use the project as a structured path for learning modern AI systems deeply enough to reason about them in interviews, design reviews, and production work.

The roadmap follows one core principle:

> **Understand → implement minimally from scratch → measure → replace with a library/framework → compare abstractions.**

The project should remain useful at every stage. Each layer should produce something concrete that can be demonstrated, measured, and discussed in an interview.

---

## Guiding goals

ForgeAgent is intended to optimize for four things simultaneously:

1. **AI systems depth** — understand the concepts underneath agent frameworks rather than only wiring libraries together.
2. **Hands-on portfolio value** — build a serious public project that demonstrates engineering judgment and progressive system design.
3. **Interview readiness** — cover the concepts most likely to appear in AI infrastructure, retrieval, agentic systems, and system design interviews.
4. **Knowledge retention** — reinforce learning through implementation, active recall, and periodic coding/system-design practice.

---

## Learning strategy

For each layer:

- learn the minimum theory required to reason about the problem;
- implement the important abstractions directly at least once;
- instrument and evaluate behavior;
- only then introduce a framework or managed component;
- compare what the framework abstracts away and what control is lost or gained.

We should prefer small, measurable milestones over large speculative designs.

---

# Layer 0 — Local LLM fundamentals

## Goal

Run an LLM locally, understand the basic mechanics of inference, and build the first direct Python interaction with a model.

## Theory

- tokens and tokenization;
- transformer intuition;
- attention at a conceptual level;
- context windows;
- inference vs training;
- temperature and top-p;
- structured output;
- streaming;
- KV cache;
- quantization;
- embeddings vs generative models;
- local vs remote inference tradeoffs.

## Build from scratch

- small Python client around a local model endpoint/runtime;
- prompt construction;
- token accounting;
- latency measurement;
- simple structured-output parsing;
- experiment harness for model parameters.

## Use existing components

Use an existing local inference runtime rather than implementing model kernels.

Initial candidates:

- Ollama for convenience;
- MLX / MLX-LM for Apple-Silicon-native experimentation;
- llama.cpp as another useful reference point.

## Initial hardware target

Development machine:

- MacBook Pro M5
- 24 GB unified memory

Practical early target:

- 7B–14B quantized instruction/coding models;
- keep enough memory available for IDE, browser, tests, and normal development tools.

## Milestone

**v0.1 — Local model playground**

A CLI or script that:

- sends prompts to a local model;
- streams output;
- records latency and token counts;
- supports structured JSON output;
- allows easy switching between a few generation parameters.

## Exit criteria

Be able to explain:

- what happens from prompt submission to generated tokens;
- why quantization matters;
- what consumes context;
- why KV caching improves autoregressive decoding;
- when local inference is useful versus a hosted model.

## Interview relevance

- LLM fundamentals;
- inference/runtime tradeoffs;
- context-window reasoning;
- local-vs-cloud model decisions.

---

# Layer 1 — Minimal agent loop

## Goal

Build an agent runtime directly, without LangGraph, Strands, or another agent framework.

## Core loop

```text
model
  ↓
structured tool request
  ↓
validate
  ↓
execute tool
  ↓
append result to context
  ↓
model
  ↓
...
```

## Theory

- tool calling;
- structured outputs;
- ReAct-style loops;
- stopping conditions;
- context accumulation;
- model vs runtime responsibilities;
- schema validation;
- retries and timeouts;
- bounded autonomy.

## Build from scratch

- agent loop;
- tool registry;
- JSON-schema-like tool definitions;
- argument validation;
- tool dispatch;
- max-iteration limits;
- token budget;
- tool timeout/error handling;
- trace/log for every model and tool step.

## First tools

- `read_file`
- `search_text`
- `run_tests`

## Milestone

**v0.2 — Minimal tool-using agent**

Give the agent a small local repository and a deliberately simple bug. The agent should:

1. inspect the task;
2. search the repository;
3. read relevant files;
4. run tests;
5. explain a likely fix.

Editing can wait until the next layer.

## Exit criteria

Be able to explain:

> The model proposes actions; the runtime validates, enforces, executes, and records them.

Also be able to explain why:

- authZ cannot be delegated to the model;
- tool schemas matter;
- loops require explicit bounds;
- tool errors belong in the agent observation stream.

## Interview relevance

- agent runtime fundamentals;
- tool execution;
- context management;
- auth and policy boundaries;
- long-running agent loops.

---

# Layer 2 — Coding tools and isolated workspace

## Goal

Turn the tool-using agent into a real coding agent that can modify code, test it, and produce a patch.

## Theory

- sandboxing;
- process isolation;
- file-system isolation;
- untrusted code execution;
- resource limits;
- workspace lifecycle;
- Git mechanics relevant to agents.

## Build from scratch

Add tools such as:

- `list_files`
- `search_text`
- `read_file`
- `apply_patch`
- `run_command`
- `run_tests`
- `git_diff`

Initially run inside a temporary local workspace.

Then introduce stronger isolation.

## Sandbox evolution

```text
repository
   ↓
base workspace
   ↓
task-specific isolated workspace
   ↓
agent edits
   ↓
tests
   ↓
diff
   ↓
cleanup
```

Explore:

- full clone;
- shallow clone;
- persistent Git mirror;
- copy-on-write snapshot;
- containerized task workspace;
- warm sandbox pool.

## Milestone

**v0.3 — Issue to patch**

Given a local task description, ForgeAgent should:

1. inspect the repo;
2. locate likely relevant code;
3. edit files;
4. run tests;
5. iterate if tests fail;
6. emit a final diff and explanation.

## Exit criteria

Be able to reason about:

- why a coding agent needs an isolated workspace;
- sandbox lifecycle;
- cleanup;
- command execution safety;
- warm vs cold workspaces;
- what state should persist across tasks.

## Interview relevance

Directly covers the coding-agent sandbox questions that frequently appear in modern AI system-design interviews.

---

# Layer 3 — Code retrieval

## Goal

Make repository navigation efficient and measurable.

This layer intentionally connects modern coding agents with retrieval/search systems.

## Version A — lexical retrieval

Implement:

- filename search;
- regex/grep;
- inverted index;
- BM25 ranking.

Understand the distinction:

> inverted index retrieves candidates; BM25 scores/ranks lexical matches.

## Version B — structural retrieval

Parse code into meaningful units:

- files;
- classes;
- functions;
- methods;
- symbols;
- imports/references.

Explore AST-based chunking and symbol graphs.

## Version C — semantic retrieval

```text
code unit
  ↓
embedding
  ↓
vector index
  ↓
semantic query
```

Store provenance:

- repository;
- commit;
- path;
- symbol;
- start/end line;
- language.

## Version D — hybrid retrieval

Combine:

- BM25;
- semantic search;
- symbol relationships.

Explore:

- weighted score fusion;
- normalization;
- calibration;
- reciprocal rank fusion (RRF);
- reranking.

## Milestone

**v0.4 — Hybrid repo search**

The agent can use a repository index to identify relevant code without scanning the entire codebase.

## Evaluation

Create retrieval-specific tests:

- known issue → expected files/symbols;
- Recall@K for relevant files;
- latency;
- index-build time;
- index freshness.

## Exit criteria

Be able to explain:

- BM25 vs embeddings;
- structural vs semantic retrieval;
- hybrid search;
- score compatibility;
- normalization vs calibration;
- when RRF is preferable.

## Interview relevance

- retrieval architecture;
- RAG;
- semantic search;
- code indexing;
- ranking/fusion;
- Pinecone/OpenSearch/vector-search style discussions.

---

# Layer 4 — Planning and execution

## Goal

Move from reactive tool use to a deliberate coding workflow.

## Candidate execution loop

```text
understand issue
      ↓
investigate
      ↓
form hypothesis
      ↓
retrieve code
      ↓
plan
      ↓
edit
      ↓
test
      ↓
inspect failure
      ↓
revise or finish
```

## Theory

- ReAct;
- planning vs acting;
- reflection;
- decomposition;
- stopping criteria;
- bounded search;
- tool-selection policy;
- planner/executor separation;
- when multi-agent designs help and when they add noise.

## Build

Start with one model and explicit phases.

Avoid adding multiple agents until a measured limitation justifies them.

Possible state:

- task understanding;
- current hypothesis;
- relevant files;
- planned edits;
- test results;
- unresolved questions.

## Milestone

**v0.5 — Plan, edit, test, revise**

The agent should produce an explicit plan, execute it, inspect test results, and revise when needed.

## Exit criteria

Be able to explain:

- why planning may help;
- why planning may also create stale commitments;
- how observations update the plan;
- when reflection adds value;
- how to bound loops.

## Interview relevance

- agent orchestration;
- planner/executor patterns;
- failure recovery;
- multi-step reasoning systems.

---

# Layer 5 — Evaluation system

## Goal

Build a real eval-driven development loop.

This is a first-class subsystem, not an afterthought.

## Theory

- offline evaluation;
- golden datasets;
- task-level success;
- deterministic tests;
- LLM-as-judge;
- pairwise comparison;
- pass@k;
- regression suites;
- human evaluation;
- failure taxonomy;
- online vs offline evals.

## Dataset

Start with tasks from one or more small repositories.

Each task should capture:

```yaml
repo_commit:
issue:
acceptance_criteria:
tests:
reference_patch: optional
metadata:
```

Potential sources:

- deliberately planted bugs;
- historical GitHub issues;
- previous commits with known fixes;
- synthetic tasks that test specific agent abilities.

## Metrics

Track at minimum:

- task success rate;
- tests passed;
- regressions introduced;
- model calls;
- tool calls;
- tokens;
- wall-clock time;
- patch size;
- retrieval quality;
- human quality score where useful.

## Failure taxonomy

Examples:

- retrieval failure;
- task-understanding failure;
- reasoning/planning failure;
- incorrect edit;
- tool failure;
- context loss;
- test misunderstanding;
- premature completion;
- unnecessary code churn.

## Experiment loop

```text
benchmark
   ↓
agent version
   ↓
run
   ↓
metrics + failure classification
   ↓
change prompt / tools / retrieval / model
   ↓
rerun
   ↓
regression comparison
```

## Milestone

**v0.6 — Eval harness**

A repeatable command should benchmark multiple agent configurations and produce comparable results.

## Exit criteria

Be able to answer concretely:

> “How would you improve an agent from 30% task success to 70%?”

with an eval-driven plan rather than intuition alone.

## Interview relevance

- agent evaluation;
- prompt/tool optimization;
- regression testing;
- empirical AI engineering.

---

# Layer 6 — Repo-specific memory and skills

## Goal

Allow ForgeAgent to become more effective on repositories it has worked with repeatedly.

## Theory

- agent memory;
- episodic vs semantic memory;
- persistent instructions;
- skill acquisition;
- provenance;
- confidence;
- staleness and invalidation;
- memory poisoning.

## Possible repository memory

```text
.agent/
  architecture.md
  build.md
  testing.md
  conventions.md
  symbols.json
  learned_patterns.md
```

Examples of useful learned facts:

- canonical build command;
- test commands by subsystem;
- architecture entry points;
- common ownership boundaries;
- generated helper scripts;
- reliable navigation shortcuts.

## Important constraint

Do not let the model blindly persist arbitrary “memories.”

Persistent knowledge should have:

- provenance;
- repository version/commit;
- validation status;
- confidence;
- invalidation strategy.

## Milestone

**v0.7 — Repo-adaptive agent**

Measure whether repeated work on the same repository becomes:

- faster;
- cheaper;
- more accurate;
- fewer tool calls.

## Exit criteria

Be able to explain:

- persistent repo context;
- agent skills;
- why memory must be versioned and validated;
- how self-improvement differs from model training.

## Interview relevance

Directly covers self-improving/self-optimizing coding-agent designs.

---

# Layer 7 — Production and distributed agent architecture

## Goal

Scale the working single-agent system into a production-style multi-user service.

This is where existing distributed-systems knowledge becomes the foundation.

## High-level architecture

```text
Task API
   ↓
authoritative task state
   ↓
scheduler
   ↓
sandbox / worker pool
   ↓
agent runtime
   ↓
tool execution
   ↓
repo/index/memory
```

## Topics

- async task APIs;
- authoritative state;
- durable workflows;
- retries;
- idempotency;
- leases and fencing;
- cancellation;
- per-tenant concurrency;
- quotas/rate limits;
- fair scheduling / DRR;
- backpressure;
- warm sandbox pools;
- autoscaling;
- observability;
- cost controls;
- security;
- secret isolation.

## Important distinctions

- queue = delivery/work notification, not necessarily authoritative state;
- lease/fencing protects internal state;
- idempotency protects logical effects under retries;
- external irreversible side effects require stronger handling;
- user task lifecycle and internal tool lifecycle are independent async boundaries.

## Milestone

**v0.8 — Multi-user ForgeAgent service**

Expose task submission and status APIs, schedule work across isolated sandboxes, and persist execution state durably.

## Exit criteria

Be able to design the system at both:

- one-task coding-agent level;
- fleet/multi-tenant production level.

## Interview relevance

- distributed systems;
- AI infrastructure;
- multi-tenant scheduling;
- durability;
- agent runtime design;
- system design.

---

# Layer 8 — Framework comparison

## Goal

After building the important abstractions directly, rebuild selected pieces using established frameworks.

The purpose is not merely to adopt them. It is to understand exactly what they abstract away.

## Candidate frameworks

- LangGraph;
- Strands Agents;
- OpenAI Agents SDK;
- durable workflow engines such as Temporal or Step Functions where appropriate;
- mature coding-agent projects such as OpenHands, SWE-agent, Aider, Goose, or other relevant systems.

## Comparison questions

For each framework:

- who owns the agent loop?
- who owns state?
- how are tools registered?
- how are schemas validated?
- how are checkpoints handled?
- how are async tools represented?
- how is observability exposed?
- how are retries controlled?
- how easy is custom scheduling?
- what assumptions are opinionated?

## Example comparison table

| Concern | ForgeAgent | LangGraph | Strands | Other |
|---|---|---|---|---|
| Agent loop | custom | framework | framework | TBD |
| Tool registry | custom | framework | framework | TBD |
| State | custom | graph state | runtime-specific | TBD |
| Checkpointing | custom | built-in options | varies | TBD |
| Sandbox | custom | external | external | TBD |
| Evals | custom | external/integration | external/integration | TBD |

## Milestone

**v0.9 — Framework-backed variants**

Implement at least one comparable workflow using a mature agent framework and document the tradeoffs.

## Exit criteria

Be able to discuss frameworks from experience rather than from documentation alone.

---

# Continuous track — Coding and system-design maintenance

AI learning should not replace core interview skills.

Use interleaving.

Suggested weekly balance:

- **70%** ForgeAgent + AI theory;
- **20%** coding and system-design maintenance;
- **10%** active recall/review.

Example:

| Day | Primary | Maintenance |
|---|---|---|
| Monday | AI theory + ForgeAgent | — |
| Tuesday | ForgeAgent | coding problem |
| Wednesday | AI theory + ForgeAgent | — |
| Thursday | ForgeAgent | system-design exercise |
| Friday | evals/review | coding problem |
| Weekend | optional/light review | rest |

Coding/system-design sessions should remain short enough that they preserve skill without derailing the main project.

---

# Continuous track — Active recall

Because recognition is not enough, each week should include closed-book retrieval.

Example questions:

- Explain the agent loop without looking at the code.
- What belongs to the model versus the runtime?
- Why does a coding agent need a sandbox?
- How does BM25 differ from semantic retrieval?
- When would you use RRF?
- What does the eval dataset prove?
- How do you prevent old repo memories from becoming stale?
- What happens if an external side effect succeeds but the worker crashes before checkpointing?
- When does a warm workspace make sense?
- What would you change if one tenant became 100× larger than the rest?

---

# Portfolio principles

ForgeAgent should remain understandable to someone arriving from GitHub.

For each major milestone:

- maintain a concise architecture diagram;
- document the design decision and alternatives;
- record measurable results;
- keep small runnable examples;
- include failure cases and lessons;
- avoid hiding the core logic behind frameworks too early.

The project should tell a coherent engineering story:

> “I built a coding agent from first principles, measured where it failed, added retrieval/evals/memory/sandboxing, then compared my implementation against mature frameworks and scaled the architecture toward production.”

---

# Initial milestone plan

## v0.1 — Local model playground

Start here.

Deliverables:

- local model running on the MacBook Pro M5 / 24 GB;
- Python environment;
- programmatic model invocation;
- streaming output;
- basic latency/token metrics;
- structured output experiment.

## v0.2 — Minimal agent loop

Deliverables:

- tool registry;
- `read_file`;
- `search_text`;
- `run_tests`;
- bounded model/tool loop;
- trace of model/tool interactions.

## v0.3 — Issue to patch

Deliverables:

- isolated task workspace;
- edit/apply-patch tool;
- test-and-revise loop;
- final Git diff.

The later milestones should remain intentionally adjustable based on what we learn.

---

# Roadmap status

| Layer | Status |
|---|---|
| 0 — Local LLM fundamentals | **Next** |
| 1 — Minimal agent loop | Planned |
| 2 — Coding tools + sandbox | Planned |
| 3 — Code retrieval | Planned |
| 4 — Planning + execution | Planned |
| 5 — Evaluation system | Planned |
| 6 — Repo-specific memory/skills | Planned |
| 7 — Distributed/production architecture | Planned |
| 8 — Framework comparison | Planned |

---

# Next action

Start **Layer 0** by selecting a local inference stack and a small coding/instruction model suitable for the M5 MacBook Pro with 24 GB unified memory.

The first implementation goal is intentionally small:

> Run a local model from Python, stream a response, capture token/latency metrics, and produce one structured JSON response.
