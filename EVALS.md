# ForgeAgent Evaluation

Evaluation will become a first-class ForgeAgent subsystem in Layer 5.

This file starts small and should evolve alongside the implementation.

## Principle

> Do not improve an agent by intuition alone. Define tasks, measure behavior, classify failures, change one thing, and rerun.

## Early-stage evaluation

For Layers 0–2, evaluation can remain lightweight.

Track:

- model latency;
- output validity;
- token usage;
- tool-call count;
- test results;
- wall-clock task time;
- whether the expected file or symbol was located;
- whether the final patch solves the task.

## Future task schema

A benchmark task may eventually look like:

```yaml
id:
repo:
repo_commit:
issue:
acceptance_criteria:
tests:
reference_patch: optional
metadata:
```

## Future metrics

Potential metrics:

- task success rate;
- pass@k;
- tests passed;
- regressions introduced;
- retrieval Recall@K;
- model calls;
- tool calls;
- input/output tokens;
- wall-clock duration;
- patch size;
- human quality score.

## Failure taxonomy

Initial categories:

- task-understanding failure;
- retrieval failure;
- planning/reasoning failure;
- incorrect edit;
- tool execution failure;
- test misunderstanding;
- context loss;
- premature completion;
- unnecessary code churn.

## Experiment discipline

When comparing agent versions:

1. keep the benchmark fixed;
2. record model + prompt + tool configuration;
3. change one major variable where practical;
4. run the same task set;
5. compare success and cost/latency;
6. inspect failure categories, not only aggregate success.

## First real eval milestone

The first meaningful eval suite should appear once ForgeAgent can modify code and run tests.

At that point, start with a small set of deliberately planted bugs with deterministic acceptance tests.

