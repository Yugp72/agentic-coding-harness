# Agentic AI Coding Harness & Evaluation Engine

A complete experimental harness for running coding agents through **parallel subagent trajectories**, executing generated code in an isolated sandbox, feeding test failures back into repair prompts, and measuring **TTFT**, approximate **token throughput**, **inter-token latency (ITL)**, pass rate, and wall-clock latency.

This repository is intentionally provider-agnostic. It includes live adapters for **OpenAI/Codex-style Responses API models** and **Anthropic/Claude models**, plus a deterministic mock provider so the entire system can be tested without API keys.

## What problem it solves

Coding-agent quality is hard to improve if generation, execution, repair, and evaluation are separate scripts. This project closes that loop:

1. Route a benchmark task into multiple independent agent lanes.
2. Stream each model response while collecting inference telemetry.
3. Execute each candidate against hidden-style tests.
4. Feed failures back into the model for iterative repair.
5. Select a passing trajectory and persist metrics for evaluation.
6. Repeat across many tasks/runs to compare providers, prompts, and concurrency settings.


## Features

- Async parallel subagents using `asyncio.gather`.
- Iterative code generation -> execution -> error feedback -> repair.
- Provider abstraction for OpenAI/Codex, Anthropic/Claude, and mock mode.
- Streaming telemetry: Time to First Token, approximate token throughput, mean ITL.
- Docker sandbox with no network, memory/CPU/PID limits, read-only root filesystem.
- Local subprocess sandbox for trusted CI/demo use.
- YAML/JSON benchmark format with executable tests.
- Concurrent evaluation across many tasks.
- JSONL trajectory output suitable for notebooks, dashboards, or experiment tracking.
- Three included benchmark tasks and automated tests.
- Script for 100+ trajectories.

## Quick start

### 1. Create an environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
```

### 2. Run the no-key demo

```bash
agentic-harness run benchmarks/tasks/fizzbuzz.yaml \
  --provider mock --sandbox local --parallel 3
```

### 3. Evaluate all included tasks

```bash
agentic-harness eval benchmarks/tasks \
  --provider mock --sandbox local --parallel 3 --task-parallelism 3
```

Outputs appear in `runs/latest/results.jsonl` and `runs/latest/summary.json`.

## Run with Docker isolation

Build the minimal sandbox image once:

```bash
docker build -f Dockerfile.sandbox -t agentic-harness-sandbox:latest .
```

Then:

```bash
agentic-harness eval benchmarks/tasks --provider mock --sandbox docker
```

**Use Docker for untrusted model-generated code.** The local runner is deliberately provided only for trusted development and CI fixtures.



## Evaluation metrics

- **Agentic success rate:** percentage of tasks with at least one passing candidate.
- **TTFT:** time from provider request start until first streamed text event.
- **Approx. token throughput:** whitespace-delimited output units / end-to-end streamed generation time.
- **Mean ITL:** mean timing gap between observed output units.
- **Sandbox duration:** execution/test time per attempt.
- **Trajectory count:** every candidate/repair attempt is preserved in the results.

The profiler intentionally labels its token metrics as approximate because provider SDKs expose streaming granularity differently. For strict tokenizer-level throughput, add a tokenizer adapter per model family.

## Why parallel subagents?

A single coding trajectory can get stuck on an early design choice. Independent lanes create diversity at the prompt level, while the harness applies the same tests to every candidate. That makes it possible to study the quality/latency/compute tradeoff as parallelism changes.

## Extend the project

Useful next additions:

- Git repository tasks: clone a fixture repo, patch multiple files, run repo-native tests.
- Tool-call layer: expose read/write/search/test tools instead of asking for a whole file.
- Real tokenizer instrumentation for exact output-token throughput.
- OpenTelemetry exporter and Prometheus metrics.
- LLM-as-judge only for subjective tasks; keep executable tests as the primary signal.
- Failure taxonomy (syntax, timeout, assertion, import, flaky test, infrastructure error).
- Adaptive routing: send easy tasks to a cheaper model and escalate failed tasks.
- Speculative parallelism: cancel other lanes as soon as one candidate passes.

## Test

```bash
pytest -q
```

## Safety note

Generated code is arbitrary code. The Docker runner removes network access and applies basic resource limits, but it is not a substitute for a hardened production sandbox such as gVisor, Firecracker, or an isolated remote execution service.
# agentic-coding-harness
# agentic-coding-harness
