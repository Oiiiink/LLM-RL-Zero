# AGENTS.md

## Scope and purpose

This file adds repo-local rules to `../AGENTS.md`; the parent rules and `../PLAN.md` remain controlling.

This repository is Xu's minimal learning toy for independently implementing LLM RL mechanics on the existing custom Qwen + Countdown scaffold. It is the selected LLM-shaped artifact for the parent plan's independent PPO+GAE and scratch GRPO gates. Optimize for mathematical visibility, correctness, and Xu's understanding—not extensibility, production hardening, or framework quality.

## Default mode: read-only

Begin every task read-only.

A write is allowed only when Xu's current request explicitly names both the action and every exact file or path that may change. Permission is request-scoped and path-scoped, expires when that request ends, and never extends to additional files by inference.

Without that permission:

- inspect and discuss only;
- do not modify files, git state, dependencies, environments, caches, logs, checkpoints, or generated artifacts;
- do not run commands expected to create or update state.

If the work requires another path, stop before writing and ask. Local permission never includes `../PLAN.md`, `../AGENTS.md`, or `../llm_wiki/`; request every required parent path explicitly before starting.

## Teaching mode

Use an attempt-first Socratic loop:

1. ask Xu for his derivation, tensor mapping, or code attempt;
2. diagnose it directly;
3. give the smallest hint that unblocks the next step;
4. provide a complete implementation only when Xu explicitly requests one.

Agents may inspect, explain, quiz, and review while read-only. Keep formulas, policy roles, tensor shapes, masks, and indexing explicit.

## Code shape

Keep the existing custom Qwen model, Countdown task, and current dependencies. Prefer flat, lowercase files such as `reinforce.py`, `ppo.py`, and `grpo.py`, with small explicit entrypoints.

Share only real plumbing: rollout generation, action masks, token log-probability extraction, and existing model/task utilities. Keep each algorithm's estimator, objective, and update logic visible in its own file. Avoid generic trainers, model/task adapter systems, class hierarchies, distributed infrastructure, and speculative extensibility.

## Verification and progression

Required deterministic tensor checks:

1. GAE with terminal handling;
2. action-mask and token-log-probability alignment;
3. PPO clipping for both positive and negative advantages;
4. GRPO group normalization, including all-equal rewards.

Before advancing to the next algorithm:

- Xu can explain the estimator and map each formula term to tensors;
- the relevant deterministic check passes;
- the implementation has a runnable command;
- PPO and GRPO have short recorded reward/completion evidence.

Benchmark-quality results, broad test coverage, CI, and production hardening are outside this toy's completion criteria.
