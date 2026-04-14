# Phase 2: Repository Integration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-04-14
**Phase:** 02-repository-integration
**Areas discussed:** SFINAE modernization

---

## SFINAE Modernization

| Option | Description | Selected |
|--------|-------------|----------|
| if constexpr (Recommended) | Collapse two std::enable_if overloads into one function with if constexpr branch. Cleaner, better error messages, idiomatic C++20. | Yes |
| Keep std::enable_if | Minimal diff -- just replace synchronized_data_batch with data_batch in existing enable_if conditions. Respects existing codebase pattern. | |
| requires clause | C++20 concepts: use requires std::copy_constructible<T> on the overloads. Modern but keeps two separate functions. | |

**User's choice:** if constexpr with `std::is_copy_constructible_v<T>`
**Notes:** User preferred the single-function-body approach. Condition is decoupled from specific type name, which is more robust.

---

## Claude's Discretion

- Include/forward-declaration cleanup
- Whether get_data_batch_by_id specializations should also use if constexpr

## Deferred Ideas

None
