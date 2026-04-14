# Phase 1: Core Type System - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-14
**Phase:** 01-core-type-system
**Areas discussed:** Template structure, Accessor methods, Clone API shape

---

## Template Structure

| Option | Description | Selected |
|--------|-------------|----------|
| Class templates | read_only_data_batch<PtrType> and mutable_data_batch<PtrType> are full class templates. Static methods also templated. Definitions in header, explicit instantiations in .cpp. | ✓ |
| data_batch templated | data_batch<PtrType> is the template. Accessor types are inner classes or derived. | |
| Non-template accessors | Accessors store type-erased wrapper internally. Simpler API but runtime overhead. | |

**User's choice:** Class templates
**Notes:** Matches existing idata_repository pattern in the codebase.

---

## Accessor Methods

### get_data() return type

| Option | Description | Selected |
|--------|-------------|----------|
| Raw pointer (Recommended) | idata_representation* — matches existing pattern, null-checkable | ✓ |
| Reference | const idata_representation& — non-nullable, throws if null | |

**User's choice:** Raw pointer
**Notes:** User asked what current implementation returns. Current returns raw pointers throughout.

### Mutability

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, full read+write (Recommended) | mutable has everything read_only has, plus set_data() and convert_to() | ✓ |
| Write-only | mutable only exposes mutation methods | |

**User's choice:** Full read+write

---

## Clone API Shape

### clone() return type

| Option | Description | Selected |
|--------|-------------|----------|
| shared_ptr<data_batch> always | Clone always returns shared_ptr regardless of PtrType | |
| Match PtrType | clone() returns PtrType — consistent with template parameter | ✓ |

**User's choice:** Match PtrType
**Notes:** User asked about deep vs shallow copy. Confirmed: clone() is deep copy (copies all GPU buffers). Also asked about difference between clone and clone_to — clone = same-type copy, clone_to = copy + representation conversion.

### clone_to<T>() placement

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, same pattern | clone_to<T>() on read_only, returns PtrType. Deep copy + conversion. | ✓ |
| You decide | Claude's discretion | |

**User's choice:** Same pattern as clone()

---

## Claude's Discretion

None — user made explicit decisions on all areas.

## Deferred Ideas

None — discussion stayed within phase scope.
