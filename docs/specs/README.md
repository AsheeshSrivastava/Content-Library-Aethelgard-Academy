# Specs

A spec says what a unit must do, in testable terms, before any code for it is written.

## The order for every unit
1. Decide (owner, in the thread).
2. ADR, if the decision is hard to reverse and had a real alternative. Status Proposed until the owner accepts.
3. Spec: purpose, data, rules, failure behaviour, acceptance tests. Status Draft until the owner accepts.
4. Tests from the acceptance list, written first and seen failing.
5. Code, then self-review and the PROGRESS.md entry.

If code or a spec changes the decision, stop and amend the ADR or spec first. A spec written after its code must say so.

| # | Spec | Unit | Status |
|---|---|---|---|
| [01](01-contracts.md) | Contracts | 1 | Draft, written after the code (unit 1 had only ADR 0003) |
| [02](02-register-and-gate.md) | Source register and licence gate | 2 | Accepted 2026-10-29 |
| [03](03-pinned-fetch.md) | Pinned fetch | 2b | Accepted 2026-10-29 |
| [04](04-licence-detection.md) | Licence detection (cross-check) | 2c | Accepted 2026-10-29 |
