# REVIEW.md
Wrong in this project means (mark these Important):
1. A secret is written, logged, printed or committed.
2. A check or test is skipped, weakened or faked to get green.
3. Data can be lost or corrupted (writes, migrations, deletes).
4. Untrusted input reaches files, the shell, SQL or a model without checks.
5. A paid API call can run without the mock mode or a cost limit.
6. Code or tests changed with no ADR or spec named for the change.
Skip: uv.lock, .venv/, generated files.
At most 3 nits, marked Nit. Every finding cites file:line and says why it fails.
Start the summary with a count. If nothing is wrong, say so.
