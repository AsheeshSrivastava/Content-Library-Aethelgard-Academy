# Spec 14: Corpus exclusions

Status: Accepted 2026-10-30, follows ADR 0008 (Accepted). Written before the code.

## 1. Purpose
Leave reviewed noise out of the searchable corpus without touching the register, the snapshots or the licence gate.

## 2. The file
`configs/corpus-exclusions.yaml`, a mapping of source id to an ordered list of rules:
```
python-docs:
  - exclude: "Doc/library/*"
    reason: "Most standard library modules are not taught in the course"
  - include: "Doc/library/sys.rst"
```
- A rule has exactly one of `exclude` or `include` (a glob) and, for `exclude`, a non-empty `reason`. `include` may carry a `reason`.
- The glob is matched against the path inside the snapshot, in posix form, without the source id. `*` matches any characters including `/`, `?` matches one character, `[abc]` one of a set. No leading `/`, no `..`, not empty.
- The last matching rule wins. A file that matches no rule stays. `include` is how an exclusion gets exceptions.
- Unknown keys, wrong types, an empty list or a source id that is not text are refused with a message that names the source and the rule position.

## 3. Applying it
`apply_exclusions(sources, exclusions)` returns the kept sources in their original order, the excluded files as `(source_id, reason)`, and the rules that matched no file as `(source id, glob)`. Sources of a source id with no rules are untouched. The relative path of a source is its `source_id` after the first `/`.

## 4. Where it is used
- `load_corpus(..., exclusions=...)` (stage runner): applies it after the snapshot loader, and adds notes: the number excluded per source with the first reason, and one note per rule that matched nothing.
- `question_check` takes `--exclusions` and applies it, so a question whose file was excluded is reported as matching no source.
- The stage runner takes `--exclusions PATH` (default: `configs/corpus-exclusions.yaml` when it exists; `none` turns it off).
- The corpus fingerprint covers the remaining sources only, so run ids change when the corpus changes.

## 5. Acceptance tests
Loading: a valid file; each refusal named above; an empty file is an empty mapping. Matching: last rule wins, include restores, default keeps, `*` crosses `/`, `?`, brackets, exact path, other sources unaffected. Applying: order kept, excluded list with reasons, unused rules found, no rules gives all sources. Integration: `load_corpus` notes, question check reports the excluded file as unmatched, the stage run id changes when a rule is added. The shipped `configs/corpus-exclusions.yaml` loads and every exclude has a reason.
