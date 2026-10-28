# Problem statement

## Problem
The Aethelgard content factory writes lessons in our own words from trusted sources. Finding the right source passage for a topic by hand does not scale, and one poor or unlicensed document can damage everything built on it.

## What the library must do
1. Take only approved, licence-checked sources, pinned to an exact version.
2. Turn each file into clean text, with a report of what was kept, skipped and flagged.
3. Cut the text into passages that can be cited word for word.
4. Find the right passage for a question and say honestly when nothing fits.
5. Prove every change with a fixed question set before it is kept.

## Not in scope now
Answer writing, the AXIS chatbot, the lesson front end, cloud hosting, paid services in the default path.

## Targets (proposed; owner to confirm)
Starting point to beat, from library3 on the same 601 questions: right lesson first 488 and in the top 3 540 (paid reranker), 428 and 517 (free local bge). Targets for rag-core are set after the embedder and reranker are chosen; no number is promised before then.

## Constraints
Runs on the owner's PC (RTX 5070 Ti 16 GB). Budget is tight: every paid step needs a printed estimate. Content is licence-sensitive; the notice needs qualified review before any commercial release.
