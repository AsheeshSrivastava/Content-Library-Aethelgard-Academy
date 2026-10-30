import glob
import sys
from collections import Counter

import yaml

root = r"D:\01-Project\rag-core"
cands = yaml.safe_load(open(root + r"\eval\passage-candidates\candidates.yaml", encoding="utf-8"))["candidates"]
picks = []
for f in sorted(glob.glob(root + r"\eval\passage-candidates\drafts\picks-*.tsv")):
    for line in open(f, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        idx, kind, q = line.split("\t", 2)
        picks.append((int(idx), kind, q))
seen = Counter(i for i, _, _ in picks)
dups = [i for i, n in seen.items() if n > 1]
print("picks", len(picks), "dups", dups)
RELABEL = {137: "context", 104: "context", 185: "context", 192: "context", 250: "context", 13: "context"}
REWRITE = {
    30: "What lesson do the authors draw about relying on compact models to produce reasoning explanations?",
    38: "How do deep agents keep working toward goals across long periods instead of improvising step by step?",
    104: "Why is learning from several datasets together usually hard, and what lets the cross-encoder trainer handle it?",
    122: "How many weights does AQLM treat as one group, and what do they get expressed as?",
    127: "How many threads load parameters asynchronously by default, and why was that number chosen?",
    163: "How can servers describe folders among the file:// items, given they have no standard media type?",
    217: "Which version of the tracing API should instrumented packages depend on, and when should they update?",
    275: "Does the sequence in which test setup helpers are written decide the sequence they run in, and what should you rely on?",
    277: "What does the environment test automation helper do, and when would you consider it?",
    285: "A repeat count of six: how many occurrences must match, and what happens with five?",
    298: "What states can a future be in, and what everyday device is it compared to?",
    303: "What does a dictionary view do after its dictionary is modified, and how do you turn it into a full list?",
    305: "How can you reliably tell whether an object can be iterated?",
    307: "How do you prevent attribute names from colliding with those of subclasses?",
    329: "Which field of a tool definition matters most to the model, and why?",
    332: "In which orchestration pattern do specialists never communicate with one another, and who dispatches the work?",
    334: "After how many days open does an issue become eligible for a contributor reward?",
    349: "Which component walks the syntax tree when a new lint rule is added, and what must you change so the violation fires?",
    353: "What happens to the line length set in project settings files when a command-line override is given?",
    362: "From which version can users choose pip-style behaviour for packages found on several indexes, and through which two means?",
}
picks = [(i, RELABEL.get(i, k), REWRITE.get(i, q)) for i, k, q in picks]
DROP = {24, 51, 144, 47, 53, 56, 59, 101, 12}  # review numbers (1-based position in picks order) that failed review
picks = [p for n, p in enumerate(picks, start=1) if n not in DROP]
out = [{"q": q, "source": cands[i]["source"], "quote": cands[i]["quote"], "kind": k} for i, k, q in picks]
print("kinds", dict(Counter(o["kind"] for o in out)))
print("groups", dict(Counter(o["source"].split("/")[0] for o in out)))
if len(sys.argv) > 1:
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        yaml.safe_dump({"questions": out}, fh, allow_unicode=True, sort_keys=False, width=1000)
    print("wrote", sys.argv[1])

