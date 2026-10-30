# Draw the random quote candidates for the passage-level questions (spec 16 section 3). Run from the rag-core folder:
#   .\scripts\pick_quotes.ps1
# The same seed always gives the same file. It reads the corpus (with the exclusions) and writes
# eval\passage-candidates\candidates.yaml. It uses no model and costs no money.
param(
  [int]$Count = 400,
  [int]$Seed = 20261030,
  [int]$Cap = 25
)
$lib = 'D:\01-Project\unsloth\aethelgard-academy\.worktrees\library3\content\library'
$snaps = 'C:\Users\sashe\.aethelgard\library\snapshots'
uv run python scripts/run_pick_quotes.py --register "$lib\sources" --snapshots $snaps `
  --exclusions configs\corpus-exclusions.yaml --count $Count --seed $Seed --cap $Cap `
  --out eval\passage-candidates\candidates.yaml
