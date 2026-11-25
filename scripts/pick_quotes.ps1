# Draw the random quote candidates for the passage-level questions (spec 16 section 3). Run from the rag-core folder:
#   .\scripts\pick_quotes.ps1
# The same seed always gives the same file. It reads the corpus (with the exclusions) and writes
# eval\passage-candidates\candidates.yaml. It uses no model and costs no money.
param(
  [int]$Count = 400,
  [int]$Seed = 20261030,
  [int]$Cap = 25
)
$lib = if ($env:RAGCORE_LIB3) { $env:RAGCORE_LIB3 } else { Join-Path (Split-Path $PSScriptRoot -Parent) 'register' }
if (-not (Test-Path -LiteralPath "$lib\sources") -or -not (Test-Path -LiteralPath "$lib\licence-rules.yaml")) {
  Write-Error "Register folder not usable: '$lib' needs a sources folder and licence-rules.yaml. Set RAGCORE_LIB3 or restore the repo's register folder."; exit 1
}
$snaps = if ($env:RAGCORE_SNAPSHOTS) { $env:RAGCORE_SNAPSHOTS } else { 'C:\Users\sashe\.aethelgard\library\snapshots' }
uv run python scripts/run_pick_quotes.py --register "$lib\sources" --snapshots $snaps `
  --exclusions configs\corpus-exclusions.yaml --count $Count --seed $Seed --cap $Cap `
  --out eval\passage-candidates\candidates.yaml
