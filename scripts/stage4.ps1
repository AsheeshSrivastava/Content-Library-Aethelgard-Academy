# Stage 4 of the comparison: the passage-level questions (spec 16). Run from the rag-core folder AFTER stage 3, once the
# questions in eval\passage-questions have been written and reviewed (it refuses a set that fails the checks).
# It reads the stage 1 runs of the two cells, builds no run and costs no money:
#   .\scripts\stage4.ps1 estimate
#   .\scripts\stage4.ps1 run
#   .\scripts\stage4.ps1 report
# `run` can be stopped and started again: pipelines already saved are not scored again.
param(
  [Parameter(Mandatory = $true)][ValidateSet('estimate', 'run', 'report')][string]$Step,
  [string[]]$Cells = @('jina-v2-base-en:llamaindex', 'bge-m3:merge-small'),
  [string]$Only = ''
)
$Cells = $Cells -join ','
$lib = if ($env:RAGCORE_LIB3) { $env:RAGCORE_LIB3 } else { Join-Path (Split-Path $PSScriptRoot -Parent) 'register' }
if (-not (Test-Path -LiteralPath "$lib\sources") -or -not (Test-Path -LiteralPath "$lib\licence-rules.yaml")) {
  Write-Error "Register folder not usable: '$lib' needs a sources folder and licence-rules.yaml. Set RAGCORE_LIB3 or restore the repo's register folder."; exit 1
}
$snaps = if ($env:RAGCORE_SNAPSHOTS) { $env:RAGCORE_SNAPSHOTS } else { 'C:\Users\sashe\.aethelgard\library\snapshots' }
$args2 = @('--register', "$lib\sources", '--snapshots', $snaps, '--questions', 'eval\questions',
           '--passage-questions', 'eval\passage-questions',
           '--exclusions', 'configs\corpus-exclusions.yaml', '--cells', $Cells)
if ($Only) { $args2 += @('--only', $Only) }
uv run --group embed --group variants python scripts/run_stage4.py $Step @args2