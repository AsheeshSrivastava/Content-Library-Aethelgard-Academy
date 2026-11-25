# Stage 3 of the comparison: hybrid search and the local reranker (spec 15). Run from the rag-core folder AFTER stages 1
# and 2 (it reads the stage 1 runs of the two chosen cells, it builds no run, and it costs no money):
#   .\scripts\stage3.ps1 estimate
#   .\scripts\stage3.ps1 run
#   .\scripts\stage3.ps1 report
# The cells are jina-v2-base-en:llamaindex and bge-m3:merge-small. Change them only with -Cells, e.g.
#   .\scripts\stage3.ps1 estimate -Cells bge-m3:merge-small
# `run` can be stopped and started again: variants already saved are not scored again.
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
           '--exclusions', 'configs\corpus-exclusions.yaml', '--cells', $Cells)
if ($Only) { $args2 += @('--only', $Only) }
uv run --group embed --group variants python scripts/run_stage3.py $Step @args2
