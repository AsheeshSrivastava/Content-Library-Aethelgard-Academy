# Stage 2 of the chunker and embedder comparison: late chunking (spec 13 section 4.3 and 6). Run from the rag-core folder
# AFTER stage 1 is finished and reported. Choose the 3 best chunkers per embedder from the stage 1 report, then:
#   .\scripts\stage2.ps1 estimate -Embedders bge-m3 -Chunkers rules,merge-small,parent-child
#   .\scripts\stage2.ps1 run      -Embedders bge-m3 -Chunkers rules,merge-small,parent-child
#   .\scripts\stage2.ps1 report   -Embedders bge-m3 -Chunkers rules,merge-small,parent-child
# Do the same for the other embedder with its own 3 chunkers (use a second terminal if you like).
# If the GPU runs out of memory, add -Window 4096 to estimate, run and report (the same number every time).
param(
  [Parameter(Mandatory = $true)][ValidateSet('estimate', 'run', 'report')][string]$Step,
  [Parameter(Mandatory = $true)][string[]]$Embedders,
  [Parameter(Mandatory = $true)][string[]]$Chunkers,
  [int]$Window = 0,
  [string]$Only = ''
)
# A comma list typed without quotes arrives as a list; join it back into one text.
$Embedders = $Embedders -join ','
$Chunkers = $Chunkers -join ','
$lib = if ($env:RAGCORE_LIB3) { $env:RAGCORE_LIB3 } else { Join-Path (Split-Path $PSScriptRoot -Parent) 'register' }
if (-not (Test-Path -LiteralPath "$lib\sources") -or -not (Test-Path -LiteralPath "$lib\licence-rules.yaml")) {
  Write-Error "Register folder not usable: '$lib' needs a sources folder and licence-rules.yaml. Set RAGCORE_LIB3 or restore the repo's register folder."; exit 1
}
$snaps = if ($env:RAGCORE_SNAPSHOTS) { $env:RAGCORE_SNAPSHOTS } else { 'C:\Users\sashe\.aethelgard\library\snapshots' }
$args2 = @('--register', "$lib\sources", '--snapshots', $snaps, '--questions', 'eval\questions',
           '--exclusions', 'configs\corpus-exclusions.yaml', '--mode', 'late',
           '--embedders', $Embedders, '--chunkers', $Chunkers)
if ($Window -gt 0) { $args2 += @('--late-window', "$Window") }
if ($Only) { $args2 += @('--only', $Only) }
uv run --group embed --group variants python scripts/run_stage.py $Step @args2
