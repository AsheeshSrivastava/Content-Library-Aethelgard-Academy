# The index (spec 17). Run from the rag-core folder. It costs no money; everything runs on this PC.
#   .\scripts\index.ps1 sync -DryRun        shows what would be added, replaced or removed, changes nothing
#   .\scripts\index.ps1 sync                fills or updates the index (safe to run again; only new text is embedded)
#   .\scripts\index.ps1 status              shows the index: counts, last sync, and the setup it was built with
#   .\scripts\index.ps1 check               scores the 137 frozen questions through the search; must give 100 first / 121 in the first 3
#   .\scripts\index.ps1 search "question"   shows the best passages with their source links  [-K 5] [-MaxPerFile 2]
# `sync` stops, and says so, if it would remove half or more of the stored sources (a wrong folder, for example);
# add -AllowMassDelete only when that is what you want. The index file is ~\.ragcore\index\index.db.
param(
  [Parameter(Mandatory = $true, Position = 0)][ValidateSet('sync', 'status', 'search', 'check')][string]$Step,
  [Parameter(Position = 1)][string]$Question = '',
  [switch]$DryRun,
  [switch]$AllowMassDelete,
  [int]$K = 5,
  [int]$MaxPerFile = 0,
  [string]$Index = ''
)
$lib = if ($env:RAGCORE_LIB3) { $env:RAGCORE_LIB3 } else { Join-Path (Split-Path $PSScriptRoot -Parent) 'register' }
if (-not (Test-Path -LiteralPath "$lib\sources") -or -not (Test-Path -LiteralPath "$lib\licence-rules.yaml")) {
  Write-Error "Register folder not usable: '$lib' needs a sources folder and licence-rules.yaml. Set RAGCORE_LIB3 or restore the repo's register folder."; exit 1
}
$snaps = if ($env:RAGCORE_SNAPSHOTS) { $env:RAGCORE_SNAPSHOTS } else { 'C:\Users\sashe\.aethelgard\library\snapshots' }
$args2 = @($Step)
if ($Question) { $args2 += $Question }
if ($Step -eq 'sync') {
  $args2 += @('--register', "$lib\sources", '--snapshots', $snaps, '--exclusions', 'configs\corpus-exclusions.yaml')
  if ($DryRun) { $args2 += '--dry-run' }
  if ($AllowMassDelete) { $args2 += '--allow-mass-delete' }
}
if ($Step -eq 'check') {
  $args2 += @('--register', "$lib\sources", '--snapshots', $snaps, '--exclusions', 'configs\corpus-exclusions.yaml')
}
if ($Step -eq 'search') {
  $args2 += @('-k', $K)
  if ($MaxPerFile -gt 0) { $args2 += @('--max-per-file', $MaxPerFile) }
}
if ($Index) { $args2 += @('--index', $Index) }
uv run --group embed --group variants python scripts/run_index.py @args2
