# Stage 1 of the chunker and embedder comparison (spec 13 section 6). Run from the rag-core folder:
#   .\scripts\stage1.ps1 estimate     (prints the plan and time; writes nothing; read it first)
#   .\scripts\stage1.ps1 run          (builds the runs; can take hours; safe to stop and start again)
#   .\scripts\stage1.ps1 report       (prints the paired tables)
# Add -Embedders bge-m3 or -Embedders jina-v2-base-en to do one model per terminal.
param(
  [Parameter(Mandatory = $true)][ValidateSet('estimate', 'run', 'report')][string]$Step,
  [string]$Embedders = 'bge-m3,jina-v2-base-en',
  [string]$Chunkers = 'rules,fixed,merge-small,parent-child,llamaindex,semantic-bge-m3,semantic-jina-v2-base-en',
  [string]$Only = ''
)
$lib = 'D:\01-Project\unsloth\aethelgard-academy\.worktrees\library3\content\library'
$snaps = 'C:\Users\sashe\.aethelgard\library\snapshots'
$args2 = @('--register', "$lib\sources", '--snapshots', $snaps, '--questions', 'eval\questions',
           '--embedders', $Embedders, '--chunkers', $Chunkers)
if ($Only) { $args2 += @('--only', $Only) }
uv run --group embed --group variants python scripts/run_stage.py $Step @args2
