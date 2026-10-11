<#
Run reproducible Section 16 acceptance checks from the repository root.
Use -SkipInstall only when dependencies are already installed intentionally.
#>
param(
    [switch]$SkipInstall,
    # Only for constrained hosts that cannot create the local sockets TestClient uses.
    # A normal clean-environment acceptance run must leave this switch off.
    [switch]$SkipApiTests
)

$ErrorActionPreference = 'Stop'

function Invoke-PythonChecked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: python $($Arguments -join ' ')"
    }
}

if (-not $SkipInstall) {
    Invoke-PythonChecked -m pip install -r requirements.txt
    Invoke-PythonChecked -m pip install -r requirements-dev.txt
}
if ($SkipApiTests) {
    Invoke-PythonChecked -m pytest -q --ignore=tests/test_api.py
} else {
    Invoke-PythonChecked -m pytest -q
}
Invoke-PythonChecked -m pytest tests/test_overfit.py -q

# Fresh Python process: checkpoint load, three fixed generations, causal perturbation.
Invoke-PythonChecked -m experiments.m5_acceptance_checks --output outputs/logs/acceptance_m5_checks.txt

$readme = Get-Content README.md -Raw
foreach ($required in @('python -m pytest', 'python -m src.train', 'python -m src.generate')) {
    if (-not $readme.Contains($required)) {
        throw "README is missing documented command: $required"
    }
}
foreach ($requiredPath in @('src/train.py', 'src/generate.py', 'tests/test_overfit.py')) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "README command target is missing from the repository: $requiredPath"
    }
}

$localEnv = Get-ChildItem -Recurse -File -Force | Where-Object {
    ($_.Name -eq '.env' -or $_.Name -like '*.env') -and $_.Name -ne '.env.example'
}
if ($localEnv) {
    throw "Found local .env file(s): $($localEnv.FullName -join ', ')"
}

# Scan text source/docs only; generated artifacts and this scanner are excluded to
# avoid regex-literal false positives.
$patterns = @(
    'AKIA[0-9A-Z]{16}',
    'sk-[A-Za-z0-9_-]{20,}',
    'gh[pousr]_[A-Za-z0-9]{30,}',
    '-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'
)
$scanFiles = Get-ChildItem -Recurse -File -Force | Where-Object {
    $_.FullName -notmatch '\\.git\\|\\outputs\\' -and
    $_.FullName -notmatch '\\scripts\\acceptance\.ps1$' -and
    $_.Extension -in '.py', '.md', '.txt', '.yaml', '.yml', '.json', '.toml', '.ps1', '.sh', '.dockerfile'
}
$matches = $scanFiles | Select-String -Pattern $patterns
if ($matches) {
    $matches | Format-Table -AutoSize | Out-String | Write-Error
    throw 'Possible secret found. Remove it before publishing.'
}

Write-Host 'Acceptance checks passed.'
