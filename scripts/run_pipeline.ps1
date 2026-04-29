param(
    [ValidateSet("bootstrap", "train", "full")]
    [string]$Mode = "bootstrap",
    [ValidateSet("cpu", "cuda", "auto")]
    [string]$Device = "auto",
    [int]$StartYear = 2024,
    [int]$EndYear = 2026,
    [switch]$SkipApi
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$argsList = @(
    "run", "python", "-m", "src.tools.user_pipeline",
    "--mode", $Mode,
    "--device", $Device,
    "--start-year", "$StartYear",
    "--end-year", "$EndYear"
)

if ($SkipApi) {
    $argsList += "--skip-api"
}

Write-Host "Executando user pipeline..."
poetry @argsList

