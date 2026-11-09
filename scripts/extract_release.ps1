[CmdletBinding()]
param(
    [string]$SourceDir = "dist",
    [string]$Destination = "",
    [switch]$SkipVerify,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = $RepoRoot
}
$Destination = [System.IO.Path]::GetFullPath($Destination)

$runner = Join-Path $Destination "scripts\run_preprocessing.py"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Destination bukan root repo MSMD: $Destination (scripts\run_preprocessing.py tidak ditemukan)"
}

if ([System.IO.Path]::IsPathRooted($SourceDir)) {
    $SourcePath = $SourceDir
} else {
    $SourcePath = Join-Path $RepoRoot $SourceDir
}

if (-not (Test-Path -LiteralPath $SourcePath)) {
    throw "Folder sumber tidak ditemukan: $SourcePath"
}

$zips = @(Get-ChildItem -LiteralPath $SourcePath -Filter "preproc-v1-*.zip" -File | Sort-Object Name)
if ($zips.Count -eq 0) {
    throw "Tidak ada preproc-v1-*.zip di $SourcePath. Jalankan scripts\build_release.py atau arahkan -SourceDir ke folder zip."
}

Write-Host "Repo     : $Destination"
Write-Host "Sumber   : $SourcePath"
Write-Host "Zip      : $($zips.Count) file"
foreach ($zip in $zips) {
    Write-Host ("  - {0} ({1:N1} MB)" -f $zip.Name, ($zip.Length / 1MB))
}

if ($DryRun) {
    Write-Host "Dry run selesai; tidak ada file yang diekstrak."
    exit 0
}

foreach ($zip in $zips) {
    Write-Host "Extract $($zip.Name) ..."
    Expand-Archive -LiteralPath $zip.FullName -DestinationPath $Destination -Force
}

$required = @(
    "data\processed\audio_windows_train.npy",
    "data\processed\score_crops_train.npy",
    "data\processed\system_crops_train.npy",
    "data\processed\error_windows_train.npy",
    "reports\validation_preprocessing.json"
)

$missing = @()
foreach ($item in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $Destination $item))) {
        $missing += $item
    }
}

if ($missing.Count -gt 0) {
    throw "Ekstraksi belum lengkap. File wajib tidak ditemukan: $($missing -join ', ')"
}

Write-Host "Ekstraksi selesai dan file wajib ditemukan."

if (-not $SkipVerify) {
    Push-Location $Destination
    try {
        & python $runner --verify-only
        if ($LASTEXITCODE -ne 0) {
            throw "Validasi gagal dengan exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}
