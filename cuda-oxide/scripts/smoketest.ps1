# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

param(
    [string]$Example = "vecadd",
    [switch]$BuildOnly,
    [switch]$SkipDoctor
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $repoRoot

function Write-SmokePass {
    param([string]$Name)
    Write-Host "SMOKE PASS: $Name"
}

function Write-SmokeFail {
    param([string]$Name, [string]$Reason)
    Write-Host "SMOKE FAIL: $Name - $Reason"
}

function Write-SmokeSkip {
    param([string]$Name, [string]$Reason)
    Write-Host "SMOKE SKIP: $Name - $Reason"
}

function Invoke-SmokeCommand {
    param(
        [string]$Name,
        [string[]]$Command
    )

    Write-Host "SMOKE RUN: $Name"
    $exe = $Command[0]
    $args = @()
    if ($Command.Count -gt 1) {
        $args = $Command[1..($Command.Count - 1)]
    }
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        Write-SmokeFail $Name "exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-SmokePass $Name
}

function Test-NvidiaGpu {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        return $false
    }

    $output = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
    return ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($output))
}

if (-not (Test-Path "Cargo.toml")) {
    Write-SmokeFail "preflight" "run from inside the cuda-oxide repository"
    exit 2
}

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-SmokeFail "preflight" "cargo was not found on PATH"
    exit 2
}

Write-SmokePass "preflight"

$cargoMetadataJson = & cargo metadata --format-version 1 --no-deps
if ($LASTEXITCODE -ne 0) {
    Write-SmokeFail "cargo metadata" "exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
$cargoMetadata = $cargoMetadataJson | ConvertFrom-Json

$cargoOxide = Join-Path $cargoMetadata.target_directory "debug\cargo-oxide.exe"
Invoke-SmokeCommand "cargo build --locked -p cargo-oxide" @("cargo", "build", "--locked", "-p", "cargo-oxide")
if (-not (Test-Path -LiteralPath $cargoOxide -PathType Leaf)) {
    Write-SmokeFail "cargo-oxide path" "local binary was not produced at $cargoOxide"
    exit 2
}

if (-not $SkipDoctor) {
    Invoke-SmokeCommand "cargo oxide doctor" @($cargoOxide, "doctor")
} else {
    Write-SmokeSkip "cargo oxide doctor" "SkipDoctor was set"
}

Invoke-SmokeCommand "cargo oxide build $Example" @($cargoOxide, "build", $Example)

if ($BuildOnly) {
    Write-SmokeSkip "cargo oxide run $Example" "BuildOnly was set"
    Write-Host "SMOKE SUMMARY: PASS"
    exit 0
}

if (-not (Test-NvidiaGpu)) {
    Write-SmokeSkip "cargo oxide run $Example" "no NVIDIA GPU detected"
    Write-Host "SMOKE SUMMARY: PASS"
    exit 0
}

Invoke-SmokeCommand "cargo oxide run $Example" @($cargoOxide, "run", $Example)
Write-Host "SMOKE SUMMARY: PASS"
