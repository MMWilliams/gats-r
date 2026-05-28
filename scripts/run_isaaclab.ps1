#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Launcher that sets the Isaac-Sim env vars then runs a Python script with the
    Isaac-Lab conda env's Python.

.EXAMPLE
    pwsh scripts/run_isaaclab.ps1 scripts/isaaclab_smoke.py
    pwsh scripts/run_isaaclab.ps1 scripts/isaaclab_benchmark.py --num_envs 16 --episodes 4

    All args after the script path are forwarded verbatim.
#>
[CmdletBinding(PositionalBinding=$false)]
param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ScriptPath,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Forwarded
)

$ErrorActionPreference = 'Stop'

$IsaacSim    = if ($env:GATSR_ISAACSIM_PATH) { $env:GATSR_ISAACSIM_PATH } else { 'C:\isaac-sim' }
$CondaEnvPy  = if ($env:GATSR_CONDA_PY) { $env:GATSR_CONDA_PY } else { 'C:\Users\reese\miniconda3\envs\isaaclab\python.exe' }

if (-not (Test-Path $IsaacSim))   { throw "Isaac Sim not found at $IsaacSim - set GATSR_ISAACSIM_PATH." }
if (-not (Test-Path $CondaEnvPy)) { throw "Conda env python not found at $CondaEnvPy - set GATSR_CONDA_PY." }
if (-not (Test-Path $ScriptPath)) { throw "Script not found at $ScriptPath." }

$env:CARB_APP_PATH = Join-Path $IsaacSim 'kit'
$env:EXP_PATH      = Join-Path $IsaacSim 'apps'
$env:ISAAC_PATH    = $IsaacSim
$env:RESOURCE_NAME = 'IsaacSim'
$existing = if ($env:PYTHONPATH) { $env:PYTHONPATH } else { '' }
$siteDir = Join-Path $IsaacSim 'site'
if ($existing -notlike "*$siteDir*") {
    $env:PYTHONPATH = if ($existing) { "$existing;$siteDir" } else { $siteDir }
}

Write-Host "[run] Isaac Sim:  $IsaacSim"
Write-Host "[run] Python:     $CondaEnvPy"
Write-Host "[run] Script:     $ScriptPath $($Forwarded -join ' ')"
Write-Host ""

& $CondaEnvPy $ScriptPath @Forwarded
exit $LASTEXITCODE
