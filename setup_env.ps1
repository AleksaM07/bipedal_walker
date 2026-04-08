<#
PowerShell helper za podesavanje okruzenja.
Pokreni ovo iz root foldera projekta:
    ./setup_env.ps1

Po defaultu pokusava da instalira CUDA PyTorch build za projekat.
Ako hoces CPU-only torch, koristi:
    ./setup_env.ps1 -CpuTorch
#>

param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$CpuTorch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-NativeStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$Description = ""
    )

    if ($Description) {
        Write-Host $Description
    }

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        $quotedArgs = $Arguments | ForEach-Object {
            if ($_ -match "\s") {
                '"' + $_ + '"'
            } else {
                $_
            }
        }
        $joinedArgs = $quotedArgs -join " "
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $joinedArgs"
    }
}

# Putanja do virtuelnog okruzenja u ovom projektu.
$venvPath = "$PSScriptRoot\\.venv"
$venvPython = "$venvPath\\Scripts\\python.exe"

# Ako .venv ne postoji, pravimo ga od nule.
if (-Not (Test-Path $venvPath)) {
    Invoke-NativeStep -FilePath "python" -Arguments @("-m", "venv", $venvPath) -Description "Creating virtual environment at $venvPath..."
} else {
    # Ako vec postoji, samo javljamo da ga necemo praviti ponovo.
    Write-Host "Virtual environment already exists at $venvPath"
}

if (-Not (Test-Path $venvPython)) {
    throw "Virtual environment python was not found at $venvPython"
}

# Sada instaliramo sve pakete koji trebaju projektu.
Write-Host "Installing dependencies from requirements.txt..."

# Prvo osvezimo pip da izbegnemo glupe stare probleme.
Invoke-NativeStep -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")

# Onda instaliramo pakete iz requirements.txt.
Invoke-NativeStep -FilePath $venvPython -Arguments @("-m", "pip", "install", "--disable-pip-version-check", "-r", "$PSScriptRoot\\requirements.txt")

if ($CpuTorch) {
    Write-Host "Keeping default PyPI torch build from requirements.txt (CPU/default install)."
} else {
    Invoke-NativeStep -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "--disable-pip-version-check", "--index-url", $TorchIndexUrl, "torch==2.10.0") -Description "Installing CUDA-enabled PyTorch from $TorchIndexUrl..."
}

# Na kraju samo kazemo korisniku kako da aktivira env.
Write-Host "Done. Activate the venv with: . $venvPath\\Scripts\\Activate.ps1"
