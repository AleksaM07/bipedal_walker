<#
PowerShell helper for setting up the project environment.
Run this from the project root:
    ./setup_env.ps1
#>

$venvPath = "$PSScriptRoot\\.venv"

if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath..."
    python -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists at $venvPath"
}

Write-Host "Installing dependencies from requirements.txt..."
& "$venvPath\\Scripts\\python" -m pip install --upgrade pip
& "$venvPath\\Scripts\\python" -m pip install --disable-pip-version-check -r "$PSScriptRoot\\requirements.txt"

Write-Host "Done. Activate the venv with: . $venvPath\\Scripts\\Activate.ps1"