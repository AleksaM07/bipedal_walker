<#
PowerShell helper za podesavanje okruzenja.
Pokreni ovo iz root foldera projekta:
    ./setup_env.ps1
#>

# Putanja do virtuelnog okruzenja u ovom projektu.
$venvPath = "$PSScriptRoot\\.venv"

# Ako .venv ne postoji, pravimo ga od nule.
if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath..."
    python -m venv $venvPath
} else {
    # Ako vec postoji, samo javljamo da ga necemo praviti ponovo.
    Write-Host "Virtual environment already exists at $venvPath"
}

# Sada instaliramo sve pakete koji trebaju projektu.
Write-Host "Installing dependencies from requirements.txt..."

# Prvo osvezimo pip da izbegnemo glupe stare probleme.
& "$venvPath\\Scripts\\python" -m pip install --upgrade pip

# Onda instaliramo pakete iz requirements.txt.
& "$venvPath\\Scripts\\python" -m pip install --disable-pip-version-check -r "$PSScriptRoot\\requirements.txt"

# Na kraju samo kazemo korisniku kako da aktivira env.
Write-Host "Done. Activate the venv with: . $venvPath\\Scripts\\Activate.ps1"
