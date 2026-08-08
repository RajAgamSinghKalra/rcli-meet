# Download whisper.cpp Vulkan binaries + large-v3-turbo model for GPU STT.
# Run once from the rcli-meet folder:
#   powershell -ExecutionPolicy Bypass -File scripts/setup-stt-gpu.ps1
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$BinDir = Join-Path $Root 'bin\whisper'
$ModelsDir = Join-Path $Root 'models'
$ZipPath = Join-Path $env:TEMP 'whisper-vulkan-win-x64.zip'
$ModelPath = Join-Path $ModelsDir 'ggml-large-v3-turbo.bin'
$BinUrl = 'https://github.com/eviscerations/whisper-windows-mcp/releases/download/v1.4.0/whisper-vulkan-win-x64.zip'
$ModelUrl = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin'

New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$server = @(
  (Join-Path $BinDir 'whisper-server.exe'),
  (Join-Path $BinDir 'server.exe')
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $server) {
  Write-Host "Downloading Vulkan whisper.cpp binaries..."
  Invoke-WebRequest -Uri $BinUrl -OutFile $ZipPath -UseBasicParsing
  Expand-Archive -Path $ZipPath -DestinationPath $BinDir -Force
  # Some zips nest a Release/ folder -- flatten common layouts.
  Get-ChildItem -Path $BinDir -Recurse -Filter 'whisper-server.exe' | ForEach-Object {
    if ($_.DirectoryName -ne $BinDir) {
      Copy-Item $_.FullName (Join-Path $BinDir $_.Name) -Force
    }
  }
  Get-ChildItem -Path $BinDir -Recurse -Filter 'server.exe' | ForEach-Object {
    if ($_.DirectoryName -ne $BinDir) {
      Copy-Item $_.FullName (Join-Path $BinDir 'whisper-server.exe') -Force
    }
  }
  Get-ChildItem -Path $BinDir -Recurse -Include '*.dll','ggml*.dll' | ForEach-Object {
    if ($_.DirectoryName -ne $BinDir) {
      Copy-Item $_.FullName (Join-Path $BinDir $_.Name) -Force
    }
  }
  Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
  Write-Host "Binaries installed to $BinDir"
} else {
  Write-Host "Binaries already present: $server"
}

if (-not (Test-Path $ModelPath)) {
  Write-Host "Downloading ggml-large-v3-turbo.bin (~1.6 GB) -- best accuracy for accented English on GPU..."
  Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelPath -UseBasicParsing
  Write-Host "Model saved to $ModelPath"
} else {
  Write-Host "Model already present: $ModelPath"
}

Write-Host ""
Write-Host "GPU STT ready (whisper.dll + ggml-vulkan.dll + large-v3-turbo)."
Write-Host "Verified Vulkan target should print as AMD Radeon RX 6800 XT on first run."
Write-Host "  npm start -- --minutes 20"
