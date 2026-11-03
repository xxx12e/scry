# Build Scry as an MSIX for the Microsoft Store (or sideloading).
#   .\packaging\build_msix.ps1 -Name "12345You.Scry" -Publisher "CN=XXXXXXXX-..." `
#       -PublisherDisplay "Your Name" -Version 1.5.0.0
# Name / Publisher come from Partner Center (Product identity) after reserving the app
# name. For a Store submission upload dist\Scry.msix unsigned: the Store signs it.
# -TestCert creates a self-signed cert matching -Publisher and signs, for local install.
param(
  [Parameter(Mandatory=$true)][string]$Name,
  [Parameter(Mandatory=$true)][string]$Publisher,
  [Parameter(Mandatory=$true)][string]$PublisherDisplay,
  [string]$Version = "1.5.0.0",
  [switch]$TestCert
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  if (-not (Test-Path models\embed\model.onnx)) { python scripts\export_onnx.py models\embed }
  $env:SCRY_CONSOLE = "0"                       # windowed exe; the manifest passes --window
  pyinstaller --noconfirm packaging\scry.spec
  $layout = "build\msix-layout"
  if (Test-Path $layout) { Remove-Item -Recurse -Force $layout }
  New-Item -ItemType Directory "$layout\Assets" | Out-Null
  Copy-Item -Recurse dist\Scry "$layout\Scry"
  python packaging\make_assets.py "$layout\Assets"
  (Get-Content packaging\msix\AppxManifest.xml -Raw) `
    -replace '\{\{NAME\}\}', $Name -replace '\{\{PUBLISHER\}\}', $Publisher `
    -replace '\{\{PUBLISHER_DISPLAY\}\}', $PublisherDisplay -replace '\{\{VERSION\}\}', $Version |
    Set-Content -Encoding UTF8 "$layout\AppxManifest.xml"
  makeappx.exe pack /o /d $layout /p dist\Scry.msix      # Windows SDK
  if ($TestCert) {
    $cert = New-SelfSignedCertificate -Type Custom -Subject $Publisher -KeyUsage DigitalSignature `
      -FriendlyName "Scry test" -CertStoreLocation "Cert:\CurrentUser\My" `
      -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}")
    signtool.exe sign /fd SHA256 /sha1 $cert.Thumbprint dist\Scry.msix
    Write-Host "Trust the cert (Trusted People) before installing: $($cert.Thumbprint)"
  }
  Write-Host "dist\Scry.msix ready"
} finally { Pop-Location }
