$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
    Write-Host "This script must be run as Administrator." -ForegroundColor Red
    Write-Host "Right-click PowerShell and choose 'Run as administrator', then run this file." -ForegroundColor Yellow
    exit 1
}

$policyPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU"
$uxPath = "HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings"

Write-Host "Reverting Windows Update training guard settings..." -ForegroundColor Cyan

if (Test-Path $policyPath) {
    Remove-ItemProperty -Path $policyPath -Name "NoAutoRebootWithLoggedOnUsers" -ErrorAction SilentlyContinue
}

# Restore the previous active-hours window observed on this machine.
New-ItemProperty -Path $uxPath -Name "ActiveHoursStart" -PropertyType DWord -Value 8 -Force | Out-Null
New-ItemProperty -Path $uxPath -Name "ActiveHoursEnd" -PropertyType DWord -Value 2 -Force | Out-Null
Remove-ItemProperty -Path $uxPath -Name "SmartActiveHoursState" -ErrorAction SilentlyContinue

$result = Get-ItemProperty -Path $uxPath

Write-Host ""
Write-Host "Reverted settings:" -ForegroundColor Green
Write-Host ("  ActiveHoursStart = {0}:00" -f $result.ActiveHoursStart)
Write-Host ("  ActiveHoursEnd   = {0}:00" -f $result.ActiveHoursEnd)
