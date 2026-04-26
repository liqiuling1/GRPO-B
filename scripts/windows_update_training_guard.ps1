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

Write-Host "Applying Windows Update guard settings for overnight training..." -ForegroundColor Cyan

New-Item -Path $policyPath -Force | Out-Null
New-ItemProperty -Path $policyPath -Name "NoAutoRebootWithLoggedOnUsers" -PropertyType DWord -Value 1 -Force | Out-Null

# Cover the overnight training window and keep a daytime reboot window.
New-ItemProperty -Path $uxPath -Name "ActiveHoursStart" -PropertyType DWord -Value 14 -Force | Out-Null
New-ItemProperty -Path $uxPath -Name "ActiveHoursEnd" -PropertyType DWord -Value 8 -Force | Out-Null

# Disable Smart Active Hours so Windows does not override the manual window.
New-ItemProperty -Path $uxPath -Name "SmartActiveHoursState" -PropertyType DWord -Value 0 -Force | Out-Null

$result = Get-ItemProperty -Path $uxPath
$policy = Get-ItemProperty -Path $policyPath

Write-Host ""
Write-Host "Applied settings:" -ForegroundColor Green
Write-Host ("  NoAutoRebootWithLoggedOnUsers = {0}" -f $policy.NoAutoRebootWithLoggedOnUsers)
Write-Host ("  ActiveHoursStart             = {0}:00" -f $result.ActiveHoursStart)
Write-Host ("  ActiveHoursEnd               = {0}:00" -f $result.ActiveHoursEnd)
Write-Host ("  SmartActiveHoursState        = {0}" -f $result.SmartActiveHoursState)
Write-Host ""
Write-Host "Recommended practice:" -ForegroundColor Cyan
Write-Host "  1. Stay signed in while long runs are active."
Write-Host "  2. Before multi-night runs, open Windows Update and confirm no restart is pending."
Write-Host "  3. Reboot manually during the daytime window (08:00-14:00) after patch Tuesday updates."
