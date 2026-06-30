<#
  optimize_server.ps1  -  Windows Server 2019 optimizer for the dual MT5 bots.
  Run ONCE, As Administrator:
      powershell -ExecutionPolicy Bypass -File optimize_server.ps1
  Then RESTART the server, then deploy:
      python runserver.py --install --start-now
  (Mirrors the script embedded in README.md. Idempotent + non-fatal.)
#>
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "STOP: Run PowerShell As Administrator (Run as administrator)!" -ForegroundColor Red
    exit 1
}
# ========================================================================
# DEPLOY & OPTIMIZATION SCRIPT FOR WINDOWS SERVER 2019
# Target Hardware: 4GB RAM / 2 CORES CPU / 64GB SSD
# ========================================================================
# This script squeezes maximum performance from a budget VPS by disabling
# every Windows component that wastes RAM, CPU, and SSD I/O - none of
# which a headless trading server needs. Each step is documented.
# Run ONCE as Administrator. RESTART the server after running.
# ========================================================================


# ----------------------------------------------------------------------
# 1. REMOVE WINDOWS DEFENDER COMPLETELY
# ----------------------------------------------------------------------
# Saves: ~200-400 MB RAM + eliminates real-time scanning disk I/O.
# On a dedicated trading server with no browsing/downloads, AV is pure
# overhead. This REMOVES it entirely (not just disables).
Uninstall-WindowsFeature -Name Windows-Defender


# ----------------------------------------------------------------------
# 2. DISABLE WINDOWS UPDATE & BITS (Background Intelligent Transfer)
# ----------------------------------------------------------------------
# Saves: Prevents surprise reboots, background downloads eating bandwidth,
# and random CPU/disk spikes during update scans.
# CRITICAL for 24/7 trading: an auto-reboot kills open positions.
Stop-Service -Name wuauserv -Force -ErrorAction SilentlyContinue
Set-Service -Name wuauserv -StartupType Disabled

Stop-Service -Name BITS -Force -ErrorAction SilentlyContinue
Set-Service -Name BITS -StartupType Disabled

# Registry-level block (belt-and-suspenders - service + policy)
$WUPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows"
if (-not (Test-Path "$WUPath\WindowsUpdate\AU")) {
    New-Item -Path $WUPath -Name "WindowsUpdate" -Force | Out-Null
    New-Item -Path "$WUPath\WindowsUpdate" -Name "AU" -Force | Out-Null
}
Set-ItemProperty -Path "$WUPath\WindowsUpdate\AU" -Name "NoAutoUpdate" -Value 1 -Type DWORD -Force


# ----------------------------------------------------------------------
# 3. DISABLE WINDOWS SEARCH (Disk Indexing Service)
# ----------------------------------------------------------------------
# Saves: ~50-100 MB RAM + eliminates constant SSD scanning.
# The indexing service continuously reads every file on disk to build
# a search index. On a server with no file searches, this is pure waste
# and shortens SSD lifespan.
Stop-Service -Name WSearch -Force -ErrorAction SilentlyContinue
Set-Service -Name WSearch -StartupType Disabled


# ----------------------------------------------------------------------
# 4. DISABLE TELEMETRY & DIAGNOSTIC TRACKING
# ----------------------------------------------------------------------
# Saves: ~30-80 MB RAM + stops background data collection.
# Microsoft's diagnostic service collects and uploads system data
# continuously. On a trading server this is unnecessary network and
# CPU overhead.
Stop-Service -Name DiagTrack -Force -ErrorAction SilentlyContinue
Set-Service -Name DiagTrack -StartupType Disabled

# Also disable Connected User Experiences (another telemetry vector)
Stop-Service -Name dmwappushservice -Force -ErrorAction SilentlyContinue
Set-Service -Name dmwappushservice -StartupType Disabled


# ----------------------------------------------------------------------
# 5. DISABLE SUPERFETCH / SYSMAIN (Memory Pre-caching)
# ----------------------------------------------------------------------
# Saves: ~100-200 MB RAM + reduces SSD writes.
# SysMain pre-loads frequently used apps into RAM. With only 4GB this
# is counterproductive - it fills RAM with cached data instead of
# leaving it free for MT5 and Python.
Stop-Service -Name SysMain -Force -ErrorAction SilentlyContinue
Set-Service -Name SysMain -StartupType Disabled


# ----------------------------------------------------------------------
# 6. DISABLE PRINT SPOOLER (No printers on a VPS)
# ----------------------------------------------------------------------
# Saves: ~20 MB RAM + removes a known attack surface.
Stop-Service -Name Spooler -Force -ErrorAction SilentlyContinue
Set-Service -Name Spooler -StartupType Disabled


# ----------------------------------------------------------------------
# 7. SET CPU TO HIGH PERFORMANCE MODE
# ----------------------------------------------------------------------
# Effect: Forces Xeon to run at maximum frequency at all times.
# The "Balanced" default throttles CPU speed to save power - on a VPS
# there are no power savings, only added latency. This eliminates the
# ~5-15ms wake-up delay when the CPU has to ramp up from idle.
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c


# ----------------------------------------------------------------------
# 8. OPTIMIZE VISUAL EFFECTS FOR SPEED (RDP Performance)
# ----------------------------------------------------------------------
# Effect: Disables all animations, shadows, transparency, smooth fonts.
# Makes the RDP session dramatically faster and more responsive.
# VisualFXSetting=2 means "Adjust for best performance".
$VisualPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\VisualEffects"
if (-not (Test-Path $VisualPath)) { New-Item -Path $VisualPath -Force | Out-Null }
Set-ItemProperty -Path $VisualPath -Name "VisualFXSetting" -Value 2 -Type DWORD -Force

# System-wide performance settings
$PerfPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced"
Set-ItemProperty -Path $PerfPath -Name "TaskbarAnimations" -Value 0 -Type DWORD -Force


# ----------------------------------------------------------------------
# 9. OPTIMIZE VIRTUAL MEMORY (Pagefile) FOR 4GB RAM
# ----------------------------------------------------------------------
# Set a FIXED pagefile size: min=8192MB, max=8192MB (2x the 4GB RAM).
# A fixed size prevents fragmentation and the constant resize I/O. 8GB (not 4GB)
# gives headroom for 2 MT5 terminals + 2 Python engines + RDP without OOM crashes.
# With 4GB RAM + 8GB pagefile, the system has 12GB virtual memory total.
$cs = Get-WmiObject -Class Win32_ComputerSystem
$cs.AutomaticManagedPagefile = $false
$cs.Put() | Out-Null

$pf = Get-WmiObject -Class Win32_PageFileSetting
if ($pf) {
    $pf.InitialSize = 8192
    $pf.MaximumSize = 8192
    $pf.Put() | Out-Null
} else {
    Set-WmiInstance -Class Win32_PageFileSetting -Arguments @{
        Name = "C:\pagefile.sys"
        InitialSize = 8192
        MaximumSize = 8192
    } | Out-Null
}


# ----------------------------------------------------------------------
# 10. OPTIMIZE SSD - DISABLE DEFRAG, ENABLE TRIM
# ----------------------------------------------------------------------
# SSDs must NEVER be defragmented - it kills the drive. Disable the
# scheduled defrag task and ensure TRIM is enabled for garbage collection.
Disable-ScheduledTask -TaskName "ScheduledDefrag" -TaskPath "\Microsoft\Windows\Defrag\" -ErrorAction SilentlyContinue
fsutil behavior set DisableDeleteNotify 0  # 0 = TRIM enabled


# ----------------------------------------------------------------------
# 11. OPTIMIZE NETWORK FOR TRADING (Low Latency TCP)
# ----------------------------------------------------------------------
# Disable Nagle's algorithm delay (sends small packets immediately)
# and enable TCP timestamps for better RTT measurement.
$tcpPath = "HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters"
Set-ItemProperty -Path $tcpPath -Name "TcpNoDelay" -Value 1 -Type DWORD -Force
Set-ItemProperty -Path $tcpPath -Name "TcpAckFrequency" -Value 1 -Type DWORD -Force
Set-ItemProperty -Path $tcpPath -Name "TCPNoNagle" -Value 1 -Type DWORD -Force

# Disable network throttling (Windows limits non-multimedia traffic by default)
$mmPath = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile"
Set-ItemProperty -Path $mmPath -Name "NetworkThrottlingIndex" -Value 0xFFFFFFFF -Type DWORD -Force


# ----------------------------------------------------------------------
# 12. DISABLE UNNECESSARY SCHEDULED TASKS
# ----------------------------------------------------------------------
# These Microsoft tasks run periodically and waste CPU/disk/network.
$tasksToDisable = @(
    "\Microsoft\Windows\Application Experience\Microsoft Compatibility Appraiser",
    "\Microsoft\Windows\Application Experience\ProgramDataUpdater",
    "\Microsoft\Windows\Autochk\Proxy",
    "\Microsoft\Windows\Customer Experience Improvement Program\Consolidator",
    "\Microsoft\Windows\Customer Experience Improvement Program\UsbCeip",
    "\Microsoft\Windows\DiskDiagnostic\Microsoft-Windows-DiskDiagnosticDataCollector",
    "\Microsoft\Windows\Maps\MapsUpdateTask",
    "\Microsoft\Windows\Windows Error Reporting\QueueReporting"
)
foreach ($task in $tasksToDisable) {
    Disable-ScheduledTask -TaskName (Split-Path $task -Leaf) -TaskPath (Split-Path $task -Parent) -ErrorAction SilentlyContinue
}


# ----------------------------------------------------------------------
# 13. SET TIMEZONE TO FRANKFURT + FORCE NTP SYNC
# ----------------------------------------------------------------------
# Frankfurt (W. Europe Standard Time) aligns logs with the FX trading day.
# NTP sync ensures the system clock is accurate - critical for the
# staleness guard that compares PC time to broker feed timestamps.
tzutil /s "W. Europe Standard Time"

sc.exe config w32time start= auto
net start w32time 2>$null
w32tm /config /manualpeerlist:"time.windows.com,0x9 pool.ntp.org,0x9 time.google.com,0x9" /syncfromflags:manual /reliable:yes /update
w32tm /resync /force


# ----------------------------------------------------------------------
# 14. CONFIGURE WINDOWS EVENT LOG SIZE LIMITS
# ----------------------------------------------------------------------
# Cap log files so they don't silently eat SSD space over months.
wevtutil sl Application /ms:20971520   # 20 MB max
wevtutil sl System /ms:20971520        # 20 MB max
wevtutil sl Security /ms:20971520      # 20 MB max


# ----------------------------------------------------------------------
# DONE
# ----------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  OPTIMIZATION COMPLETE!" -ForegroundColor Green
Write-Host "  Server is now tuned for 24/7 trading on 4GB/2Core/64GB" -ForegroundColor Green
Write-Host "  RESTART the server now: Restart-Computer -Force" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green