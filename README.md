# ⚡ INSTITUTIONAL-GRADE DUAL TRADING SYSTEM (BTC + XAU)

> **24/7 Automated Signal Engine** — Two independent bots (Bitcoin + Gold) running
> side-by-side on a single Windows VPS, each with its own isolated MetaTrader 5
> terminal. Self-healing, auto-restarting, zero-downtime architecture.

---

## 📋 TABLE OF CONTENTS

1. [System Architecture](#-system-architecture)
2. [Hardware Requirements](#-hardware-requirements)
3. [Pre-Installation Checklist](#-pre-installation-checklist)
4. [Step 1 — Server Optimization Script](#-step-1--server-optimization-script-run-first)
5. [Step 2 — Python Environment Setup](#-step-2--python-environment-setup)
6. [Step 3 — Configure Bot Credentials](#-step-3--configure-bot-credentials)
7. [Step 4 — Deploy & Start](#-step-4--deploy--start)
8. [Daily Operations](#-daily-operations)
9. [Monitoring & Health Checks](#-monitoring--health-checks)
10. [Troubleshooting](#-troubleshooting)
11. [File Structure](#-file-structure)

---

## 🏗 SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│                   MASTER RUNSERVER                       │
│              (runserver.py — root level)                 │
│                                                         │
│  • Provisions separate MT5 terminals per bot            │
│  • Supervises both child watchdogs forever               │
│  • Self-heals via Task Scheduler (every 1 min check)    │
│  • Single-instance lock (port 49345)                    │
│                                                         │
│  ┌────────────────────┐   ┌────────────────────┐        │
│  │  BTC WATCHDOG      │   │  XAU WATCHDOG      │        │
│  │  (BTC/runserver.py)│   │  (XAU/runserver.py)│        │
│  │  Lock: port 49347  │   │  Lock: port 49346  │        │
│  │                    │   │                    │        │
│  │  ┌──────────────┐  │   │  ┌──────────────┐  │        │
│  │  │  BTC main.py │  │   │  │  XAU main.py │  │        │
│  │  │  BTCUSDm     │  │   │  │  XAUUSDm     │  │        │
│  │  │  Magic:234001│  │   │  │  Magic:234000│  │        │
│  │  └──────────────┘  │   │  └──────────────┘  │        │
│  └────────┬───────────┘   └────────┬───────────┘        │
│           │                        │                    │
│  ┌────────▼───────────┐   ┌────────▼───────────┐        │
│  │  MT5 Terminal BTC  │   │  MT5 Terminal XAU  │        │
│  │  C:\MT5_Terminals\ │   │  C:\MT5_Terminals\ │        │
│  │  BTC\terminal64.exe│   │  XAU\terminal64.exe│        │
│  └────────────────────┘   └────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### Trading Engine Features (per bot)
- **6-timeframe analysis**: D1, H4, H1, M15, M5, M1
- **SMC (Smart Money Concepts)**: Order blocks, institutional zones
- **CVD + VWAP**: Cumulative Volume Delta, Volume-Weighted Average Price
- **Risk management**: ATR-based SL/TP, breakeven, partial TP, trailing stop
- **Signal scorecard**: Forward paper-test every signal on the real market
- **Telegram alerts**: Real-time notifications for signals and execution
- **Anti-chasing**: Overextension veto, spread filter, entry drift guard

### MT5 Terminal Isolation
The `_mt5_isolation/sitecustomize.py` shim ensures each bot connects to its
**own private MT5 terminal** in portable mode — zero code changes to the bots,
zero conflicts. The master sets `BOT_MT5_TERMINAL` per child process.

---

## 💻 HARDWARE REQUIREMENTS

| Resource | Minimum | This Server | Status |
|----------|---------|-------------|--------|
| OS | Windows Server 2019 | Windows Server 2019 | ✅ |
| CPU | 2 cores (Xeon/Ryzen) | 2 Cores Xeon | ✅ |
| RAM | 4 GB | 4 GB | ✅ |
| Disk | 40 GB SSD | 64 GB SSD | ✅ |
| Network | Stable internet | VPS datacenter | ✅ |
| Python | 3.10+ | 3.12 | ✅ |

### Resource Usage (measured)
| Component | RAM | CPU (idle) | Disk |
|-----------|-----|------------|------|
| Windows Server 2019 (optimized) | ~800 MB | 1-3% | — |
| MT5 Terminal × 2 | ~300 MB each | 1-2% each | ~500 MB each |
| Python bot × 2 | ~80 MB each | <1% each | — |
| **Total** | **~1.6 GB** | **~5-8%** | **~2 GB** |
| **Free after optimization** | **~2.4 GB** | **~92%** | **~60 GB** |

---

## ✅ PRE-INSTALLATION CHECKLIST

Before running anything, ensure:

- [ ] Windows Server 2019 is installed and activated
- [ ] RDP access is configured and working
- [ ] Internet connection is stable
- [ ] MetaTrader 5 (Exness) is installed (`exness5setup.exe` in `_mt5_isolation/`)
- [ ] MT5 is logged in to your trading account at least once
- [ ] Python 3.12 is installed (add to PATH)
- [ ] Telegram bot token is created via @BotFather
- [ ] Your Telegram chat ID is known (use @userinfobot)

---

## 🔧 STEP 1 — SERVER OPTIMIZATION SCRIPT (RUN FIRST!)

> **Target**: Windows Server 2019 — 4GB RAM / 2 Cores CPU / 64GB SSD
>
> Run this **ONCE** in PowerShell as **Administrator**, then **RESTART** the server.

```powershell
# ════════════════════════════════════════════════════════════════════════
# DEPLOY & OPTIMIZATION SCRIPT FOR WINDOWS SERVER 2019
# Target Hardware: 4GB RAM / 2 CORES CPU / 64GB SSD
# ════════════════════════════════════════════════════════════════════════
# This script squeezes maximum performance from a budget VPS by disabling
# every Windows component that wastes RAM, CPU, and SSD I/O — none of
# which a headless trading server needs. Each step is documented.
# Run ONCE as Administrator. RESTART the server after running.
# ════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# 1. REMOVE WINDOWS DEFENDER COMPLETELY
# ──────────────────────────────────────────────────────────────────────
# Saves: ~200-400 MB RAM + eliminates real-time scanning disk I/O.
# On a dedicated trading server with no browsing/downloads, AV is pure
# overhead. This REMOVES it entirely (not just disables).
Uninstall-WindowsFeature -Name Windows-Defender


# ──────────────────────────────────────────────────────────────────────
# 2. DISABLE WINDOWS UPDATE & BITS (Background Intelligent Transfer)
# ──────────────────────────────────────────────────────────────────────
# Saves: Prevents surprise reboots, background downloads eating bandwidth,
# and random CPU/disk spikes during update scans.
# CRITICAL for 24/7 trading: an auto-reboot kills open positions.
Stop-Service -Name wuauserv -Force -ErrorAction SilentlyContinue
Set-Service -Name wuauserv -StartupType Disabled

Stop-Service -Name BITS -Force -ErrorAction SilentlyContinue
Set-Service -Name BITS -StartupType Disabled

# Registry-level block (belt-and-suspenders — service + policy)
$WUPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows"
if (-not (Test-Path "$WUPath\WindowsUpdate\AU")) {
    New-Item -Path $WUPath -Name "WindowsUpdate" -Force | Out-Null
    New-Item -Path "$WUPath\WindowsUpdate" -Name "AU" -Force | Out-Null
}
Set-ItemProperty -Path "$WUPath\WindowsUpdate\AU" -Name "NoAutoUpdate" -Value 1 -Type DWORD -Force


# ──────────────────────────────────────────────────────────────────────
# 3. DISABLE WINDOWS SEARCH (Disk Indexing Service)
# ──────────────────────────────────────────────────────────────────────
# Saves: ~50-100 MB RAM + eliminates constant SSD scanning.
# The indexing service continuously reads every file on disk to build
# a search index. On a server with no file searches, this is pure waste
# and shortens SSD lifespan.
Stop-Service -Name WSearch -Force -ErrorAction SilentlyContinue
Set-Service -Name WSearch -StartupType Disabled


# ──────────────────────────────────────────────────────────────────────
# 4. DISABLE TELEMETRY & DIAGNOSTIC TRACKING
# ──────────────────────────────────────────────────────────────────────
# Saves: ~30-80 MB RAM + stops background data collection.
# Microsoft's diagnostic service collects and uploads system data
# continuously. On a trading server this is unnecessary network and
# CPU overhead.
Stop-Service -Name DiagTrack -Force -ErrorAction SilentlyContinue
Set-Service -Name DiagTrack -StartupType Disabled

# Also disable Connected User Experiences (another telemetry vector)
Stop-Service -Name dmwappushservice -Force -ErrorAction SilentlyContinue
Set-Service -Name dmwappushservice -StartupType Disabled


# ──────────────────────────────────────────────────────────────────────
# 5. DISABLE SUPERFETCH / SYSMAIN (Memory Pre-caching)
# ──────────────────────────────────────────────────────────────────────
# Saves: ~100-200 MB RAM + reduces SSD writes.
# SysMain pre-loads frequently used apps into RAM. With only 4GB this
# is counterproductive — it fills RAM with cached data instead of
# leaving it free for MT5 and Python.
Stop-Service -Name SysMain -Force -ErrorAction SilentlyContinue
Set-Service -Name SysMain -StartupType Disabled


# ──────────────────────────────────────────────────────────────────────
# 6. DISABLE PRINT SPOOLER (No printers on a VPS)
# ──────────────────────────────────────────────────────────────────────
# Saves: ~20 MB RAM + removes a known attack surface.
Stop-Service -Name Spooler -Force -ErrorAction SilentlyContinue
Set-Service -Name Spooler -StartupType Disabled


# ──────────────────────────────────────────────────────────────────────
# 7. SET CPU TO HIGH PERFORMANCE MODE
# ──────────────────────────────────────────────────────────────────────
# Effect: Forces Xeon to run at maximum frequency at all times.
# The "Balanced" default throttles CPU speed to save power — on a VPS
# there are no power savings, only added latency. This eliminates the
# ~5-15ms wake-up delay when the CPU has to ramp up from idle.
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c


# ──────────────────────────────────────────────────────────────────────
# 8. OPTIMIZE VISUAL EFFECTS FOR SPEED (RDP Performance)
# ──────────────────────────────────────────────────────────────────────
# Effect: Disables all animations, shadows, transparency, smooth fonts.
# Makes the RDP session dramatically faster and more responsive.
# VisualFXSetting=2 means "Adjust for best performance".
$VisualPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\VisualEffects"
if (-not (Test-Path $VisualPath)) { New-Item -Path $VisualPath -Force | Out-Null }
Set-ItemProperty -Path $VisualPath -Name "VisualFXSetting" -Value 2 -Type DWORD -Force

# System-wide performance settings
$PerfPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced"
Set-ItemProperty -Path $PerfPath -Name "TaskbarAnimations" -Value 0 -Type DWORD -Force


# ──────────────────────────────────────────────────────────────────────
# 9. OPTIMIZE VIRTUAL MEMORY (Pagefile) FOR 4GB RAM
# ──────────────────────────────────────────────────────────────────────
# Set a FIXED pagefile size: min=4096MB, max=4096MB (matches RAM).
# A fixed size prevents fragmentation and the constant resize I/O.
# With 4GB RAM + 4GB pagefile, the system has 8GB virtual memory total.
$cs = Get-WmiObject -Class Win32_ComputerSystem
$cs.AutomaticManagedPagefile = $false
$cs.Put() | Out-Null

$pf = Get-WmiObject -Class Win32_PageFileSetting
if ($pf) {
    $pf.InitialSize = 4096
    $pf.MaximumSize = 4096
    $pf.Put() | Out-Null
} else {
    Set-WmiInstance -Class Win32_PageFileSetting -Arguments @{
        Name = "C:\pagefile.sys"
        InitialSize = 4096
        MaximumSize = 4096
    } | Out-Null
}


# ──────────────────────────────────────────────────────────────────────
# 10. OPTIMIZE SSD — DISABLE DEFRAG, ENABLE TRIM
# ──────────────────────────────────────────────────────────────────────
# SSDs must NEVER be defragmented — it kills the drive. Disable the
# scheduled defrag task and ensure TRIM is enabled for garbage collection.
Disable-ScheduledTask -TaskName "ScheduledDefrag" -TaskPath "\Microsoft\Windows\Defrag\" -ErrorAction SilentlyContinue
fsutil behavior set DisableDeleteNotify 0  # 0 = TRIM enabled


# ──────────────────────────────────────────────────────────────────────
# 11. OPTIMIZE NETWORK FOR TRADING (Low Latency TCP)
# ──────────────────────────────────────────────────────────────────────
# Disable Nagle's algorithm delay (sends small packets immediately)
# and enable TCP timestamps for better RTT measurement.
$tcpPath = "HKLM:\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters"
Set-ItemProperty -Path $tcpPath -Name "TcpNoDelay" -Value 1 -Type DWORD -Force
Set-ItemProperty -Path $tcpPath -Name "TcpAckFrequency" -Value 1 -Type DWORD -Force
Set-ItemProperty -Path $tcpPath -Name "TCPNoNagle" -Value 1 -Type DWORD -Force

# Disable network throttling (Windows limits non-multimedia traffic by default)
$mmPath = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile"
Set-ItemProperty -Path $mmPath -Name "NetworkThrottlingIndex" -Value 0xFFFFFFFF -Type DWORD -Force


# ──────────────────────────────────────────────────────────────────────
# 12. DISABLE UNNECESSARY SCHEDULED TASKS
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# 13. SET TIMEZONE TO FRANKFURT + FORCE NTP SYNC
# ──────────────────────────────────────────────────────────────────────
# Frankfurt (W. Europe Standard Time) aligns logs with the FX trading day.
# NTP sync ensures the system clock is accurate — critical for the
# staleness guard that compares PC time to broker feed timestamps.
tzutil /s "W. Europe Standard Time"

sc.exe config w32time start= auto
net start w32time 2>$null
w32tm /config /manualpeerlist:"time.windows.com,0x9 pool.ntp.org,0x9 time.google.com,0x9" /syncfromflags:manual /reliable:yes /update
w32tm /resync /force


# ──────────────────────────────────────────────────────────────────────
# 14. CONFIGURE WINDOWS EVENT LOG SIZE LIMITS
# ──────────────────────────────────────────────────────────────────────
# Cap log files so they don't silently eat SSD space over months.
wevtutil sl Application /ms:20971520   # 20 MB max
wevtutil sl System /ms:20971520        # 20 MB max
wevtutil sl Security /ms:20971520      # 20 MB max


# ──────────────────────────────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "  OPTIMIZATION COMPLETE!" -ForegroundColor Green
Write-Host "  Server is now tuned for 24/7 trading on 4GB/2Core/64GB" -ForegroundColor Green
Write-Host "  RESTART the server now: Restart-Computer -Force" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
```

### What this saves (total):

| Optimization | RAM Saved | CPU Saved | SSD I/O Saved |
|---|---|---|---|
| Remove Windows Defender | ~300 MB | 3-5% | High |
| Disable Windows Update | ~50 MB | Burst 10%+ | High |
| Disable Windows Search | ~80 MB | 1-3% | Very High |
| Disable Telemetry | ~50 MB | 1% | Medium |
| Disable SysMain/Superfetch | ~150 MB | 1% | High |
| Disable Print Spooler | ~20 MB | 0% | Low |
| Fixed Pagefile | — | — | Medium |
| Disable Network Throttling | — | — | — |
| **TOTAL SAVED** | **~650 MB** | **~8-15%** | **Massive** |

After optimization + restart, expect **~800 MB** base OS usage (down from ~1.5 GB).

---

## 🐍 STEP 2 — PYTHON ENVIRONMENT SETUP

```powershell
# Navigate to project
cd C:\Users\Kabir\Desktop\Trader

# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install all dependencies (includes TA-Lib from local wheel)
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`):
```
numpy==2.5.0
pandas==2.2.3
./_mt5_isolation/ta_lib-0.6.4-cp312-cp312-win_amd64.whl
MetaTrader5==5.0.5735
pyTelegramBotAPI==4.34.0
requests==2.34.2
```

---

## 🔑 STEP 3 — CONFIGURE BOT CREDENTIALS

Each bot has its own `.env` file. Create them from the examples:

### `BTC/.env`
```env
# Telegram
TG_BOT_TOKEN=123456:ABC-DEF...your-bot-token
TG_ADMIN_CHAT_ID=your-chat-id

# MetaTrader 5 (Exness)
MT5_LOGIN=12345678
MT5_PASSWORD=your-mt5-password
MT5_SERVER=Exness-MT5Real6
MT5_PATH=C:\MT5_Terminals\BTC\terminal64.exe

# Risk (optional overrides)
RISK_PCT=1.0
MAGIC=234001
```

### `XAU/.env`
```env
# Telegram
TG_BOT_TOKEN=123456:ABC-DEF...your-bot-token
TG_ADMIN_CHAT_ID=your-chat-id

# MetaTrader 5 (Exness)
MT5_LOGIN=12345678
MT5_PASSWORD=your-mt5-password
MT5_SERVER=Exness-MT5Real6
MT5_PATH=C:\MT5_Terminals\XAU\terminal64.exe

# Risk (optional overrides)
RISK_PCT=1.0
MAGIC=234000
```

> ⚠️ **IMPORTANT**: Both bots can use the SAME MT5 account but MUST have
> **different MAGIC numbers** (234001 for BTC, 234000 for XAU) so they never
> touch each other's positions.

---

## 🚀 STEP 4 — DEPLOY & START

### One-command deployment (run as Administrator):

```powershell
cd C:\Users\Kabir\Desktop\Trader
python runserver.py --install --start-now
```

**This single command:**
1. ✅ Runs pre-flight checks (Python deps, .env, MT5, disk space, network)
2. ✅ Sets timezone to Frankfurt + forces NTP sync
3. ✅ Clones MT5 terminal → `C:\MT5_Terminals\BTC\` and `\XAU\`
4. ✅ Creates Windows Task Scheduler task (self-healing, every 1 minute)
5. ✅ Starts supervising both bots immediately

### Verify isolation:
```powershell
python runserver.py --verify
```
This proves each bot connects to its **own separate** MT5 terminal.

---

## 📟 DAILY OPERATIONS

| Command | What it does |
|---------|-------------|
| `python runserver.py` | Provision + run both bots (foreground) |
| `python runserver.py --install --start-now` | Full install + start (first time) |
| `python runserver.py --verify` | Prove terminal isolation works |
| `python runserver.py --status` | Show health, PIDs, scheduled tasks |
| `python runserver.py --kill` | **STOP EVERYTHING** + remove autostart |
| `python runserver.py --set-timezone` | Force Frankfurt time + NTP sync |

### Per-bot commands (if needed):
```powershell
# BTC bot only
cd BTC
python runserver.py --status
python runserver.py --kill
python runserver.py --kill --close-all   # kill + close all BTC positions

# XAU bot only
cd XAU
python runserver.py --status
python runserver.py --kill
python runserver.py --kill --close-all   # kill + close all XAU positions
```

---

## 📊 MONITORING & HEALTH CHECKS

### Health files (auto-updated):
- `Logs/master_health.json` — Master supervisor status
- `BTC/Logs/runserver_health.json` — BTC watchdog status
- `XAU/Logs/runserver_health.json` — XAU watchdog status

### Log files:
- `Logs/master.log` — Master supervisor log
- `BTC/Logs/runserver.log` — BTC watchdog log
- `BTC/Logs/main_supervised.log` — BTC trading engine output
- `XAU/Logs/runserver.log` — XAU watchdog log
- `XAU/Logs/main_supervised.log` — XAU trading engine output

### Telegram notifications:
Both bots send real-time alerts:
- 🟢 Bot started / restarted
- 📊 Signal detected (with confidence, direction, SL/TP)
- ✅ Order executed
- 🛑 Kill switch activated
- ⚠️ Errors and warnings

---

## 🔧 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| Bot won't start | Check `.env` — `TG_BOT_TOKEN` and `TG_ADMIN_CHAT_ID` required |
| MT5 connection fails | Ensure MT5 is installed and logged in at least once |
| "Another instance running" | Run `python runserver.py --kill` first |
| Port conflict | Ports 49345-49347 must be free (localhost only) |
| Terminal not provisioned | Run `python runserver.py --setup-only` |
| Python module missing | Run `pip install -r requirements.txt` in venv |
| SSD running low | Check `Logs/` folder sizes, old logs auto-rotate (10MB × 10) |
| Clock drift warnings | Run `python runserver.py --set-timezone` as Admin |

---

## 📁 FILE STRUCTURE

```
C:\Users\Kabir\Desktop\Trader\
│
├── runserver.py              # MASTER — supervises both bots
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── BTC/                      # Bitcoin Signal Bot
│   ├── .env                  # Credentials (secrets)
│   ├── config.py             # All tunable parameters
│   ├── main.py               # Trading engine entry point
│   ├── analysis.py           # 6-TF signal analysis + SMC
│   ├── mt5_clent.py          # MT5 connection + order execution
│   ├── signal_tracker.py     # Forward paper-test scorecard
│   ├── trade_manager.py      # Breakeven / partial TP / trailing
│   └── runserver.py          # Per-bot watchdog (child of master)
│
├── XAU/                      # Gold Signal Bot
│   ├── .env                  # Credentials (secrets)
│   ├── config.py             # All tunable parameters
│   ├── main.py               # Trading engine entry point
│   ├── analysis.py           # 6-TF signal analysis + SMC
│   ├── mt5_clent.py          # MT5 connection + order execution
│   ├── signal_tracker.py     # Forward paper-test scorecard
│   ├── trade_manager.py      # Breakeven / partial TP / trailing
│   └── runserver.py          # Per-bot watchdog (child of master)
│
├── _mt5_isolation/           # Terminal isolation layer
│   ├── sitecustomize.py      # Auto-loaded shim (pins MT5 per bot)
│   ├── exness5setup.exe      # MT5 installer (Exness)
│   └── ta_lib-*.whl          # TA-Lib wheel for Python 3.12
│
├── .venv/                    # Python virtual environment
└── Logs/                     # Master-level logs
    ├── master.log
    └── master_health.json
```

---

## 🛡 SELF-HEALING ARCHITECTURE

The system has **3 layers of protection** against any failure:

1. **Layer 1 — Watchdog** (per-bot `runserver.py`):
   Auto-restarts `main.py` on crash with exponential backoff (5s → 60s max)

2. **Layer 2 — Master Supervisor** (root `runserver.py`):
   Monitors both watchdogs, restarts any that exit, cleans orphans on startup

3. **Layer 3 — Task Scheduler** (Windows):
   Fires every **1 minute**. If the master is alive (holds port 49345), the
   relaunch exits instantly. If the master is dead, the next tick takes over
   within 60 seconds — including after crashes, OOM kills, or server reboots.

**Result**: The bots auto-recover from ANY failure — process crash, network
drop, MT5 hang, even a full server reboot — with zero manual intervention.

---

*Built for 24/7 unattended operation on Windows Server 2019 (4GB RAM / 2 Cores / 64GB SSD)*