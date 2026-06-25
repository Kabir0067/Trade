"""
═══════════════════════════════════════════════════════════════════════════
  Bitcoin Signal Bot — Windows production watchdog (DevOps, 24/7)
═══════════════════════════════════════════════════════════════════════════

ONE COMMAND does everything (install auto-start + start now + run forever):

        python runserver.py --install --start-now

That single command:
  • writes a launcher .bat and registers Windows Task Scheduler tasks
    (ONLOGON always, ONSTART/SYSTEM when run as Administrator) so the bot
    auto-starts on every login / reboot,
  • starts the watchdog immediately,
  • the watchdog supervises main.py forever and auto-restarts it on any crash
    (exponential backoff), survives network drops, and sends Telegram alerts.

Nothing else is required. The only commands you need are:
    
    python runserver.py --install --start-now
    python runserver.py --kill

"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# Emoji/Unicode-safe console on Windows cp1251 (logger writes here too)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR         = Path(__file__).resolve().parent
MAIN_PY             = PROJECT_DIR / "main.py"
VENV_PYTHON         = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
LOG_DIR             = PROJECT_DIR / "Logs"
DOTENV_FILE         = PROJECT_DIR / ".env"

RUNSERVER_LOG       = LOG_DIR / "runserver.log"
MAIN_SUPERVISED_LOG = LOG_DIR / "main_supervised.log"
HEALTH_TICK_FILE    = LOG_DIR / "runserver_health.json"
LAUNCHER_BAT        = PROJECT_DIR / "runserver_start.bat"

BOOT_TASK_NAME      = "BitcoinSignalBotBoot"
LOGON_TASK_NAME     = "BitcoinSignalBotLogon"
FALLBACK_LOCK_PORT  = 49347     # singleton guard (distinct from other bots)
TELEGRAM_TIMEOUT_SEC = 8

# Server time. The bot's session detection is UTC-based (uses explicit zones), so
# the OS timezone does NOT change trading logic — but an ACCURATE clock does (the
# staleness guard compares the PC clock to the broker feed; drift => false "stale").
# So we set Frankfurt for readable logs AND force an NTP resync so it cannot drift.
SERVER_TIMEZONE = "W. Europe Standard Time"   # Frankfurt/Berlin (auto-handles DST)
NTP_PEERS = "time.windows.com,0x9 pool.ntp.org,0x9 time.google.com,0x9"

NETWORK_TARGETS: Tuple[Tuple[str, int], ...] = (
    ("1.1.1.1", 443),
    ("8.8.8.8", 53),
    ("api.telegram.org", 443),
)

# Only these are hard requirements. MT5 creds are optional (the bot attaches to
# the already logged-in terminal); they are reported as warnings if missing.
_REQUIRED_ENV_KEYS = ("TG_BOT_TOKEN", "TG_ADMIN_CHAT_ID")


class RunServerError(RuntimeError):
    """Raised for expected runserver failures."""


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────
def build_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("runserver")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(RUNSERVER_LOG, maxBytes=10_000_000, backupCount=10,
                             encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


log = build_logger()


# ─────────────────────────────────────────────────────────────────────────────
# OS / env helpers
# ─────────────────────────────────────────────────────────────────────────────
def is_windows() -> bool:
    return os.name == "nt"

def is_admin() -> bool:
    if not is_windows():
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def project_python() -> Path:
    return VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable).resolve()


def quote_cmd(value: Path | str) -> str:
    return '"' + str(value).replace('"', r'\"') + '"'


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_dotenv(path: Path) -> Dict[str, str]:
    """Parse .env into a dict (no python-dotenv dependency)."""
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            env[key] = val
    except Exception as exc:
        log.warning("Failed to parse .env: %s", exc)
    return env


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight validation
# ─────────────────────────────────────────────────────────────────────────────
def preflight_check(env: Dict[str, str]) -> Tuple[bool, List[str], List[str]]:
    failures: List[str] = []
    warnings: List[str] = []

    if not MAIN_PY.exists():
        failures.append(f"main.py not found: {MAIN_PY}")
    if not project_python().exists():
        failures.append(f"Python not found: {project_python()}")

    if not DOTENV_FILE.exists():
        failures.append(".env file missing — copy .env.example to .env and fill it")
    else:
        missing = [k for k in _REQUIRED_ENV_KEYS if not env.get(k, "").strip()]
        if missing:
            failures.append(f".env missing required keys: {', '.join(missing)}")
        tok = env.get("TG_BOT_TOKEN", "").strip()
        if tok and ":" not in tok:
            failures.append("TG_BOT_TOKEN looks invalid (must contain ':')")
        if not (env.get("MT5_PASSWORD") or env.get("EXNESS_PASSWORD")):
            warnings.append("MT5_PASSWORD not set — bot will attach to the logged-in "
                            "terminal (fine if MT5 is already logged in)")

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        t = LOG_DIR / ".write_test"
        t.write_text("ok")
        t.unlink()
    except Exception as exc:
        failures.append(f"Logs/ not writable: {exc}")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", FALLBACK_LOCK_PORT)) == 0:
            warnings.append(f"Port {FALLBACK_LOCK_PORT} already bound — runserver may "
                            "already be running")
        s.close()
    except Exception:
        pass

    try:
        free_mb = shutil.disk_usage(str(PROJECT_DIR)).free / (1024 * 1024)
        if free_mb < 300:
            warnings.append(f"Low disk space: {free_mb:.0f} MB free")
    except Exception:
        pass

    if sys.version_info < (3, 10):
        failures.append(f"Python {sys.version.split()[0]} too old — require 3.10+")

    try:
        if not network_available(timeout=2.0):
            warnings.append("No network — Telegram/MT5 may be down; watchdog retries each cycle")
    except Exception:
        pass

    return len(failures) == 0, failures, warnings


def run_preflight_and_report(env: Dict[str, str], *, die_on_failure: bool = True) -> bool:
    log.info("=" * 60)
    log.info("PRE-FLIGHT CHECK")
    log.info("=" * 60)
    ok, failures, warnings = preflight_check(env)
    for w in warnings:
        log.warning("  [WARN]  %s", w)
    for f in failures:
        log.error("  [FAIL]  %s", f)
    if ok:
        log.info("  [PASS]  Pre-flight complete — system is ready")
    else:
        log.error("  PRE-FLIGHT FAILED (%d) — fix before starting", len(failures))
        if die_on_failure:
            sys.exit(1)
    log.info("=" * 60)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Telegram alerts (standalone urllib)
# ─────────────────────────────────────────────────────────────────────────────
def _alert_creds(env: Dict[str, str]) -> Tuple[str, str]:
    token = (env.get("TG_BOT_TOKEN") or env.get("BOT_TOKEN") or "").strip()
    chat = (env.get("TG_ADMIN_CHAT_ID") or env.get("ADMIN_ID") or "").strip()
    return token, chat


def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=TELEGRAM_TIMEOUT_SEC) as resp:
            return bool(json.loads(resp.read().decode("utf-8", errors="replace")).get("ok"))
    except Exception as exc:
        log.debug("Telegram alert failed (non-critical): %s", exc)
        return False


def telegram_alert(env: Dict[str, str], text: str) -> None:
    token, chat = _alert_creds(env)
    if not token or not chat:
        return
    threading.Thread(target=_send_telegram, args=(token, chat, text), daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Health tick
# ─────────────────────────────────────────────────────────────────────────────
def write_health_tick(state: dict) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        HEALTH_TICK_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        log.debug("Health tick write failed: %s", exc)


def health_tick_loop(stop_event: threading.Event, state: dict,
                     interval: float, lock: threading.Lock) -> None:
    while not stop_event.is_set():
        try:
            with lock:
                state["last_tick_utc"] = _now_utc()
                state["uptime_seconds"] = int(time.monotonic() - state.get("_t0", time.monotonic()))
                snapshot = {k: v for k, v in state.items() if not k.startswith("_")}
            write_health_tick(snapshot)
        except Exception as exc:
            log.debug("Health tick iteration failed (non-fatal): %s", exc)
        stop_event.wait(interval)


# ─────────────────────────────────────────────────────────────────────────────
# Project / launcher / Task Scheduler
# ─────────────────────────────────────────────────────────────────────────────
def ensure_project_ready() -> None:
    if not MAIN_PY.exists():
        raise RunServerError(f"main.py not found: {MAIN_PY}")
    if not project_python().exists():
        raise RunServerError(f"Python not found: {project_python()}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_launcher_bat() -> Path:
    ensure_project_ready()
    py = project_python()
    content = (
        "@echo off\n"
        "setlocal\n"
        "title Bitcoin Signal Bot RunServer\n"
        'cd /d "%~dp0"\n'
        "set PYTHONUNBUFFERED=1\n"
        "set PYTHONIOENCODING=utf-8\n"
        "set RUNSERVER_MANAGED=1\n"
        f"{quote_cmd(py)} " + '"%~dp0runserver.py" --run\n'
        "exit /b %ERRORLEVEL%\n"
    )
    LAUNCHER_BAT.write_text(content, encoding="utf-8", newline="\r\n")
    log.info("Launcher written: %s", LAUNCHER_BAT)
    return LAUNCHER_BAT


def run_command(command: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    log.info("CMD: %s", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_DIR, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace")
    if result.stdout.strip():
        log.info("CMD output:\n%s", result.stdout.strip())
    if check and result.returncode != 0:
        raise RunServerError(f"Command failed ({result.returncode}): {' '.join(command)}")
    return result


def scheduler_available() -> bool:
    return is_windows() and shutil.which("schtasks") is not None


def _create_task(name: str, extra_flags: list[str]) -> bool:
    cmd = ["schtasks", "/Create", "/TN", name, "/TR", quote_cmd(LAUNCHER_BAT), "/F"] + extra_flags
    return run_command(cmd).returncode == 0


def create_logon_task() -> bool:
    flags = ["/SC", "ONLOGON"]
    if is_admin():
        flags += ["/RL", "HIGHEST"]
    ok = _create_task(LOGON_TASK_NAME, flags)
    log.info("Logon task %s: %s", LOGON_TASK_NAME, "created" if ok else "FAILED")
    return ok


def create_boot_task() -> bool:
    if not is_admin():
        log.warning("Boot task needs Administrator — only the logon task was installed")
        return False
    ok = _create_task(BOOT_TASK_NAME, ["/SC", "ONSTART", "/RU", "SYSTEM", "/RL", "HIGHEST"])
    log.info("Boot task %s: %s", BOOT_TASK_NAME, "created" if ok else "FAILED")
    return ok


def task_exists(name: str) -> bool:
    return is_windows() and run_command(["schtasks", "/Query", "/TN", name]).returncode == 0


def configure_server_time(tz_name: str = SERVER_TIMEZONE) -> bool:
    """Set the Windows timezone to Frankfurt and FORCE an NTP time sync.

    The bot's session logic is UTC-based (timezone-independent), so the timezone is
    cosmetic — but the NTP resync is what actually matters: it stops the system
    clock from drifting, which would otherwise make the staleness guard misfire.
    Admin-only on Windows; non-fatal (prints the manual steps if it can't run)."""
    if not is_windows():
        log.info("Time config skipped — not Windows.")
        return False
    if not is_admin():
        log.warning("Timezone/NTP sync needs Administrator — SKIPPED. Run once, elevated:")
        log.warning('    tzutil /s "%s"', tz_name)
        log.warning("    w32tm /resync /force")
        return False

    log.info("Configuring server time → %s + NTP sync ...", tz_name)
    ok = True

    # 1) timezone (Windows auto-handles DST for this ID)
    if run_command(["tzutil", "/s", tz_name]).returncode == 0:
        log.info("  timezone set: %s", tz_name)
    else:
        ok = False
        log.error("  tzutil failed for '%s'", tz_name)

    # 2) ensure the Windows Time service is enabled + running
    run_command(["sc", "config", "w32time", "start=", "auto"])
    run_command(["sc", "start", "w32time"])        # 'already running' is fine (ignored)

    # 3) point it at reliable public NTP peers and force a sync
    run_command(["w32tm", "/config", "/manualpeerlist:" + NTP_PEERS,
                 "/syncfromflags:manual", "/reliable:yes", "/update"])
    if run_command(["w32tm", "/resync", "/force"]).returncode == 0:
        log.info("  NTP resync OK")
    else:
        log.warning("  NTP resync non-zero (peer unreachable now?) — service will retry")

    run_command(["w32tm", "/query", "/status"])    # read-only: show the result
    return ok


def install(start_now: bool = False) -> int:
    if not is_windows():
        raise RunServerError("--install is Windows-only.")
    if not scheduler_available():
        raise RunServerError("schtasks not available.")
    ensure_project_ready()
    # FIRST: fix the server clock/timezone (Frankfurt + NTP) so sessions & the
    # staleness guard are reliable from the very first cycle.
    configure_server_time()
    # quick, non-fatal preflight so the user sees issues right away
    run_preflight_and_report(parse_dotenv(DOTENV_FILE), die_on_failure=False)
    write_launcher_bat()
    logon_ok = create_logon_task()
    boot_ok = create_boot_task()
    if not (logon_ok or boot_ok):
        raise RunServerError("No scheduled task could be created.")
    log.info("Install complete. logon=%s boot=%s launcher=%s", logon_ok, boot_ok, LAUNCHER_BAT)
    if not boot_ok:
        log.info("TIP: run this once 'As administrator' to also auto-start on reboot "
                 "(before any login).")
    return start_tasks() if start_now else 0


def delete_task(name: str) -> bool:
    ok = run_command(["schtasks", "/Delete", "/TN", name, "/F"]).returncode == 0
    log.info("Task %s: %s", name, "deleted" if ok else "not found")
    return ok


def uninstall(remove_bat: bool = False) -> int:
    if is_windows():
        delete_task(LOGON_TASK_NAME)
        delete_task(BOOT_TASK_NAME)
    if remove_bat and LAUNCHER_BAT.exists():
        LAUNCHER_BAT.unlink()
        log.info("Removed: %s", LAUNCHER_BAT)
    return 0


def start_tasks() -> int:
    started = any(run_command(["schtasks", "/Run", "/TN", name]).returncode == 0
                  for name in (LOGON_TASK_NAME, BOOT_TASK_NAME) if task_exists(name))
    if started:
        log.info("✅ Started via Task Scheduler — the bot now runs in the background 24/7.")
        return 0
    log.warning("No task started; running watchdog directly (foreground).")
    return run_watchdog()


def print_status() -> int:
    print(f"Project:  {PROJECT_DIR}")
    print(f"Python:   {project_python()}")
    print(f"Main:     {MAIN_PY}")
    print(f"Launcher: {LAUNCHER_BAT} ({'exists' if LAUNCHER_BAT.exists() else 'missing'})")
    print(f"Admin:    {is_admin()}")
    print(f"Health:   {HEALTH_TICK_FILE} ({'exists' if HEALTH_TICK_FILE.exists() else 'missing'})")
    if HEALTH_TICK_FILE.exists():
        try:
            h = json.loads(HEALTH_TICK_FILE.read_text(encoding="utf-8"))
            print(f"  status={h.get('status')} restarts={h.get('restart_count')} "
                  f"uptime={h.get('uptime_seconds')}s last_tick={h.get('last_tick_utc')}")
        except Exception:
            pass
    if is_windows():
        for name in (LOGON_TASK_NAME, BOOT_TASK_NAME):
            r = run_command(["schtasks", "/Query", "/TN", name, "/FO", "LIST"])
            print(f"\n[{name}]")
            print(r.stdout.strip() if r.stdout.strip() else "not registered")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# KILL SWITCH
# ─────────────────────────────────────────────────────────────────────────────
def _pids_on_port(port: int) -> set:
    """PIDs LISTENING on a TCP port (the running watchdog holds the lock port)."""
    pids: set = set()
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True,
                             text=True, errors="replace").stdout
        for line in out.splitlines():
            parts = line.split()
            if (len(parts) >= 5 and parts[0].upper() == "TCP"
                    and parts[1].endswith(f":{port}") and parts[3].upper() == "LISTENING"
                    and parts[4].isdigit()):
                pids.add(int(parts[4]))
    except Exception as exc:
        log.debug("netstat failed: %s", exc)
    return pids


def _kill_pid_tree(pid: int) -> bool:
    return run_command(["taskkill", "/PID", str(pid), "/T", "/F"]).returncode == 0


def _kill_by_cmdline() -> int:
    """Kill leftover python running THIS project's runserver --run / main.py (scoped)."""
    if not is_windows():
        return 0
    me = os.getpid()
    pdir = str(PROJECT_DIR)
    ps = (
        f"$me={me}; Get-CimInstance Win32_Process -Filter "
        f"\"name='python.exe' or name='pythonw.exe'\" | Where-Object {{ "
        f"$_.ProcessId -ne $me -and ("
        f"$_.CommandLine -like '*{pdir}\\runserver.py*--run*' -or "
        f"$_.CommandLine -like '*{pdir}\\main.py*') }} | "
        f"ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }}"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, errors="replace").stdout
        return len([x for x in out.split() if x.strip().isdigit()])
    except Exception as exc:
        log.debug("cmdline kill failed: %s", exc)
        return 0


def _close_all_positions() -> None:
    """Panic-flatten: close every open position opened by this bot (magic)."""
    try:
        import MetaTrader5 as mt5
        import config
        from mt5_clent import _pick_filling_mode   # single source for the filling rule
        if not mt5.initialize():
            log.warning("close-all: MT5 initialize failed")
            return
        closed = 0
        for p in (mt5.positions_get() or []):
            if p.magic != config.MAGIC:
                continue
            si = mt5.symbol_info(p.symbol)
            tick = mt5.symbol_info_tick(p.symbol)
            ctype = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
            fill = _pick_filling_mode(si)
            r = mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": p.volume,
                "type": ctype, "position": p.ticket, "price": price, "deviation": 50,
                "magic": config.MAGIC, "comment": "KILL close",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": fill,
            })
            if r is not None and r.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
        log.info("close-all: closed %d bot position(s)", closed)
        mt5.shutdown()
    except Exception as exc:
        log.error("close-all failed: %s", exc)


def kill_switch(close_all: bool = False) -> int:
    """Stop EVERYTHING: disable auto-start tasks, kill watchdog + main.py, optionally flatten."""
    log.info("=" * 60)
    log.info("KILL SWITCH%s", "  (+ CLOSE ALL POSITIONS)" if close_all else "")
    log.info("=" * 60)

    # 1) stop scheduled tasks, then completely uninstall them
    if is_windows():
        for name in (LOGON_TASK_NAME, BOOT_TASK_NAME):
            if task_exists(name):
                run_command(["schtasks", "/End", "/TN", name])
        uninstall(remove_bat=True)

    # 2) kill the running watchdog (holds the lock port) + its main.py child tree
    killed = 0
    for pid in _pids_on_port(FALLBACK_LOCK_PORT):
        if _kill_pid_tree(pid):
            killed += 1
    # 3) fallback: any leftover python running this project's runserver/main
    killed += _kill_by_cmdline()

    env = parse_dotenv(DOTENV_FILE)
    telegram_alert(env, f"🛑 KILL SWITCH activated\nServer: {socket.gethostname()}\n"
                        f"Processes killed: {killed}\nTime: {_now_utc()}")

    # 4) optional panic flatten (after the bot is dead, so no race)
    if close_all:
        time.sleep(1.0)
        _close_all_positions()

    log.info("Kill switch complete. processes killed=%d, auto-start DISABLED.", killed)
    log.info("To start again: python runserver.py --install --start-now")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────
def network_available(timeout: float = 3.0) -> bool:
    for host, port in NETWORK_TARGETS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def wait_for_network(stop_event: threading.Event, check_interval: float = 10.0) -> bool:
    while not stop_event.is_set():
        if network_available():
            return True
        log.warning("Network unavailable — retry in %.0fs", check_interval)
        stop_event.wait(check_interval)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Process management
# ─────────────────────────────────────────────────────────────────────────────
def _set_high_priority() -> None:
    if not is_windows():
        return
    try:
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(h, 0x00008000)  # ABOVE_NORMAL
        log.info("Runserver priority: ABOVE_NORMAL")
    except Exception as exc:
        log.warning("Could not set priority: %s", exc)


def _child_creation_flags() -> int:
    if not is_windows():
        return 0
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) | 0x00008000


def _build_child_env() -> Dict[str, str]:
    child = os.environ.copy()
    child["PYTHONUNBUFFERED"] = "1"
    child["PYTHONIOENCODING"] = "utf-8"
    child["RUNSERVER_MANAGED"] = "1"
    child["PATH"] = str(PROJECT_DIR / ".venv" / "Scripts") + os.pathsep + child.get("PATH", "")
    return child


def _append_child_log_header() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    header = "\n" + "=" * 78 + f"\nrunserver: child start | {_now_utc()}\n" + "=" * 78 + "\n"
    with MAIN_SUPERVISED_LOG.open("ab") as fh:
        fh.write(header.encode("utf-8", errors="replace"))


def start_child() -> subprocess.Popen[bytes]:
    py = project_python()
    _append_child_log_header()
    child_log = MAIN_SUPERVISED_LOG.open("ab", buffering=0)
    try:
        proc = subprocess.Popen([str(py), "-u", str(MAIN_PY)], cwd=PROJECT_DIR,
                                env=_build_child_env(), stdout=child_log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                creationflags=_child_creation_flags())
    except Exception:
        child_log.close()
        raise
    proc._runserver_log_handle = child_log  # type: ignore[attr-defined]
    return proc


def close_child_log(proc: subprocess.Popen[bytes]) -> None:
    h = getattr(proc, "_runserver_log_handle", None)
    if h is not None:
        try:
            h.close()
        except Exception:
            pass


def terminate_child(proc: subprocess.Popen[bytes], timeout: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    log.info("Stopping child pid=%s", proc.pid)
    try:
        if is_windows():
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning("Child did not stop in %.0fs — killing", timeout)
        proc.kill()
        proc.wait(timeout=10)


def wait_child(proc: subprocess.Popen[bytes], stop_event: threading.Event) -> Optional[int]:
    while not stop_event.is_set():
        code = proc.poll()
        if code is not None:
            return code
        stop_event.wait(1.0)
    terminate_child(proc)
    return proc.returncode


# ─────────────────────────────────────────────────────────────────────────────
# Single-instance guard
# ─────────────────────────────────────────────────────────────────────────────
@contextmanager
def single_instance() -> Iterator[object]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if is_windows() and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", FALLBACK_LOCK_PORT))
        sock.listen(1)
        yield sock
    except OSError as exc:
        raise RunServerError("Another runserver instance is already running.") from exc
    finally:
        sock.close()


# ─────────────────────────────────────────────────────────────────────────────
# Watchdog loop
# ─────────────────────────────────────────────────────────────────────────────
def run_watchdog(*, no_network_wait: bool = False, no_preflight: bool = False,
                 max_restart_delay: float = 60.0) -> int:
    ensure_project_ready()
    _set_high_priority()
    env = parse_dotenv(DOTENV_FILE)

    if not no_preflight:
        run_preflight_and_report(env, die_on_failure=True)

    stop_event = threading.Event()

    def request_stop(signum: int, _frame: object) -> None:
        log.info("Stop signal: %s", signum)
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)

    health_lock = threading.Lock()
    health: Dict = {
        "_t0": time.monotonic(), "started_utc": _now_utc(), "pid": os.getpid(),
        "status": "starting", "restart_count": 0, "last_exit_code": None,
        "last_tick_utc": _now_utc(), "uptime_seconds": 0,
    }

    restart_delay = 5.0
    rapid_crash_window: List[float] = []

    try:
        instance_ctx = single_instance()
        instance_ctx.__enter__()
    except RunServerError as exc:
        log.error("%s", exc)
        return 0

    try:
        write_health_tick({k: v for k, v in health.items() if not k.startswith("_")})
        threading.Thread(target=health_tick_loop, args=(stop_event, health, 60.0, health_lock),
                         daemon=True).start()
        telegram_alert(env, f"🟢 Bitcoin Bot runserver started\nServer: {socket.gethostname()}\n"
                            f"Time: {_now_utc()}")
        log.info("Watchdog started. project=%s python=%s", PROJECT_DIR, project_python())
        with health_lock:
            health["status"] = "running"

        while not stop_event.is_set():
            if not no_network_wait and not wait_for_network(stop_event):
                break

            started_at = time.monotonic()
            proc: Optional[subprocess.Popen[bytes]] = None
            code: Optional[int] = None
            try:
                proc = start_child()
                with health_lock:
                    health["status"] = "running"
                    health["child_pid"] = proc.pid
                    restart_no = health["restart_count"]
                log.info("main.py started (pid=%s restart#%d)", proc.pid, restart_no)
                code = wait_child(proc, stop_event)
            except Exception as exc:
                log.exception("Failed to start/supervise main.py: %s", exc)
                code = -1
            finally:
                if proc is not None:
                    close_child_log(proc)

            if stop_event.is_set():
                break

            uptime = time.monotonic() - started_at
            with health_lock:
                health["last_exit_code"] = code
                health["restart_count"] += 1
                health["status"] = "restarting"
                restart_no = health["restart_count"]

            # main.py runs forever; ANY exit is unexpected -> backoff restart.
            if uptime >= 300:
                restart_delay = 5.0
                reason = "exit_after_long_run"
            else:
                restart_delay = min(max_restart_delay, max(5.0, restart_delay * 1.5))
                reason = "early_exit"

            log.warning("main.py exited code=%s uptime=%.1fs reason=%s restart#%d in %.1fs",
                        code, uptime, reason, restart_no, restart_delay)

            now = time.monotonic()
            rapid_crash_window = [t for t in rapid_crash_window if now - t < 300]
            rapid_crash_window.append(now)
            if len(rapid_crash_window) >= 5:
                telegram_alert(env, f"⚠️ RAPID CRASH LOOP\nmain.py exited "
                                    f"{len(rapid_crash_window)}x in 5 min\nlast code={code}\n"
                                    f"Server: {socket.gethostname()}\nTime: {_now_utc()}")
                rapid_crash_window.clear()
            else:
                telegram_alert(env, f"🔴 main.py exited (code={code})\nUptime: {uptime:.0f}s\n"
                                    f"Restarting in {restart_delay:.0f}s (#{restart_no})")

            stop_event.wait(restart_delay)
    finally:
        try:
            instance_ctx.__exit__(None, None, None)
        except Exception:
            pass

    with health_lock:
        health["status"] = "stopped"
        total = health["restart_count"]
        snap = {k: v for k, v in health.items() if not k.startswith("_")}
    write_health_tick(snap)
    telegram_alert(env, f"🔴 Bitcoin Bot runserver stopped\nServer: {socket.gethostname()}\n"
                        f"Restarts: {total}\nTime: {_now_utc()}")
    log.info("Runserver stopped. Total restarts: %d", total)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bitcoin Signal Bot RunServer — Windows production watchdog.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--install", action="store_true", help="Install auto-start tasks.")
    p.add_argument("--start-now", action="store_true", help="With --install, start immediately.")
    p.add_argument("--run", action="store_true",
                   help="Run the watchdog in the foreground (used by Task Scheduler / the .bat).")
    p.add_argument("--status", action="store_true", help="Print install/health status and exit.")
    p.add_argument("--set-time", action="store_true",
                   help="Set timezone to Frankfurt + force NTP sync (Admin), then exit.")
    p.add_argument("--kill", action="store_true", help="KILL SWITCH: stop completely and uninstall.")
    p.add_argument("--close-all", action="store_true",
                   help="With --kill, also flatten all bot positions (panic close).")
    return p.parse_args(list(argv))


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.kill:
            return kill_switch(close_all=args.close_all)
        if args.set_time:
            return 0 if configure_server_time() else 1
        if args.status:
            return print_status()
        if args.install:
            return install(start_now=args.start_now)
        # --run (from the .bat / Task Scheduler) and the no-arg default both
        # supervise main.py in the foreground.
        return run_watchdog()
    except RunServerError as exc:
        log.error("FATAL: %s", exc)
        return 2
    except KeyboardInterrupt:
        log.info("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
