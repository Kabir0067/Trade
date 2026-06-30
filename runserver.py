"""
═══════════════════════════════════════════════════════════════════════════
  MASTER RUNSERVER  —  runs BOTH bots (BTC + XAU) side-by-side, conflict-free
═══════════════════════════════════════════════════════════════════════════

WHAT THIS DOES (and what it deliberately does NOT do)
-----------------------------------------------------
It launches the *existing* per-bot watchdogs — BTC/runserver.py and
XAU/runserver.py — as two independent child processes and supervises them
forever. It does NOT modify either bot's code or trading logic in any way.

The ONE thing the two bots could collide on is the MetaTrader 5 terminal: by
default both attach to the same single terminal. This master fixes that with
two cheap, fully reversible pieces of plumbing:

  1. It provisions a SEPARATE MT5 terminal for each bot (a slim copy of the
     installed terminal, minus the regenerable history cache), under
     C:\\MT5_Terminals\\BTC and \\XAU.  Your original terminal is never touched.

  2. It launches each bot with BOT_MT5_TERMINAL set to that bot's own terminal
     and the isolation shim (_mt5_isolation/sitecustomize.py) on PYTHONPATH, so
     every mt5.initialize() in that process is pinned — in portable mode — to
     that one terminal.  BTC <-> its terminal, XAU <-> its terminal. Guaranteed.

ONE COMMAND does everything (provision terminals + install autostart + run):

        python runserver.py --install --start-now

Day-to-day commands:

        python runserver.py                 # provision (if needed) + run both, foreground
        python runserver.py --verify        # prove each bot binds to its OWN terminal, then exit
        python runserver.py --status        # show state of everything
        python runserver.py --kill          # stop EVERYTHING (master + both bots) and uninstall
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Emoji/Unicode-safe console on Windows code pages.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Layout & constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent
ROOT_VENV_PY    = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
SHIM_DIR        = ROOT_DIR / "_mt5_isolation"
LOG_DIR         = ROOT_DIR / "Logs"
LAUNCHER_BAT    = ROOT_DIR / "runserver_start.bat"
HEALTH_FILE     = LOG_DIR / "master_health.json"

# Where the per-bot terminal clones live. The SOURCE terminal to clone is
# auto-detected at run time (see _find_source_terminal) so this works on ANY
# server regardless of where MetaTrader 5 is installed.
TERMINALS_ROOT   = Path(r"C:\MT5_Terminals")
# Folders inside the terminal that are pure cache/logs — skipped when cloning so
# each copy is small and fast; the terminal regenerates them on first launch.
CLONE_EXCLUDE_DIRS = ("Bases", "logs", "Logs", "temp", "Tester")

MASTER_LOCK_PORT = 49345   # distinct from BTC (49347) and XAU (49346)
BOOT_TASK_NAME   = "AllSignalBotsBoot"
LOGON_TASK_NAME  = "AllSignalBotsLogon"


class Bot:
    def __init__(self, name: str, terminal_name: str, lock_port: int):
        self.name = name
        self.dir = ROOT_DIR / name
        self.runserver = self.dir / "runserver.py"
        self.terminal = TERMINALS_ROOT / terminal_name / "terminal64.exe"
        self.console_log = LOG_DIR / f"{name}_console.log"
       
        self.lock_port = lock_port
        self.proc: Optional[subprocess.Popen] = None
        self.restarts = 0
        self.restart_delay = 5.0
        self.next_start = 0.0
        self.started_at = 0.0   # monotonic time of the last successful launch


BOTS: List[Bot] = [
    Bot("BTC", "BTC", 49347),
    Bot("XAU", "XAU", 49346),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tiny logger (stdout + Logs/master.log)
# ─────────────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"{_now()} | {msg}"
    # Under Task Scheduler there is no console, so sys.stdout can be None — a bare
    # print() would then raise and kill the master before it ever starts. Guard it.
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / "master.log").open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def is_windows() -> bool:
    return os.name == "nt"


def is_admin() -> bool:
    if not is_windows():
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def root_python() -> Path:
    return ROOT_VENV_PY if ROOT_VENV_PY.exists() else Path(sys.executable).resolve()


def quote(p) -> str:
    return '"' + str(p).replace('"', r'\"') + '"'


# ─────────────────────────────────────────────────────────────────────────────
# 1) Provision a private terminal per bot (idempotent, original untouched)
# ─────────────────────────────────────────────────────────────────────────────
def _read_env_value(env_path: Path, key: str) -> str:
    """Pull a single KEY=VALUE out of a bot's .env (no dependency)."""
    try:
        for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _find_source_terminal() -> Optional[Path]:
    """Locate a real, installed MetaTrader 5 terminal to clone FROM — on ANY
    server, without hard-coding a path. Search order:
      1) MT5_PATH set in either bot's .env (this is the broker the bots target),
      2) the common Program Files install folders (incl. broker-named ones),
      3) any terminal64.exe one level under Program Files (covers custom names).
    Returns the folder that contains terminal64.exe, or None if MT5 isn't found.
    """
    candidates: List[Path] = []

    # 1) Whatever the bots' own .env already points at.
    for bot in BOTS:
        p = _read_env_value(bot.dir / ".env", "MT5_PATH")
        if p:
            candidates.append(Path(p).parent if p.lower().endswith(".exe") else Path(p))

    # 2) Common fixed install locations.
    pf = [os.environ.get("ProgramFiles", r"C:\Program Files"),
          os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
    for base in pf:
        candidates += [
            Path(base) / "MetaTrader 5 EXNESS",
            Path(base) / "MetaTrader 5",
            Path(base) / "Exness MetaTrader 5",
        ]

    for c in candidates:
        try:
            if (c / "terminal64.exe").exists():
                return c
        except Exception:
            continue

    # 3) Glob: any *MetaTrader*/terminal64.exe directly under Program Files.
    for base in pf:
        try:
            for d in Path(base).glob("*[Mm]eta[Tt]rader*"):
                if (d / "terminal64.exe").exists():
                    return d
        except Exception:
            continue
    return None


def provision_terminals() -> None:
    if not is_windows():
        log("Not Windows — skipping terminal provisioning.")
        return

    # Skip the whole search if both clones already exist (fast idempotent path).
    if all(bot.terminal.exists() for bot in BOTS):
        for bot in BOTS:
            log(f"[{bot.name}] terminal already present: {bot.terminal}")
        return

    src = _find_source_terminal()
    if src is None:
        log("WARN  No installed MetaTrader 5 terminal found to clone from. "
            "Install MT5 (the broker's terminal) on this server, then re-run "
            "`python runserver.py --setup-only`.")
        return
    log(f"Source terminal detected: {src}")

    for bot in BOTS:
        dest = bot.terminal.parent
        if bot.terminal.exists():
            log(f"[{bot.name}] terminal already present: {bot.terminal}")
            continue
        log(f"[{bot.name}] cloning terminal -> {dest}  (excluding cache: "
            f"{', '.join(CLONE_EXCLUDE_DIRS)}) ...")
        dest.mkdir(parents=True, exist_ok=True)
        # robocopy is the reliable Windows mirror tool. /E recurse, /XD excludes
        # the cache dirs. Exit codes 0-7 are success; >=8 is a real failure.
        cmd = ["robocopy", str(src), str(dest), "/E",
               "/XD", *CLONE_EXCLUDE_DIRS,
               "/R:1", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP"]
        rc = subprocess.run(cmd, capture_output=True, text=True,
                            errors="replace").returncode
        if rc >= 8:
            log(f"[{bot.name}] robocopy failed (code {rc}). Falling back to "
                f"shutil.copytree ...")
            try:
                shutil.copytree(
                    src, dest, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(*CLONE_EXCLUDE_DIRS))
            except Exception as exc:
                log(f"[{bot.name}] ERROR copy failed: {exc}")
                continue
        if bot.terminal.exists():
            log(f"[{bot.name}] OK terminal ready: {bot.terminal}")
        else:
            log(f"[{bot.name}] ERROR terminal64.exe missing after clone — check {dest}")


# ─────────────────────────────────────────────────────────────────────────────
# 1b) Preflight — fail LOUDLY on a fresh server instead of silently
# ─────────────────────────────────────────────────────────────────────────────
def preflight() -> bool:
    """Warn about anything that would stop the bots running on this machine.
    Returns False if a hard requirement is missing (still continues — the
    watchdogs retry — but the operator sees exactly what to fix)."""
    ok = True
    py = root_python()
    log(f"Preflight: python = {py}")

    # Python dependencies the bots import.
    probe = ("import importlib,sys;"
             "miss=[m for m in ['MetaTrader5','numpy','pandas','telebot'] "
             "if importlib.util.find_spec(m) is None];"
             "print('MISSING='+','.join(miss))")
    try:
        out = subprocess.run([str(py), "-c", probe], capture_output=True,
                             text=True, errors="replace", timeout=60).stdout
        miss = ""
        for ln in out.splitlines():
            if ln.startswith("MISSING="):
                miss = ln[len("MISSING="):].strip()
        if miss:
            ok = False
            log(f"WARN  Python is missing packages: {miss}. Install them, e.g.: "
                f'"{py}" -m pip install MetaTrader5 numpy pandas pytelegrambotapi')
        else:
            log("Preflight: all Python deps present (MetaTrader5, numpy, pandas, telebot).")
    except Exception as exc:
        log(f"WARN  Could not check Python deps: {exc}")

    # Each bot must have its code + an .env with a Telegram token.
    for bot in BOTS:
        if not bot.runserver.exists():
            ok = False
            log(f"[{bot.name}] WARN  missing {bot.runserver}")
        env_file = bot.dir / ".env"
        if not env_file.exists():
            ok = False
            log(f"[{bot.name}] WARN  missing {env_file} (Telegram/MT5 creds).")
        elif not _read_env_value(env_file, "TG_BOT_TOKEN"):
            ok = False
            log(f"[{bot.name}] WARN  TG_BOT_TOKEN not set in {env_file}.")

    if _find_source_terminal() is None and not all(b.terminal.exists() for b in BOTS):
        ok = False
        log("WARN  No MetaTrader 5 terminal installed to clone from (see above).")

    log("Preflight: " + ("OK — ready." if ok else "issues found (see WARN lines)."))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# 2) Per-bot child environment (pins the bot to its own terminal via the shim)
# ─────────────────────────────────────────────────────────────────────────────
def build_child_env(bot: Bot) -> Dict[str, str]:
    env = os.environ.copy()
    env["BOT_MT5_TERMINAL"] = str(bot.terminal)            # read by the shim
    env["MT5_PATH"] = str(bot.terminal)                    # belt-and-suspenders for config.py
    # Prepend the isolation shim dir so Python auto-loads our sitecustomize.py.
    env["PYTHONPATH"] = str(SHIM_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["RUNSERVER_MANAGED"] = "1"
    return env


def _creation_flags() -> int:
    if not is_windows():
        return 0
    # New process group so we can deliver CTRL_BREAK to the child without killing
    # ourselves; ABOVE_NORMAL priority for responsiveness.
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) | 0x00008000


# ─────────────────────────────────────────────────────────────────────────────
# 3) Launch + supervise both bot watchdogs
# ─────────────────────────────────────────────────────────────────────────────
def start_bot(bot: Bot) -> bool:
    """Launch one bot watchdog. Returns True on success, False if the launch itself
    failed (so the supervisor can back off instead of re-spinning every 2s)."""
    if not bot.runserver.exists():
        log(f"[{bot.name}] ERROR {bot.runserver} not found — cannot start.")
        return False
    py = root_python()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = bot.console_log.open("ab", buffering=0)
    fh.write((f"\n{'='*70}\nmaster: starting {bot.name} watchdog | {_now()}\n"
              f"terminal: {bot.terminal}\n{'='*70}\n").encode("utf-8", "replace"))
    try:
        bot.proc = subprocess.Popen(
            [str(py), "-u", str(bot.runserver), "--run"],
            cwd=str(bot.dir), env=build_child_env(bot),
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=_creation_flags())
        bot._log_handle = fh  # type: ignore[attr-defined]
        bot.started_at = time.monotonic()
        log(f"[{bot.name}] STARTED watchdog (pid={bot.proc.pid}) -> terminal "
            f"{bot.terminal}")
        return True
    except Exception as exc:
        fh.close()
        bot.proc = None
        log(f"[{bot.name}] ERROR failed to start: {exc}")
        return False


def stop_bot(bot: Bot, timeout: float = 30.0) -> None:
    proc = bot.proc
    if proc is None or proc.poll() is not None:
        return
    log(f"[{bot.name}] stopping watchdog pid={proc.pid} ...")
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
        log(f"[{bot.name}] did not stop in {timeout:.0f}s — killing tree.")
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True)
        except Exception:
            pass
    h = getattr(bot, "_log_handle", None)
    if h is not None:
        try:
            h.close()
        except Exception:
            pass


def _kill_orphan_bots() -> None:
    """Kill any bot watchdog (+ its main.py child tree) left running by a PREVIOUS
    master that died. We are the new master and have not started our own bots yet,
    so anything holding a bot lock port is, by definition, an orphan. This is what
    keeps the self-healing relaunch from ever stacking duplicate bots."""
    for bot in BOTS:
        for pid in _pids_on_port(bot.lock_port):
            log(f"[{bot.name}] cleaning orphan watchdog pid={pid} from a prior master.")
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True)
            except Exception:
                pass


def _master_already_running() -> bool:
    """Quick, silent check: is another master already holding the lock port?
    Returns True if the port is occupied (another master is alive)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if is_windows() and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        s.bind(("127.0.0.1", MASTER_LOCK_PORT))
        s.close()          # port was free — no master running
        return False
    except OSError:
        return True        # port occupied — master alive
    finally:
        try:
            s.close()
        except Exception:
            pass


def supervise() -> int:
    """Keep both bot watchdogs alive forever; restart any that exits (backoff)."""

    # ── FAST single-instance check FIRST ──────────────────────────────────
    # Acquire the lock port before doing ANY work (preflight, logging, etc.)
    # so that duplicate instances spawned by the scheduled task every minute
    # exit instantly and silently — no log spam, no wasted resources.
    guard = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if is_windows() and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            guard.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        guard.bind(("127.0.0.1", MASTER_LOCK_PORT))
        guard.listen(1)
    except OSError:
        # Another master is alive — exit silently (no log, no preflight).
        return 0

    stop_event = threading.Event()

    def _request_stop(signum, _frame):
        log(f"Stop signal received ({signum}).")
        stop_event.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _request_stop)
        except Exception:
            pass
    if hasattr(signal, "SIGBREAK"):
        try:
            signal.signal(signal.SIGBREAK, _request_stop)
        except Exception:
            pass

    _set_high_priority()
    preflight()
    log("=" * 70)
    log(f"MASTER RUNSERVER up. root={ROOT_DIR}  python={root_python()}")
    for bot in BOTS:
        log(f"  - {bot.name:<4} -> {bot.terminal}")
    log("=" * 70)

    started_utc = _now()
    try:
        # We now own the master lock. Clear any orphaned bots from a dead prior
        # master BEFORE starting ours, so the self-healing relaunch never stacks.
        _kill_orphan_bots()
        # Stagger the two starts slightly so their terminals come up cleanly.
        for i, bot in enumerate(BOTS):
            start_bot(bot)
            if i + 1 < len(BOTS):
                stop_event.wait(3.0)

        while not stop_event.is_set():
            for bot in BOTS:
                proc = bot.proc
                if proc is not None and proc.poll() is None:
                    continue  # healthy
                now = time.monotonic()
                if proc is not None:  # it had been running and just exited
                    code = proc.returncode
                    h = getattr(bot, "_log_handle", None)
                    if h is not None:
                        try:
                            h.close()
                        except Exception:
                            pass
                    bot.proc = None
                    bot.restarts += 1
                    # Reset the backoff after a long healthy run (mirrors the per-bot
                    # watchdog) so an isolated crash weeks later restarts promptly
                    # instead of eating the full 60s that early flapping pinned.
                    uptime = (now - bot.started_at) if bot.started_at else 0.0
                    if uptime >= 300.0:
                        bot.restart_delay = 5.0
                    else:
                        bot.restart_delay = min(60.0, max(5.0, bot.restart_delay * 1.5))
                    bot.next_start = now + bot.restart_delay
                    log(f"[{bot.name}] watchdog exited (code={code}). "
                        f"restart #{bot.restarts} in {bot.restart_delay:.0f}s.")
                    continue
                if now >= bot.next_start:  # time to (re)start
                    if not start_bot(bot):
                        # the launch ITSELF failed (missing file, disk full, venv gone
                        # mid-update) — back off instead of re-spinning every 2s and
                        # churning CPU/SSD on the undersized VPS.
                        bot.restart_delay = min(60.0, max(5.0, bot.restart_delay * 1.5))
                        bot.next_start = now + bot.restart_delay
                        log(f"[{bot.name}] start failed — retry in {bot.restart_delay:.0f}s.")
            _write_health(started_utc)
            stop_event.wait(2.0)
    finally:
        log("Shutting down — stopping both bot watchdogs ...")
        for bot in BOTS:
            stop_bot(bot)
        try:
            guard.close()
        except Exception:
            pass

    log("Master runserver stopped.")
    return 0


def _set_high_priority() -> None:
    if not is_windows():
        return
    try:
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(h, 0x00008000)  # ABOVE_NORMAL
    except Exception:
        pass


def _write_health(started_utc: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        snap = {
            "started_utc": started_utc, "last_tick_utc": _now(), "pid": os.getpid(),
            "bots": {b.name: {
                "pid": (b.proc.pid if b.proc and b.proc.poll() is None else None),
                "alive": bool(b.proc and b.proc.poll() is None),
                "restarts": b.restarts, "terminal": str(b.terminal),
            } for b in BOTS},
        }
        HEALTH_FILE.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 4) Verify isolation — prove each bot binds to its OWN terminal (no orders)
# ─────────────────────────────────────────────────────────────────────────────
_PROBE = r'''
import json
import config                      # loads the bot's own .env (login/pwd/server)
import mt5_clent as mc
ok = mc.connect_to_mt5()
import MetaTrader5 as mt5
ti = mt5.terminal_info()
ai = mt5.account_info()
out = {
    "connected": bool(ok),
    "terminal_path": getattr(ti, "path", None),
    "terminal_data_path": getattr(ti, "data_path", None),
    "terminal_connected": getattr(ti, "connected", None),
    "account_login": getattr(ai, "login", None),
    "account_server": getattr(ai, "server", None),
    "symbol": getattr(config, "SYMBOL", None),
}
try:
    mt5.shutdown()
except Exception:
    pass
print("PROBE_RESULT=" + json.dumps(out))
'''


def verify() -> int:
    log("=" * 70)
    log("VERIFY — connecting each bot to its dedicated terminal (read-only) ...")
    log("=" * 70)
    provision_terminals()
    py = root_python()
    results: Dict[str, dict] = {}
    for bot in BOTS:
        log(f"[{bot.name}] probing ...")
        try:
            r = subprocess.run([str(py), "-c", _PROBE], cwd=str(bot.dir),
                               env=build_child_env(bot), capture_output=True,
                               text=True, errors="replace", timeout=180)
        except subprocess.TimeoutExpired:
            log(f"[{bot.name}] ERROR probe timed out (terminal slow to launch?).")
            results[bot.name] = {}
            continue
        data = None
        for line in (r.stdout or "").splitlines():
            if line.startswith("PROBE_RESULT="):
                try:
                    data = json.loads(line[len("PROBE_RESULT="):])
                except Exception:
                    data = None
        if data is None:
            log(f"[{bot.name}] ERROR probe produced no result.")
            if (r.stdout or "").strip():
                log(f"[{bot.name}] stdout tail: {r.stdout.strip()[-600:]}")
            if (r.stderr or "").strip():
                log(f"[{bot.name}] stderr tail: {r.stderr.strip()[-600:]}")
            results[bot.name] = {}
            continue
        results[bot.name] = data
        log(f"[{bot.name}] connected={data['connected']} "
            f"login={data['account_login']} server={data['account_server']} "
            f"symbol={data['symbol']}")
        log(f"[{bot.name}] terminal={data['terminal_path']}")

    # Verdict: both connected AND bound to DIFFERENT terminal paths, each under
    # its own dedicated folder.
    paths = [(results.get(b.name, {}) or {}).get("terminal_path") for b in BOTS]
    all_conn = all((results.get(b.name, {}) or {}).get("connected") for b in BOTS)
    nonempty = [p for p in paths if p]
    distinct = bool(nonempty) and len(nonempty) == len(BOTS) and len(set(nonempty)) == len(BOTS)
    expected = all(
        ((results.get(b.name, {}) or {}).get("terminal_path") or "").lower().startswith(
            str(b.terminal.parent).lower())
        for b in BOTS)
    log("-" * 70)
    if all_conn and distinct and expected:
        log("OK VERIFIED: both bots connected, each to its OWN separate terminal. "
            "No conflict possible.")
        return 0
    if not all_conn:
        log("FAIL not verified: at least one bot failed to connect (check creds / "
            "terminal). See probe output above.")
    elif not distinct:
        log("FAIL not verified: bots resolved to the SAME terminal path — the "
            "isolation shim did not engage.")
    else:
        log("WARN connected to distinct terminals, but a path did not match the "
            "expected dedicated folder. Review the paths above.")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# 5) Autostart (Windows Task Scheduler) + launcher .bat
# ─────────────────────────────────────────────────────────────────────────────
def write_launcher_bat() -> Path:
    # A convenience launcher for starting the master BY HAND (double-click or
    # `runserver_start.bat`). The Windows autostart task does NOT use this file —
    # it runs python.exe directly (see install()), which sidesteps cmd.exe's
    # OEM-code-page handling of the Cyrillic project path entirely.
    #
    # The project path can contain non-ASCII characters, so we bake NO literal
    # path into the file: everything is derived at run time from %~dp0, keeping
    # the file pure ASCII and immune to code-page corruption. Output is teed to
    # Logs\launcher.log so a failed manual start is never silent.
    content = (
        "@echo off\r\n"
        "setlocal\r\n"
        "chcp 65001 >nul\r\n"
        "title All Signal Bots - Master RunServer\r\n"
        'cd /d "%~dp0"\r\n'
        "set PYTHONUNBUFFERED=1\r\n"
        "set PYTHONIOENCODING=utf-8\r\n"
        'set "PY=%~dp0.venv\\Scripts\\python.exe"\r\n'
        'if not exist "%PY%" set "PY=python"\r\n'
        'if not exist "%~dp0Logs" mkdir "%~dp0Logs"\r\n'
        'echo [%date% %time%] launch PY=%PY% >> "%~dp0Logs\\launcher.log"\r\n'
        '"%PY%" "%~dp0runserver.py" --run >> "%~dp0Logs\\launcher.log" 2>&1\r\n'
        'echo [%date% %time%] exited code=%ERRORLEVEL% >> "%~dp0Logs\\launcher.log"\r\n'
        "exit /b %ERRORLEVEL%\r\n"
    )
    # Write as ASCII bytes — guaranteed safe for cmd.exe regardless of locale.
    LAUNCHER_BAT.write_bytes(content.encode("ascii"))
    log(f"Launcher written: {LAUNCHER_BAT}")
    return LAUNCHER_BAT


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT_DIR), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding="utf-8", errors="replace")


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _task_xml() -> str:
    """Task Scheduler XML for ONE self-healing master task.

    Triggers:
      • TimeTrigger — fires on a fixed schedule and repeats EVERY MINUTE forever.
        This is the self-heal: the master holds a single-instance lock (port
        49345); while it is alive each minute's relaunch exits instantly, but the
        moment it dies — crash, kill, OOM, anything — the next minute's launch
        takes over (and first sweeps any orphaned bots). Unlike a logon-trigger
        repetition, a TimeTrigger fires on the clock, so 24/7 recovery does not
        depend on a fresh logon.
      • LogonTrigger + BootTrigger — start promptly at logon / reboot instead of
        waiting up to a minute for the first TimeTrigger tick.

    ExecutionTimeLimit=PT0S removes the default 72-hour cap so the master may run
    indefinitely; IgnoreNew makes the minute ticks no-ops while it is healthy.
    Runs as the logged-on user (InteractiveToken) so the MT5 terminals share that
    desktop session — the standard setup for an always-logged-on trading VPS.
    """
    py = _xml_escape(str(root_python()))
    args = _xml_escape(f'"{ROOT_DIR / "runserver.py"}" --run')
    workdir = _xml_escape(str(ROOT_DIR))
    user = _xml_escape((os.environ.get("USERDOMAIN", "") + "\\" +
                        os.environ.get("USERNAME", "")).strip("\\"))
    return (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
        '  <RegistrationInfo>\n'
        '    <Description>All Signal Bots master (BTC+XAU) — self-healing 24/7</Description>\n'
        '  </RegistrationInfo>\n'
        '  <Triggers>\n'
        '    <TimeTrigger>\n'
        '      <StartBoundary>2024-01-01T00:00:00</StartBoundary>\n'
        '      <Enabled>true</Enabled>\n'
        '      <Repetition><Interval>PT1M</Interval>'
        '<StopAtDurationEnd>false</StopAtDurationEnd></Repetition>\n'
        '    </TimeTrigger>\n'
        '    <LogonTrigger><Enabled>true</Enabled></LogonTrigger>\n'
        '    <BootTrigger><Enabled>true</Enabled></BootTrigger>\n'
        '  </Triggers>\n'
        '  <Principals>\n    <Principal id="Author">\n'
        f'      <UserId>{user}</UserId>\n'
        '      <LogonType>InteractiveToken</LogonType>\n'
        '      <RunLevel>HighestAvailable</RunLevel>\n'
        '    </Principal>\n  </Principals>\n'
        '  <Settings>\n'
        '    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n'
        '    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n'
        '    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n'
        '    <AllowHardTerminate>true</AllowHardTerminate>\n'
        '    <StartWhenAvailable>true</StartWhenAvailable>\n'
        '    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>\n'
        '    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>\n'
        '    <Enabled>true</Enabled>\n'
        '    <Hidden>false</Hidden>\n'
        '    <RestartOnFailure><Interval>PT1M</Interval><Count>999</Count></RestartOnFailure>\n'
        '  </Settings>\n'
        '  <Actions Context="Author">\n    <Exec>\n'
        f'      <Command>{py}</Command>\n'
        f'      <Arguments>{args}</Arguments>\n'
        f'      <WorkingDirectory>{workdir}</WorkingDirectory>\n'
        '    </Exec>\n  </Actions>\n</Task>\n'
    )


def _create_master_task(name: str) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    xml_path = LOG_DIR / f"_task_{name}.xml"
    xml_path.write_text(_task_xml(), encoding="utf-16")  # schtasks /XML wants UTF-16
    rc = _run(["schtasks", "/Create", "/TN", name, "/XML", str(xml_path),
               "/F"]).returncode
    try:
        xml_path.unlink()
    except Exception:
        pass
    return rc == 0


SERVER_TIMEZONE = "W. Europe Standard Time"   # Frankfurt/Berlin — CET/CEST, DST-auto


def ensure_clock() -> None:
    """Set the server timezone to Frankfurt AND force an NTP resync. The bots' sessions
    read the BROKER tick clock (timezone-independent, absorbs <=30 min of OS skew), so
    the timezone itself is cosmetic — but an ACCURATE clock matters, and Frankfurt makes
    the logs read in the FX/metals trading day. Needs Administrator; without it we warn
    and continue (Windows' own w32time usually keeps the clock close enough)."""
    if not is_windows():
        return
    if not is_admin():
        log("WARN  Not Administrator — timezone/NTP NOT changed. Run `--install` ONCE "
            "'As administrator' to force Frankfurt time + NTP. (Sessions still work — "
            "they use the broker clock — but keep the OS clock NTP-synced.)")
        return
    try:
        current = _run(["tzutil", "/g"]).stdout.strip()
    except Exception:
        current = ""
    if current != SERVER_TIMEZONE:
        if _run(["tzutil", "/s", SERVER_TIMEZONE]).returncode == 0:
            log(f"Timezone set to FRANKFURT ({SERVER_TIMEZONE}).")
        else:
            log(f"WARN  tzutil could not set {SERVER_TIMEZONE}.")
    else:
        log(f"Timezone already Frankfurt ({SERVER_TIMEZONE}).")
    # force an accurate clock (the part that actually matters)
    _run(["sc", "config", "w32time", "start=", "auto"])
    _run(["net", "start", "w32time"])
    if _run(["w32tm", "/resync", "/force"]).returncode == 0:
        log("NTP resync OK — clock is accurate.")
    else:
        log("WARN  NTP resync non-zero (peer unreachable now?) — w32time will keep retrying.")


def install(start_now: bool) -> int:
    if not is_windows():
        log("--install is Windows-only.")
        return 1
    preflight()
    ensure_clock()                # Frankfurt timezone + NTP (Administrator); non-fatal
    provision_terminals()
    write_launcher_bat()  # convenience for manual starts; the task runs python directly

    ok = _create_master_task(LOGON_TASK_NAME)
    log(f"Autostart task {LOGON_TASK_NAME} (self-healing, every minute): "
        f"{'created' if ok else 'FAILED'}")
    if not ok:
        log("ERROR Could not create the scheduled task (need rights?).")
        return 1
    log("OK Install complete. The master auto-starts at logon/boot and self-heals "
        "within ~1 minute of ANY stoppage, 24/7.")
    return supervise() if start_now else 0


def _task_exists(name: str) -> bool:
    return is_windows() and _run(["schtasks", "/Query", "/TN", name]).returncode == 0


def kill_switch() -> int:
    log("=" * 70)
    log("KILL SWITCH — stopping master + BOTH bots, removing all autostart tasks.")
    log("=" * 70)

    # 1) Stop & remove the master's own scheduled tasks + launcher.
    if is_windows():
        for name in (LOGON_TASK_NAME, BOOT_TASK_NAME):
            if _task_exists(name):
                _run(["schtasks", "/End", "/TN", name])
                _run(["schtasks", "/Delete", "/TN", name, "/F"])
                log(f"Removed task {name}.")
    if LAUNCHER_BAT.exists():
        try:
            LAUNCHER_BAT.unlink()
        except Exception:
            pass

    # 2) Kill any running master (holds the lock port) + its children.
    killed = 0
    for pid in _pids_on_port(MASTER_LOCK_PORT):
        if _run(["taskkill", "/PID", str(pid), "/T", "/F"]).returncode == 0:
            killed += 1

    # 3) Delegate to each bot's OWN kill switch (uninstalls its tasks + kills its
    #    watchdog/main.py tree). This reuses the bots' tested teardown verbatim.
    py = root_python()
    for bot in BOTS:
        if bot.runserver.exists():
            log(f"[{bot.name}] invoking its kill switch ...")
            r = _run([str(py), str(bot.runserver), "--kill"])
            if (r.stdout or "").strip():
                for ln in r.stdout.strip().splitlines()[-6:]:
                    log(f"[{bot.name}]   {ln}")

    log(f"Kill switch complete. master processes killed={killed}. "
        f"To start again: python runserver.py --install --start-now")
    return 0


def _pids_on_port(port: int) -> set:
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
    except Exception:
        pass
    return pids


def status() -> int:
    print(f"Master root : {ROOT_DIR}")
    print(f"Python      : {root_python()}")
    print(f"Shim        : {SHIM_DIR / 'sitecustomize.py'} "
          f"({'ok' if (SHIM_DIR / 'sitecustomize.py').exists() else 'MISSING'})")
    print(f"Launcher    : {LAUNCHER_BAT} ({'exists' if LAUNCHER_BAT.exists() else 'missing'})")
    print(f"Admin       : {is_admin()}")
    for bot in BOTS:
        print(f"\n[{bot.name}]")
        print(f"  bot dir   : {bot.dir}")
        print(f"  terminal  : {bot.terminal} "
              f"({'ready' if bot.terminal.exists() else 'NOT provisioned'})")
        print(f"  console   : {bot.console_log} "
              f"({'exists' if bot.console_log.exists() else 'none yet'})")
    if HEALTH_FILE.exists():
        print("\n[health]")
        try:
            print(HEALTH_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    if is_windows():
        for name in (LOGON_TASK_NAME, BOOT_TASK_NAME):
            r = _run(["schtasks", "/Query", "/TN", name, "/FO", "LIST"])
            print(f"\n[{name}]")
            print(r.stdout.strip() if r.stdout.strip() else "not registered")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Master RunServer — runs BOTH bots side-by-side, each on its "
                    "own MT5 terminal, conflict-free.")
    p.add_argument("--install", action="store_true",
                   help="Provision terminals + register Windows autostart tasks.")
    p.add_argument("--start-now", action="store_true",
                   help="With --install, also start running immediately.")
    p.add_argument("--run", action="store_true",
                   help="Provision (if needed) + run/supervise both bots (foreground).")
    p.add_argument("--setup-only", action="store_true",
                   help="Only provision the two terminals and exit.")
    p.add_argument("--verify", action="store_true",
                   help="Prove each bot binds to its OWN terminal (read-only), then exit.")
    p.add_argument("--status", action="store_true", help="Print status and exit.")
    p.add_argument("--kill", action="store_true",
                   help="KILL SWITCH: stop master + both bots, remove autostart.")
    p.add_argument("--set-timezone", action="store_true",
                   help="Set server timezone to Frankfurt + force NTP sync (Admin), then exit.")
    return p.parse_args(list(argv))


def _ensure_std_streams() -> None:
    """Headless (Task Scheduler) processes can have sys.stdout/stderr == None.
    Point them at a real file so no stray print() anywhere can crash the master."""
    import io
    need = any(getattr(sys, n, None) is None for n in ("stdout", "stderr"))
    if not need:
        return
    fh = None
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = open(LOG_DIR / "master_console.log", "a", encoding="utf-8", errors="replace")
    except Exception:
        fh = io.StringIO()
    for n in ("stdout", "stderr"):
        if getattr(sys, n, None) is None:
            setattr(sys, n, fh)


def main(argv=None) -> int:
    _ensure_std_streams()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.kill:
        return kill_switch()
    if args.status:
        return status()
    if args.verify:
        return verify()
    if args.setup_only:
        provision_terminals()
        return 0
    if args.set_timezone:
        ensure_clock()
        return 0
    if args.install:
        return install(start_now=args.start_now)
    # default and --run: provision (idempotent) then supervise both bots.
    # Fast bail-out: if another master is already alive, skip the expensive
    # provision_terminals() + preflight and exit immediately and silently.
    if _master_already_running():
        return 0
    provision_terminals()
    return supervise()


if __name__ == "__main__":
    raise SystemExit(main())
