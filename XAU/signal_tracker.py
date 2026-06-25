"""
═══════════════════════════════════════════════════════════════════════════
  SIGNAL SCORECARD  —  forward paper-test on the REAL market (no money risk)
═══════════════════════════════════════════════════════════════════════════

Every signal that EITHER engine version (v1 classic / v2 math-gated) would fire
is recorded as a *paper* trade and then graded against the real M1 price path:
did price touch TP or SL first? This is how we MEASURE signal quality WITHOUT a
historical backtest — it is pure forward observation of live ticks.

Honesty rules baked in (so the numbers can be trusted):
  • Strictly causal: a paper trade is only evaluated against M1 bars that formed
    AFTER the signal — never the signal bar or earlier. No look-ahead.
  • Pessimistic ties: if a single M1 bar's range touches BOTH SL and TP, we
    assume the SL was hit first (the worst case).
  • Cost haircut: every trade pays SCORECARD_COST_R (spread+commission, in R).
  • It NEVER sends an order. It is observation only.

This module is imported and driven by the live loop, so — like trade_manager —
it is part of the running system and persists between sessions.
"""

import json
import os
import threading
import time

import MetaTrader5 as mt5

import config
import mt5_clent as mc

_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logs")
_STATE_FILE = os.path.join(_LOGS_DIR, "scorecard_open.json")     
_RESULTS_FILE = os.path.join(_LOGS_DIR, "scorecard.jsonl")  


_io_lock = threading.RLock()


# ───────────────────────────────────────────────────────────────────────────
#  STATE  (open paper trades + per-zone last-record times) — survives restarts
# ───────────────────────────────────────────────────────────────────────────
def _load():
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            st = json.load(f)
        st.setdefault("open", [])
        st.setdefault("last_rec", {})
        st.setdefault("seq", 0)
        return st
    except (OSError, json.JSONDecodeError, ValueError):
        return {"open": [], "last_rec": {}, "seq": 0}


def _save(st):
    """Atomically persist state. Returns True on success — the caller only writes
    result rows AFTER a successful save so a save failure can never re-grade a
    resolved trade into a duplicate row."""
    try:
        os.makedirs(_LOGS_DIR, exist_ok=True)
        tmp = _STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False)
        os.replace(tmp, _STATE_FILE)            
        return True
    except OSError:
        return False


def _append_result(row):
    try:
        os.makedirs(_LOGS_DIR, exist_ok=True)
        with open(_RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _zone_key(version, direction, entry, scale):
    bucket = round(entry / max(scale * 0.5, 0.01))
    return f"{version}_{direction}_{bucket}"


def _fetch_m1():
    """Latest M1 bars (structured array) — enough to cover the timeout window."""
    try:
        bars = max(200, int(config.SCORECARD_TIMEOUT_MIN) + 40)
        return mc.GetRealXauData(num_bars=bars).get_data(specific_tf=mt5.TIMEFRAME_M1)
    except Exception:
        return None


# ───────────────────────────────────────────────────────────────────────────
#  RECORD  (open new paper trades from a signal dict's A/B views)
# ───────────────────────────────────────────────────────────────────────────
def _record(st, signal_dict, raw):
    ab = signal_dict.get("ab") or {}
    if not ab:
        return
    opened_epoch = float(raw[-1]["time"]) if (raw is not None and len(raw)) else None
    now_wall = time.time()

    for ver, v in ab.items():
        try:
            if v.get("signal") not in ("BUY", "SELL"):
                continue
            entry, sl, tp = v.get("close_price"), v.get("sl"), v.get("tp")
            if entry is None or sl is None or tp is None or not (entry > 0):
                continue
            direction = v["signal"]
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk <= 0 or reward <= 0:
                continue
            atr = v.get("atr") or risk
            key = _zone_key(ver, direction, entry, atr)

            # dedupe: skip if a same-zone paper trade is already open for this version
            if any(o.get("version") == ver and o.get("key") == key for o in st["open"]):
                continue
            # ...or if we recorded this zone within the cooldown (avoids 10s re-fires)
            if now_wall - float(st["last_rec"].get(key, 0)) < config.SCORECARD_DEDUPE_MIN * 60:
                continue

            st["seq"] += 1
            conf = v.get("buy_conf", 0) if direction == "BUY" else v.get("sell_conf", 0)
            st["open"].append({
                "id": st["seq"], "version": ver, "key": key, "direction": direction,
                "entry": round(float(entry), 3), "sl": round(float(sl), 3),
                "tp": round(float(tp), 3), "risk": round(float(risk), 3),
                "reward": round(float(reward), 3), "conf": int(conf),
                "session": v.get("session", "?"),
                "opened_wall": now_wall, "opened_epoch": opened_epoch,
            })
            st["last_rec"][key] = now_wall
        except Exception:
            continue

    # keep last_rec from growing forever
    if len(st["last_rec"]) > 2000:
        cutoff = now_wall - 24 * 3600
        st["last_rec"] = {k: t for k, t in st["last_rec"].items() if t > cutoff}


# ───────────────────────────────────────────────────────────────────────────
#  EVALUATE  (grade open paper trades against the real M1 path — first touch)
# ───────────────────────────────────────────────────────────────────────────
def _resolve(sig, raw, now_wall):
    """Return a result row if SL/TP/timeout reached, else None (still open)."""
    direction = sig["direction"]
    entry, sl, tp, risk = sig["entry"], sig["sl"], sig["tp"], sig["risk"]

    anchor = sig.get("opened_epoch")
    if anchor is None:
        ow = sig.get("opened_wall")
        if ow is None:
            return None                               # un-gradeable -> keep open
        try:
            anchor = float(ow) + mc.get_server_utc_offset_hours() * 3600.0
        except Exception:
            return None

    outcome = exit_price = None
    for row in raw:
        t = float(row["time"])
        if t <= anchor:
            continue                                  # only bars AFTER the signal
        hi, lo = float(row["high"]), float(row["low"])
        if direction == "BUY":
            sl_hit, tp_hit = (lo <= sl), (hi >= tp)
        else:
            sl_hit, tp_hit = (hi >= sl), (lo <= tp)
        if sl_hit:                                    # ties -> SL first (pessimistic)
            outcome, exit_price = "sl", sl
            break
        if tp_hit:
            outcome, exit_price = "tp", tp
            break

    if outcome is None:
        elapsed_min = (now_wall - float(sig["opened_wall"])) / 60.0
        if elapsed_min >= config.SCORECARD_TIMEOUT_MIN and raw is not None and len(raw):
            outcome, exit_price = "timeout", float(raw[-1]["close"])
        else:
            return None

    realized = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
    r_mult = realized / risk - config.SCORECARD_COST_R     # net of spread/commission
    return {
        "id": sig["id"], "version": sig["version"], "direction": direction,
        "entry": entry, "sl": sl, "tp": tp, "exit": round(float(exit_price), 3),
        "outcome": outcome, "R": round(float(r_mult), 3), "conf": sig.get("conf", 0),
        "session": sig.get("session", "?"),
        "opened_wall": sig["opened_wall"], "closed_wall": now_wall,
        "duration_min": round((now_wall - float(sig["opened_wall"])) / 60.0, 1),
    }


def _evaluate(st, raw):
    """Resolve open paper trades against the M1 path. Returns the list of result
    rows (does NOT write them) so the caller can persist the open-list FIRST and
    only then append results — guaranteeing no resolved trade is graded twice."""
    if raw is None or len(raw) < 2:
        return []
    now_wall = time.time()
    still_open, resolved = [], []
    for sig in st["open"]:
        try:
            res = _resolve(sig, raw, now_wall)
        except Exception:
            res = None
        if res is None:
            still_open.append(sig)
        else:
            resolved.append(res)
    st["open"] = still_open
    return resolved


# ───────────────────────────────────────────────────────────────────────────
#  PUBLIC: called once per loop cycle by main.py — never raises into the loop
# ───────────────────────────────────────────────────────────────────────────
def update(signal_dict):
    """Evaluate open paper trades, then record any new ones from `signal_dict`."""
    if not config.SCORECARD_ENABLED:
        return
    try:
        with _io_lock:
            raw = _fetch_m1()
            st = _load()
            resolved = _evaluate(st, raw)
            if (raw is not None and len(raw)
                    and signal_dict and isinstance(signal_dict, dict)):
                _record(st, signal_dict, raw)
            if _save(st):
                for row in resolved:
                    _append_result(row)
    except Exception as e:
        print(f"⚠️ scorecard update error: {e}")


# ───────────────────────────────────────────────────────────────────────────
#  STATS  (read the resolved-trade log, compute real metrics, format report)
# ───────────────────────────────────────────────────────────────────────────
def _read_results():
    rows = []
    try:
        with open(_RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def _metrics(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    Rs = [float(r.get("R", 0)) for r in rows]
    wins = [r for r in Rs if r > 0]
    losses = [r for r in Rs if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    # max drawdown on the R equity curve
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    cons_loss = max_cons_loss = 0
    for r in Rs:
        eq += r
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
        if r <= 0:
            cons_loss += 1
            max_cons_loss = max(max_cons_loss, cons_loss)
        else:
            cons_loss = 0
    return {
        "n": n,
        "tp": sum(1 for r in rows if r.get("outcome") == "tp"),
        "sl": sum(1 for r in rows if r.get("outcome") == "sl"),
        "timeout": sum(1 for r in rows if r.get("outcome") == "timeout"),
        "win_rate": round(100.0 * len(wins) / n, 1),
        "expectancy_R": round(sum(Rs) / n, 3),
        "total_R": round(sum(Rs), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "avg_win_R": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss_R": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "max_dd_R": round(max_dd, 2),
        "max_cons_loss": max_cons_loss,
    }


def stats():
    """Return {'v1': metrics, 'v2': metrics, 'open': n}."""
    with _io_lock:                          
        rows = _read_results()
        try:
            open_n = len(_load()["open"])
        except Exception:
            open_n = 0
    out = {}
    versions = sorted({r.get("version") for r in rows if r.get("version")}) or ["live"]
    for ver in versions:
        out[ver] = _metrics([r for r in rows if r.get("version") == ver])
    out["open"] = open_n
    out["_versions"] = versions
    return out


def _fmt_block(name, m):
    if m.get("n", 0) == 0:
        return f"*{name}*: пока нет данных (0 сделок)"
    pf = "∞" if m["profit_factor"] == float("inf") else f"{m['profit_factor']}"
    return (
        f"*{name}*  ({m['n']} сделок)\n"
        f"  ✅ Win-rate: *{m['win_rate']}%*   |   TP:{m['tp']} · SL:{m['sl']} · TO:{m['timeout']}\n"
        f"  📈 Ожидание: *{m['expectancy_R']}R*/сделку   |   Σ {m['total_R']}R\n"
        f"  ⚖️ Profit factor: *{pf}*   |   ср.+{m['avg_win_R']}R / {m['avg_loss_R']}R\n"
        f"  📉 Макс. просадка: {m['max_dd_R']}R   |   серия убытков: {m['max_cons_loss']}"
    )


def format_report():
    """Telegram-ready scorecard for the single live engine (forward paper-test)."""
    s = stats()
    versions = s.get("_versions", ["live"])
    blocks = [_fmt_block("📈 Движок XAU", s[v]) for v in versions]
    main_m = s.get("live") or (s[versions[0]] if versions else {"n": 0})
    n = main_m.get("n", 0)
    if n >= 20:
        pos = main_m.get("expectancy_R", 0) > 0
        sign = "🟢" if pos else "🔴"
        verdict = (f"\n{sign} Ожидание: *{main_m['expectancy_R']}R*/сделку на {n} сделках "
                   f"— {'положительное преимущество' if pos else 'пока отрицательное'}.")
    else:
        verdict = f"\n⏳ Для надёжного вывода нужно ≥20 сделок (сейчас {n})."
    return (
        f"📊 *SCORECARD — форвард-тест на реальном рынке*\n"
        f"🔓 Открыто сделок: {s['open']}\n"
        f"{'─' * 28}\n"
        + "\n\n".join(blocks)
        + f"\n{verdict}\n\n"
        f"_R = единица риска. +1R = прибыль размером в один SL._\n"
        f"_Ожидание > 0 ⇒ положительное преимущество._"
    )





