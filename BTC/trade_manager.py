import json
import math
import os

import MetaTrader5 as mt5

import config
from mt5_clent import _locked, _pick_filling_mode, close_position   # shared helpers (single source)

MAGIC = config.MAGIC

_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logs")
_STATE_FILE = os.path.join(_STATE_DIR, "trade_state.json")

# ticket -> {entry, orig_sl, orig_vol, partial_done, be_done, price_ema}
# PERSISTED to disk: the watchdog auto-restarts main.py on any crash, and an
# in-memory-only state would lose each position's ORIGINAL risk basis (orig_sl).
# After a restart that re-derives orig_sl from the already-moved (break-even /
# trailed) stop, R collapses to ~0 -> a 2nd 50% partial fires AND trailing snaps
# the SL right under price (instant stop-out). Persisting orig_sl/partial_done/
# be_done fixes that: the risk basis is written once at first sight and reused
# verbatim across every restart.
_state = {}
_state_loaded = False


def _load_state():
    """Restore per-ticket management state from disk (once per process)."""
    global _state, _state_loaded
    if _state_loaded:
        return
    _state_loaded = True
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # JSON object keys are strings -> restore integer ticket keys.
        _state = {int(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        _state = {}


def _save_state():
    """Atomically persist _state (tmp + os.replace) so a crash can't corrupt it."""
    try:
        os.makedirs(_STATE_DIR, exist_ok=True)
        tmp = _STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_state, f, ensure_ascii=False)
        os.replace(tmp, _STATE_FILE)
    except Exception:                      # persistence is best-effort — never crash the loop
        pass


_OUR_MAGICS = {MAGIC, *getattr(config, "LEGACY_MAGICS", [])}


def _our_positions(symbol=None):
    pos = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    # ONLY the bot's own orders (current + legacy magic) — manual trades (magic 0)
    # and the OTHER bot's trades are deliberately left untouched. manage_positions
    # always passes this bot's symbol, so legacy adoption can't reach across symbols.
    return [p for p in (pos or []) if p.magic in _OUR_MAGICS]


@_locked
def modify_sl(ticket, new_sl):
    """Set a position's stop-loss (keeps its current TP). Clamps to the broker's
    minimum stop distance on the correct side of the live price (so a too-close
    break-even/trail SL is accepted, not silently rejected) and NEVER loosens past
    the existing stop. Returns True on success. None-safe on symbol/tick."""
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return False
    p = pos[0]
    si = mt5.symbol_info(p.symbol)
    tick = mt5.symbol_info_tick(p.symbol)
    if si is None or tick is None:
        return False
    point = si.point or 0.01
    gap = max(max(getattr(si, "trade_stops_level", 0) or 0, 0) * point, point)
    new_sl = float(new_sl)
    cur = p.sl if (p.sl and p.sl > 0) else None
    if p.type == mt5.POSITION_TYPE_BUY:
        new_sl = min(new_sl, tick.bid - gap)     # keep a broker-valid distance below bid
        if cur is not None:
            new_sl = max(new_sl, cur)            # never loosen below the current stop
    else:
        new_sl = max(new_sl, tick.ask + gap)     # broker-valid distance above ask
        if cur is not None:
            new_sl = min(new_sl, cur)
    new_sl = round(new_sl, si.digits)
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": p.symbol,
        "position": ticket,
        "sl": new_sl,
        "tp": p.tp,
        "magic": MAGIC,
    }
    r = mt5.order_send(req)
    return r is not None and r.retcode == mt5.TRADE_RETCODE_DONE


@_locked
def partial_close(ticket, pct):
    """Close pct% of a position's volume (leaving a valid remainder). Returns dict."""
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return {"ok": False, "error": "not found"}
    p = pos[0]
    si = mt5.symbol_info(p.symbol)
    tick = mt5.symbol_info_tick(p.symbol)
    if si is None or tick is None:
        return {"ok": False, "error": "no symbol/tick info"}
    
    # Safe fallback for step and min volume in Crypto
    step = si.volume_step if (si.volume_step and si.volume_step > 0) else 0.01
    vmin = si.volume_min if (si.volume_min and si.volume_min > 0) else 0.01

    raw = p.volume * pct / 100.0
    vol = math.floor(raw / step) * step
    
    vol_digits = max(2, len(str(step).rstrip('0').split('.')[-1])) if '.' in str(step) else 2
    vol = round(vol, vol_digits)
    vol = max(vmin, vol)
    
    if vol <= 0:                                        # floor reduced it to zero
        return {"ok": False, "error": "volume too small after rounding"}
    
    # Check if remainder is valid for Bitcoin
    remainder = round(p.volume - vol, vol_digits)
    if remainder < vmin and remainder > 0:                    # can't leave a tradable remainder -> skip
        return {"ok": False, "error": "remainder too small"}

    ctype = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol, "volume": float(vol),
        "type": ctype, "position": ticket, "price": price,
        "deviation": config.MAX_DEVIATION_POINTS,  # wide on EXIT — guarantee the fill
        "magic": MAGIC, "comment": "partial TP", "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _pick_filling_mode(si),
    }
    r = mt5.order_send(req)
    if r is None or r.retcode != mt5.TRADE_RETCODE_DONE:
        return {"ok": False, "error": f"retcode={None if r is None else r.retcode}"}
    return {"ok": True, "closed": vol}


@_locked
def manage_positions(symbol=None):
    """Apply partial-TP / break-even / trailing to all our open positions."""
    if not config.ENABLE_TRADE_MGMT:
        return []
    _load_state()                          # restore risk basis (survives restarts)
    actions = []
    dirty = False
    positions = _our_positions(symbol)
    open_tickets = set()

    for p in positions:
        open_tickets.add(p.ticket)
        st = _state.get(p.ticket)
        if st is None:
            # coerce to native float so JSON persistence can never fail on a type
            st = {"entry": float(p.price_open), "orig_sl": float(p.sl),
                  "orig_vol": float(p.volume), "partial_done": False,
                  "be_done": False, "price_ema": float(p.price_open)}
            _state[p.ticket] = st
            dirty = True

        R = abs(st["entry"] - st["orig_sl"])
        if R <= 0:                     
            continue

        tick = mt5.symbol_info_tick(p.symbol)
        if tick is None:
            continue
        is_buy = p.type == mt5.POSITION_TYPE_BUY
        price = tick.bid if is_buy else tick.ask

        spread = tick.ask - tick.bid
        # Crypto Spread Buffer
        spread_buffer = spread * 0.5    

        profit = (price - st["entry"]) if is_buy else (st["entry"] - price)
        r_mult = profit / R

        # ── 0. Time-stop ── a stale signal that hasn't reached TP/SL has lost its
        # momentum; harvest the small profit fast before it reverts. tick.time and
        # p.time are both broker-server epochs, so the age is offset-independent.
        if config.TIME_STOP_MIN > 0 and getattr(p, "time", 0):
            age_min = (tick.time - p.time) / 60.0
            if age_min >= config.TIME_STOP_MIN and p.profit > config.TIME_STOP_MIN_PROFIT_USD:
                res = close_position(p.ticket)
                if res.get("ok"):
                    actions.append(f"time-stop #{p.ticket} @ {age_min:.0f}min (+${p.profit:.2f})")
                    continue                       # done with this ticket
                # if the close failed, fall through to normal management

        # ── 1. Partial TP ──
        if (config.PARTIAL_TP_R > 0 and not st["partial_done"]
                and r_mult >= config.PARTIAL_TP_R):
            res = partial_close(p.ticket, config.PARTIAL_TP_PCT)
            if res.get("ok"):
                st["partial_done"] = True
                _save_state()              # persist NOW — partial is irreversible (no double-take on restart)
                actions.append(f"partial {res['closed']} on #{p.ticket} @ {r_mult:.1f}R")

        # ── 2. Break-even ──
        if (config.BREAKEVEN_AT_R > 0 and not st["be_done"]
                and r_mult >= config.BREAKEVEN_AT_R):
            buf = config.BREAKEVEN_BUFFER_USD
            if is_buy:
                be = st["entry"] + buf + spread_buffer   # spread-aware BE
                improves = be > p.sl
            else:
                be = st["entry"] - buf - spread_buffer   # spread-aware BE
                improves = p.sl > 0 and be < p.sl
            if improves and modify_sl(p.ticket, be):
                st["be_done"] = True
                dirty = True
                actions.append(f"break-even #{p.ticket} -> {be:.2f}")

        # ── 3. Trailing stop ──
        if config.TRAIL_START_R > 0 and r_mult >= config.TRAIL_START_R:
            smooth_alpha = 2.0 / (config.TRAIL_SMOOTH_BARS + 1)
            st["price_ema"] = st["price_ema"] + smooth_alpha * (price - st["price_ema"])
            smoothed = st["price_ema"]
            dirty = True                   # persist the smoothed trail anchor (continuous across restarts)

            if is_buy:
                new_sl = smoothed - config.TRAIL_R * R - spread_buffer
            else:
                new_sl = smoothed + config.TRAIL_R * R + spread_buffer

            cur = mt5.positions_get(ticket=p.ticket)
            cur_sl = cur[0].sl if cur else p.sl

            if is_buy:
                improves = new_sl > cur_sl
            else:
                improves = cur_sl > 0 and new_sl < cur_sl

            if improves and modify_sl(p.ticket, new_sl):
                actions.append(f"trail #{p.ticket} -> {new_sl:.2f} ({r_mult:.1f}R)")

    # forget closed tickets
    for t in [t for t in _state if t not in open_tickets]:
        _state.pop(t, None)
        dirty = True
    if dirty:
        _save_state()
    return actions