"""
═══════════════════════════════════════════════════════════════════════════
  MT5 CONNECTION & DATA ENGINE — BITCOIN (24/7 Crypto Version)
═══════════════════════════════════════════════════════════════════════════
"""

import functools
import math
import os
import subprocess
import threading
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

import config

_last_terminal_restart = 0.0     # wall-clock of the last hard terminal restart

SYMBOL = config.SYMBOL  # resolved/overwritten by resolve_symbol()

_SERVER_UTC_OFFSET_H = None
_SERVER_UTC_OFFSET_TS = 0.0
_OFFSET_TTL_SEC = 1800          # 30 min


mt5_lock = threading.RLock()


def _locked(fn):
    @functools.wraps(fn)
    def _wrap(*a, **k):
        with mt5_lock:
            return fn(*a, **k)
    return _wrap


# ───────────────────────────────────────────────────────────────────────────
#  CONNECTION
# ───────────────────────────────────────────────────────────────────────────
@_locked
def connect_to_mt5():
    try:
        # BIND TO THIS BOT'S OWN TERMINAL FIRST (by explicit path) so two bots on
        # one machine can NEVER cross-attach to each other's terminal. A no-path
        # initialize() would connect to whichever terminal the library finds first
        # — non-deterministic with 2 terminals running. initialize(path) connects
        # to that exact instance if running, or launches it. Fall back to a no-path
        # attach (single-terminal setups), then to explicit creds — but only with a
        # real MT5_LOGIN, so we never call initialize/login with login=0.
        ok = False
        if config.MT5_PATH:
            ok = mt5.initialize(path=config.MT5_PATH)
        if not ok:
            ok = mt5.initialize()
        if not ok and config.MT5_PASSWORD and config.MT5_LOGIN:
            kw = {"path": config.MT5_PATH} if config.MT5_PATH else {}
            ok = mt5.initialize(login=config.MT5_LOGIN,
                                password=config.MT5_PASSWORD,
                                server=config.MT5_SERVER, **kw)
        if not ok:
            print(f"❌ initialize() failed: {mt5.last_error()}  "
                  f"(set MT5_PATH in .env to bind/auto-launch this bot's own terminal)")
            return False

        info = mt5.account_info()
        if info is not None:
            if config.MT5_PASSWORD and config.MT5_LOGIN and info.login != config.MT5_LOGIN:
                if not mt5.login(login=config.MT5_LOGIN,
                                 password=config.MT5_PASSWORD,
                                 server=config.MT5_SERVER):
                    print(f"❌ Login switch failed #{config.MT5_LOGIN}: {mt5.last_error()}")
                    return False
            return True

        if config.MT5_PASSWORD and config.MT5_LOGIN:
            if mt5.login(login=config.MT5_LOGIN,
                         password=config.MT5_PASSWORD,
                         server=config.MT5_SERVER):
                return True
            print(f"❌ Login failed #{config.MT5_LOGIN}: {mt5.last_error()}")
            return False

        print("❌ Terminal not logged in and no MT5_LOGIN/MT5_PASSWORD configured.")
        return False
    except Exception as e:
        print(f"❌ Exception during MT5 connection: {e}")
        return False


def resolve_symbol():
    """Pick the broker's Bitcoin symbol; cache into module + config SYMBOL."""
    global SYMBOL
    SYMBOL = "BTCUSDm" # Оптимизатсия барои Биткоин
    info = mt5.symbol_info(SYMBOL)
    if info is not None:
        mt5.symbol_select(SYMBOL, True)
        return SYMBOL
    return None


def get_server_utc_offset_hours():
    """Broker-server clock offset from real UTC, in whole hours."""
    global _SERVER_UTC_OFFSET_H, _SERVER_UTC_OFFSET_TS
    now = datetime.now(timezone.utc).timestamp()
    if _SERVER_UTC_OFFSET_H is not None and (now - _SERVER_UTC_OFFSET_TS) < _OFFSET_TTL_SEC:
        return _SERVER_UTC_OFFSET_H
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick and tick.time:
            off = int(round((tick.time - now) / 3600.0))
            if -12 <= off <= 14:
                _SERVER_UTC_OFFSET_H = off
                _SERVER_UTC_OFFSET_TS = now
                return off
    except Exception:
        pass
    return _SERVER_UTC_OFFSET_H if _SERVER_UTC_OFFSET_H is not None else 0


@_locked
def get_balance() -> float:
    try:
        info = mt5.account_info()
        return float(info.balance) if info else 0.0
    except Exception:
        return 0.0


@_locked
def value_per_price_unit():
    """USD P/L per 1.0 of price movement, for 1.0 lot of SYMBOL.

    Reads the broker's real tick value/size so risk sizing matches what set_sl
    actually uses — instead of assuming CONTRACT_SIZE. Falls back to CONTRACT_SIZE
    only if the broker doesn't expose tick metadata."""
    try:
        si = mt5.symbol_info(SYMBOL)
        if si is not None:
            tick_size = getattr(si, "trade_tick_size", 0) or si.point
            tick_value = (getattr(si, "trade_tick_value", 0)
                          or getattr(si, "trade_tick_value_profit", 0))
            if tick_size and tick_value:
                return float(tick_value) / float(tick_size)
    except Exception:
        pass
    return float(config.CONTRACT_SIZE)


@_locked
def get_account_info():
    try:
        info = mt5.account_info()
        if not info:
            return None
        return {
            "login": info.login, "server": info.server, "balance": info.balance,
            "equity": info.equity, "currency": info.currency,
            "trade_mode": info.trade_mode,
            "leverage": info.leverage,
        }
    except Exception:
        return None


_isolation_reported = False


@_locked
def isolation_report():
    """Log (ONCE per process) the terminal + account THIS process actually bound to,
    so the operator can see each bot is on its OWN terminal from the log / Telegram
    WITHOUT running --verify. Under the master (BOT_MT5_TERMINAL set) it also flags
    when the bound terminal is NOT the expected one — i.e. the isolation shim didn't
    engage. Returns the report dict on the first connected call, None afterwards.
    Read-only; never raises."""
    global _isolation_reported
    if _isolation_reported:
        return None
    try:
        ti = mt5.terminal_info()
        ai = mt5.account_info()
    except Exception as e:
        print(f"⚠️ isolation_report failed: {e}")
        return None
    bound = (getattr(ti, "path", None) or getattr(ti, "data_path", None)) if ti else None
    data_path = getattr(ti, "data_path", None) if ti else None
    login = getattr(ai, "login", None) if ai else None
    server = getattr(ai, "server", None) if ai else None
    if login is None and not bound:
        return None                         # not connected yet — try again next cycle
    _isolation_reported = True
    print(f"🔗 [{SYMBOL}] bound terminal: {bound} | account: {login}@{server}")
    expected = (os.environ.get("BOT_MT5_TERMINAL") or "").strip().strip('"')
    iso_ok = True
    if expected:
        exp_dir = os.path.dirname(expected).lower()
        got = f"{bound or ''} {data_path or ''}".lower()
        if exp_dir and exp_dir not in got:
            iso_ok = False
            print(f"⚠️ ISOLATION WARNING [{SYMBOL}]: expected terminal under "
                  f"{os.path.dirname(expected)} but bound to {bound} — the isolation "
                  f"shim may not be active! Run: python runserver.py --verify")
    return {"terminal": bound, "data_path": data_path, "login": login,
            "server": server, "isolation_ok": iso_ok}


# ───────────────────────────────────────────────────────────────────────────
#  MARKET STATE (100% 24/7 FOR CRYPTO)
# ───────────────────────────────────────────────────────────────────────────
@_locked
def is_market_open():
    try:
        terminal = mt5.terminal_info()
        if terminal is None or not terminal.connected:
            raise ConnectionError("MT5 terminal not connected to trade server (internet?).")

        info = mt5.symbol_info(SYMBOL)
        if info is None:
            if resolve_symbol() is None:
                return _weekend_guess()
            info = mt5.symbol_info(SYMBOL)

        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            return False

        # Fresh tick within ~2 minutes => market is live.
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick and tick.time:
            offset = get_server_utc_offset_hours() * 3600
            age = datetime.now(timezone.utc).timestamp() - (tick.time - offset)
            if abs(age) < 120:
                return True
            return _weekend_guess()
        return _weekend_guess()
    except ConnectionError:
        raise
    except Exception as e:
        print(f"⚠️ is_market_open error: {e}. Assuming CLOSED (safe).")
        return False


def _weekend_guess():
    """
    БАРОИ БИТКОИН: Бозори крипто 24/7 кор мекунад.
    Рӯзҳои шанбе ва якшанбе ҳеҷ гоҳ маҳкам намешавад!
    """
    return True


def market_should_be_open():
    """
    БАРОИ БИТКОИН: Ҳамеша кушода аст (24/7).
    """
    return True


def _kill_terminal_by_path(path):
    """Kill ONLY the terminal64.exe whose ExecutablePath matches `path`, so this
    bot's hung-recovery never kills the OTHER bot's terminal. `taskkill /IM
    terminal64.exe` would kill EVERY MT5 terminal on the machine by image name —
    fatal when two bots run side by side. Best-effort; returns the count killed."""
    if not path:
        return 0
    try:
        ps = ("Get-CimInstance Win32_Process -Filter \"name='terminal64.exe'\" | "
              "Where-Object { $_.ExecutablePath -ieq '" + path + "' } | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, errors="replace",
                             timeout=20).stdout
        return len([x for x in out.split() if x.strip().isdigit()])
    except Exception as e:
        print(f"⚠️ kill-terminal-by-path failed: {e}")
        return 0


def restart_terminal():
    # NOTE: deliberately NOT @_locked. The kill + 6s settle must NOT hold mt5_lock,
    # or every order/close/SL-modify on the Telegram thread freezes for the whole
    # recovery window. We grab the lock only for the two brief MT5 calls (shutdown
    # and re-initialize); the terminal is down during the sleep anyway, so a manual
    # order tap simply fails fast against the dead terminal instead of blocking.
    global _last_terminal_restart, _SERVER_UTC_OFFSET_H, _SERVER_UTC_OFFSET_TS
    if not config.MT5_PATH:
        print("⚠️ Cannot auto-restart terminal: MT5_PATH not set in .env")
        return False
    _last_terminal_restart = time.time()
    with mt5_lock:
        try:
            mt5.shutdown()
        except Exception:
            pass
    # force-close ONLY this bot's (possibly frozen) terminal — by path, never /IM
    killed = _kill_terminal_by_path(config.MT5_PATH)
    print(f"   killed {killed} terminal process(es) at {config.MT5_PATH}")
    time.sleep(6)
    _SERVER_UTC_OFFSET_H = None
    _SERVER_UTC_OFFSET_TS = 0.0
    with mt5_lock:
        try:
            ok = bool(mt5.initialize(path=config.MT5_PATH))
            if not ok:
                print(f"❌ relaunch initialize() failed: {mt5.last_error()}")
            return ok
        except Exception as e:
            print(f"❌ relaunch exception: {e}")
            return False


def seconds_since_terminal_restart():
    return time.time() - _last_terminal_restart if _last_terminal_restart else 1e18


# ───────────────────────────────────────────────────────────────────────────
#  ORDER EXECUTION
# ───────────────────────────────────────────────────────────────────────────
def _pick_filling_mode(symbol_info):
    fm = getattr(symbol_info, "filling_mode", 0)
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def _dynamic_deviation_points(symbol_info, spread=0.0, risk=None):
    point = getattr(symbol_info, "point", 0) or 0.01
    risk_budget = (risk * config.MAX_SLIPPAGE_RISK_PCT) if (risk and risk > 0) else 0.0
    spread_budget = max(spread, 0.0) * 0.5
    budget = min(config.MAX_SLIPPAGE_USD,
                 max(config.MAX_SLIPPAGE_USD * 0.20, risk_budget, spread_budget))
    points = int(round(budget / point))
    return max(config.MIN_DEVIATION_POINTS, min(config.MAX_DEVIATION_POINTS, points))


def _order_check_error(request):
    if not config.ENABLE_ORDER_CHECK:
        return None
    try:
        check = mt5.order_check(request)
    except Exception as e:
        return f"order_check exception: {e}"
    if check is None:
        return f"order_check returned None: {mt5.last_error()}"
    allowed = {
        0,                                   # order_check success ("Done") — NOT 10009
        mt5.TRADE_RETCODE_DONE,
        getattr(mt5, "TRADE_RETCODE_PLACED", mt5.TRADE_RETCODE_DONE),
        getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", mt5.TRADE_RETCODE_DONE),
    }
    retcode = getattr(check, "retcode", None)
    if retcode not in allowed:
        return f"order_check retcode={retcode} ({getattr(check, 'comment', '')})"
    return None


def _normalize_stops(symbol_info, order_type, price, sl, tp):
    digits = symbol_info.digits
    point = symbol_info.point
    min_dist = max(symbol_info.trade_stops_level, 0) * point

    if order_type == mt5.ORDER_TYPE_BUY:
        if sl is not None and sl >= price - min_dist:
            sl = price - max(min_dist, point)
        if tp is not None and tp <= price + min_dist:
            tp = price + max(min_dist, point)
    else:  # SELL
        if sl is not None and sl <= price + min_dist:
            sl = price + max(min_dist, point)
        if tp is not None and tp >= price - min_dist:
            tp = price - max(min_dist, point)

    sl = round(sl, digits) if sl is not None else 0.0
    tp = round(tp, digits) if tp is not None else 0.0
    return sl, tp


@_locked
def open_order(order_type, lot_size, sl=None, tp=None, risk=None, reward=None,
               signal_price=None, comment="BitcoinSignalBot"):
    if not mt5.symbol_select(SYMBOL, True):
        return {"ok": False, "error": f"Symbol {SYMBOL} not available"}

    si = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    if si is None or tick is None:
        return {"ok": False, "error": "No symbol/tick info"}

    spread = tick.ask - tick.bid
    if spread > config.MAX_SPREAD_USD:
        return {"ok": False, "error": f"Spread too wide: {spread:.2f} > {config.MAX_SPREAD_USD}"}

    step = si.volume_step if (si.volume_step and si.volume_step > 0) else 0.01
    # Defense-in-depth: open_order is the ONE function that commits capital, so it
    # enforces this bot's own hard cap (config.MAX_LOT) here too — never trusting
    # every caller (incl. the open_order_btcusd wrapper) to clamp first.
    cap = si.volume_max
    if getattr(config, "MAX_LOT", 0):
        cap = min(cap, config.MAX_LOT)
    lot_size = min(lot_size, cap)
    # FLOOR to the volume step (never round up) so realized size can't silently
    # exceed the risk budget; then never go below the broker minimum. (+1e-9 keeps
    # float division like 0.03/0.01 from flooring to the wrong step.)
    lot_size = round(max(si.volume_min, math.floor(lot_size / step + 1e-9) * step), 2)

    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

    # anti-chasing on the FILL: refuse ONLY adverse drift — for a BUY, price having
    # run UP (you'd buy higher); for a SELL, price having dropped (you'd sell lower
    # = chasing the move). A FAVOURABLE drift (a better price than the signal) is
    # allowed. SL/TP re-anchor to the live fill, so this blocks gross chasing, not
    # normal manual-tap latency.
    if signal_price is not None and risk and risk > 0:
        sp = float(signal_price)
        adverse = (price - sp) if order_type == mt5.ORDER_TYPE_BUY else (sp - price)
        max_drift = max(spread, risk * config.MAX_ENTRY_DRIFT_R)
        if adverse > max_drift:
            return {"ok": False,
                    "error": f"Entry drift too large: {adverse:.2f} > {max_drift:.2f} (chasing)"}

    if risk is not None and reward is not None and risk > 0 and reward > 0:
        if order_type == mt5.ORDER_TYPE_BUY:
            sl, tp = price - risk, price + reward
        else:
            sl, tp = price + risk, price - reward

    sl, tp = _normalize_stops(si, order_type, price, sl, tp)

    # Slippage control: convert the USD/ATR slippage budget into broker points
    # instead of a blind hardcoded deviation (was 1000) — caps real slippage.
    deviation_pts = _dynamic_deviation_points(si, spread=spread, risk=risk)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": float(lot_size),
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation_pts,
        "magic": config.MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _pick_filling_mode(si),
    }

    # Pre-flight validation (margin, stops, invalid volume) before going live.
    chk = _order_check_error(request)
    if chk:
        return {"ok": False, "error": chk}

    result = mt5.order_send(request)
    if result is None:
        return {"ok": False, "error": f"order_send returned None: {mt5.last_error()}"}
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"ok": False, "error": f"retcode={result.retcode} ({result.comment})",
                "retcode": result.retcode}
    return {
        "ok": True, "ticket": result.order, "price": result.price,
        "volume": result.volume, "sl": sl, "tp": tp,
    }


def open_order_btcusd(order_type, lot_size, sl, tp):
    """Backwards-compatible wrapper for direct BTC order calling."""
    res = open_order(order_type, lot_size, sl, tp)
    return res.get("ok", False)


@_locked
def close_position(ticket):
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return {"ok": False, "error": "position not found"}
    pos = positions[0]
    si = mt5.symbol_info(pos.symbol)
    tick = mt5.symbol_info_tick(pos.symbol)
    if si is None or tick is None:
        return {"ok": False, "error": "no symbol/tick info"}
    if pos.type == mt5.POSITION_TYPE_BUY:
        ctype, cprice = mt5.ORDER_TYPE_SELL, tick.bid
    else:
        ctype, cprice = mt5.ORDER_TYPE_BUY, tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume,
        "type": ctype, "position": pos.ticket, "price": cprice, 
        "deviation": config.MAX_DEVIATION_POINTS,  # wide on EXIT — guarantee the fill
        "magic": config.MAGIC, "comment": "BitcoinSignalBot close",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": _pick_filling_mode(si),
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        rc = None if result is None else result.retcode
        return {"ok": False, "error": f"close failed retcode={rc}"}
    return {"ok": True, "price": result.price}


@_locked
def set_tp_in_usd(target_profit_usd: float):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return {"ok": False, "error": "Позитсияҳои кушода ёфт нашуданд"}

    si = mt5.symbol_info(SYMBOL)
    if not si:
        return {"ok": False, "error": "Маълумоти символ дастрас нест"}

    point = si.point
    tick_size = getattr(si, "trade_tick_size", 0) or point
    tick_value = (getattr(si, "trade_tick_value", 0)
                  or getattr(si, "trade_tick_value_profit", 0))

    if tick_size == 0 or tick_value == 0:
        return {"ok": False, "error": "Арзиши тик нодуруст аст"}

    results = []

    for pos in positions:
        # USER-initiated /tp /sl applies to ALL of this symbol's positions (manual
        # + bot). (Auto-management in trade_manager still touches bot orders only.)

        if pos.profit >= target_profit_usd:
            res = close_position(pos.ticket)
            results.append({"ticket": pos.ticket, "retcode": "CLOSED" if res.get("ok") else "CLOSE_FAIL"})
            continue

        if pos.type == mt5.POSITION_TYPE_BUY:
            price_change = (target_profit_usd * tick_size) / (pos.volume * tick_value)
            tp_price = pos.price_open + price_change
        elif pos.type == mt5.POSITION_TYPE_SELL:
            price_change = (target_profit_usd * tick_size) / (pos.volume * tick_value)
            tp_price = pos.price_open - price_change
        else:
            continue

        tp_price = round(tp_price, si.digits)

        if abs(pos.tp - tp_price) < point:
            continue

        _, final_tp = _normalize_stops(si, pos.type, pos.price_current, pos.sl, tp_price)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "sl": pos.sl, 
            "tp": final_tp,
            "magic": pos.magic,          # match the position (modify_sl includes it too)
        }
        
        res = mt5.order_send(request)
        results.append({"ticket": pos.ticket,
                        "retcode": res.retcode if res else None,
                        "comment": getattr(res, "comment", "") if res else "no response"})

    return {"ok": True, "results": results}


@_locked
def set_sl_in_usd(target_loss_usd: float):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return {"ok": False, "error": "Позитсияҳои кушода ёфт нашуданд"}

    si = mt5.symbol_info(SYMBOL)
    if not si:
        return {"ok": False, "error": "Маълумоти символ дастрас нест"}

    point = si.point
    tick_size = getattr(si, "trade_tick_size", 0) or point
    tick_value = (getattr(si, "trade_tick_value", 0)
                  or getattr(si, "trade_tick_value_profit", 0))

    if tick_size == 0 or tick_value == 0:
        return {"ok": False, "error": "Арзиши тик нодуруст аст"}

    results = []

    for pos in positions:
        # USER-initiated /tp /sl applies to ALL of this symbol's positions (manual
        # + bot). (Auto-management in trade_manager still touches bot orders only.)

        if pos.profit <= -target_loss_usd:
            res = close_position(pos.ticket)
            results.append({"ticket": pos.ticket, "retcode": "CLOSED" if res.get("ok") else "CLOSE_FAIL"})
            continue

        price_change = (target_loss_usd * tick_size) / (pos.volume * tick_value)

        if pos.type == mt5.POSITION_TYPE_BUY:
            sl_price = pos.price_open - price_change
        elif pos.type == mt5.POSITION_TYPE_SELL:
            sl_price = pos.price_open + price_change
        else:
            continue

        sl_price = round(sl_price, si.digits)

        if abs(pos.sl - sl_price) < point:
            continue

        final_sl, _ = _normalize_stops(si, pos.type, pos.price_current, sl_price, pos.tp)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "sl": final_sl,
            "tp": pos.tp,
            "magic": pos.magic,          # match the position (modify_sl includes it too)
        }
        
        res = mt5.order_send(request)
        results.append({"ticket": pos.ticket,
                        "retcode": res.retcode if res else None,
                        "comment": getattr(res, "comment", "") if res else "no response"})

    return {"ok": True, "results": results}


# ───────────────────────────────────────────────────────────────────────────
#  DATA ENGINE
# ───────────────────────────────────────────────────────────────────────────
class GetRealBtcData:
    """Fetches Bitcoin OHLCV data from MT5 for any timeframe and builds a clean frame."""

    MASTER_TIMEFRAMES = [
        mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15,
        mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1,
    ]

    def __init__(self, timeframe=mt5.TIMEFRAME_M15, num_bars=None):
        self.timeframe = timeframe
        self.num_bars = num_bars or config.NUM_BARS

    @_locked
    def get_data(self, specific_tf=None):
        tf = specific_tf if specific_tf is not None else self.timeframe
        if not mt5.symbol_select(SYMBOL, True):
            return None
        return mt5.copy_rates_from_pos(SYMBOL, tf, 0, self.num_bars)

    @staticmethod
    def to_dataframe(data):
        if data is None or len(data) == 0:
            return None
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        if 'tick_volume' not in df.columns:
            df['tick_volume'] = 0
        df['direction'] = np.where(df['close'] > df['open'], 1,
                                   np.where(df['close'] < df['open'], -1, 0))
        df['candle_size'] = df['high'] - df['low']
        df['body_size'] = (df['close'] - df['open']).abs()
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        return df

    def last_bar_age_seconds(self, tf):
        raw = self.get_data(specific_tf=tf)
        if raw is None or len(raw) == 0:
            return None
        last_epoch = float(raw[-1]['time'])
        offset = get_server_utc_offset_hours() * 3600
        return datetime.now(timezone.utc).timestamp() - (last_epoch - offset)

    def broker_utc_now(self):
        """Real UTC 'now' derived from the BROKER's tick clock — not the server OS
        clock. Session detection uses this so sessions stay correct on ANY server
        in the world, regardless of its local timezone OR a drifting/skewed clock:
        broker_time − broker_utc_offset cancels out a skewed local clock (the
        offset rounds to the nearest hour, so any skew under ~30 min is absorbed).
        Falls back to the OS UTC clock only if no tick is available yet."""
        try:
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick and tick.time:
                off = get_server_utc_offset_hours() * 3600
                return datetime.fromtimestamp(tick.time - off, tz=timezone.utc)
        except Exception:
            pass
        return datetime.now(timezone.utc)
