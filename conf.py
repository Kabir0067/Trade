import MetaTrader5 as mt5
import math, time
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
from datetime import datetime

LOGIN    = 248532703
PASSWORD = "1q2w3e0p$Q"
SERVER   = "Exness-MT5Trial"



def check_clos_orders():
    login = 248532703
    password = "1q2w3e0p$Q"
    server = "Exness-MT5Trial"

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    if not mt5.login(login=login, password=password, server=server):
        raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    while True:
        pos_total = mt5.positions_total()
        ord_total = mt5.orders_total()
        if pos_total == 0 and ord_total == 0:
            break
        time.sleep(1)

def _round_to_step(value: float, step: float, max_digits: int = 8) -> float:
    if step <= 0: return round(float(value), max_digits)
    return round(math.floor((value + 1e-12) / step) * step, max_digits)

def _snap_price(symbol: str, price: float) -> float:
    info = mt5.symbol_info(symbol)
    if not info: return float(price)
    tick = info.trade_tick_size or info.point
    p = _round_to_step(price, tick, info.digits)
    return float(round(p, info.digits))

def _lot_sanitize(symbol: str, lot: float) -> float:
    info = mt5.symbol_info(symbol)
    if not info: return float(lot)
    lot = max(info.volume_min, min(info.volume_max, float(lot)))
    step = info.volume_step or 0.01
    return float(round(_round_to_step(lot, step, 3), 3))

def _min_stops_points(symbol: str) -> int:
    info = mt5.symbol_info(symbol)
    if not info: return 0
    stops = int(getattr(info, "stops_level", 0) or 0)
    freeze = int(getattr(info, "freeze_level", 0) or 0)
    return max(stops, freeze, 0)

def _allowed_fillings(symbol: str):
    info = mt5.symbol_info(symbol)
    if not info: return []
    fm = int(getattr(info, "filling_mode", 0) or 0)
    allowed = []
    if fm & mt5.ORDER_FILLING_IOC:    allowed.append(mt5.ORDER_FILLING_IOC)
    if fm & mt5.ORDER_FILLING_FOK:    allowed.append(mt5.ORDER_FILLING_FOK)
    if fm & mt5.ORDER_FILLING_RETURN: allowed.append(mt5.ORDER_FILLING_RETURN)
    return allowed

def _send_with_fillings(symbol: str, request: dict):
    last = None
    tried = []
    for fm in _allowed_fillings(symbol):
        r = dict(request); r["type_filling"] = fm
        res = mt5.order_send(r); last = res
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            return res
        tried.append(("fill", fm, getattr(res, "retcode", None), getattr(res, "comment", None)))
    # Фолбэк: бе type_filling
    r2 = dict(request); r2.pop("type_filling", None)
    res2 = mt5.order_send(r2); last = res2
    if not (res2 and res2.retcode == mt5.TRADE_RETCODE_DONE):
        print("❌ order_send failed. Tried:", tried,
              "fallback_retcode:", getattr(res2, "retcode", None),
              "comment:", getattr(res2, "comment", None))
    return last

# ==== USD→POINTS ва тарафи нарх ====
def _usd_to_points(symbol: str, usd_amount: float, lot: float) -> Optional[float]:
    info = mt5.symbol_info(symbol)
    if not info or not info.trade_tick_value or not info.trade_tick_size or not info.point:
        return None
    vpp_1lot = (info.trade_tick_value / info.trade_tick_size) * info.point  # USD per point per 1 lot
    vpp = vpp_1lot * float(lot)
    if vpp <= 0: return None
    return float(usd_amount) / vpp

def _get_price_for_side(tick, side: str) -> float:
    return float(tick.ask) if side.lower() == "buy" else float(tick.bid)

# ==== SL/TP ҳисобкунии дақиқ ====
SLTPUnits = Literal["points", "usd", "price"]

@dataclass
class SLTP:
    sl: Optional[float]
    tp: Optional[float]

def _compute_sltp_exact(
    symbol: str,
    side: str,
    entry_price: float,   # база = нархи вуруд/иҷро
    lot: float,
    sl_value: Optional[float],
    tp_value: Optional[float],
    units: SLTPUnits = "points",
    enforce_exact: bool = True,
) -> SLTP:
    info = mt5.symbol_info(symbol)
    if not info: raise RuntimeError("symbol_info failed")

    pt = info.point
    min_pts = _min_stops_points(symbol)

    def _level_from_points(points: float, is_sl: bool) -> float:
        if side.lower() == "buy":
            level = entry_price - points * pt if is_sl else entry_price + points * pt
        else:
            level = entry_price + points * pt if is_sl else entry_price - points * pt
        return _snap_price(symbol, level)

    def _ensure_min(points: float, name: str) -> float:
        if enforce_exact and points < min_pts:
            raise ValueError(f"{name}={points:.1f} < min_stops={min_pts} points")
        return max(points, float(min_pts))

    sl = tp = None

    if sl_value is not None:
        if units == "points":
            pts = _ensure_min(float(sl_value), "SL")
            sl = _level_from_points(pts, True)
        elif units == "usd":
            pts = _usd_to_points(symbol, sl_value, lot)
            if pts is None: raise ValueError("Cannot convert SL USD to points")
            pts = _ensure_min(float(pts), "SL")
            sl = _level_from_points(pts, True)
        elif units == "price":
            lvl = _snap_price(symbol, float(sl_value))
            dist_pts = abs(lvl - entry_price) / pt
            if enforce_exact and dist_pts < min_pts:
                raise ValueError(f"SL distance {dist_pts:.1f} < min_stops={min_pts}")
            sl = lvl
        else:
            raise ValueError("Unknown units for SL")

    if tp_value is not None:
        if units == "points":
            pts = _ensure_min(float(tp_value), "TP")
            tp = _level_from_points(pts, False)
        elif units == "usd":
            pts = _usd_to_points(symbol, tp_value, lot)
            if pts is None: raise ValueError("Cannot convert TP USD to points")
            pts = _ensure_min(float(pts), "TP")
            tp = _level_from_points(pts, False)
        elif units == "price":
            lvl = _snap_price(symbol, float(tp_value))
            dist_pts = abs(lvl - entry_price) / pt
            if enforce_exact and dist_pts < min_pts:
                raise ValueError(f"TP distance {dist_pts:.1f} < min_stops={min_pts}")
            tp = lvl
        else:
            raise ValueError("Unknown units for TP")

    return SLTP(sl=sl, tp=tp)

# ==== Ёфтани последняя позиция ====
def _latest_position_ticket(symbol: str, side: str) -> Optional[int]:
    want = 0 if side.lower() == "buy" else 1
    poss = mt5.positions_get(symbol=symbol) or []
    poss = sorted(poss, key=lambda p: p.time_update, reverse=True)
    for p in poss:
        if int(p.type) == want:
            return int(p.ticket)
    return None

# ==== Санҷиши бозор ====
def is_market_open(symbol: str, recent_seconds: int = 180) -> Tuple[bool, str]:
    info = mt5.symbol_info(symbol)
    if not info: return False, "no symbol info"
    if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return False, "trade disabled"
    tick = mt5.symbol_info_tick(symbol)
    if not tick: return False, "no tick"
    now_ts = time.time()
    # Агар тики нав набошад — эҳтимол бозор баста, вале иҷозат медиҳем ки барномат бе краш кор кунад
    if (now_ts - float(tick.time)) > recent_seconds:
        return False, "no recent quotes"
    return True, "ok"

# ==== Тағйири SL/TP баъд аз кушодан, бо нархи ИҶРО ====
def _apply_sltp_to_position_with_base(
    symbol: str,
    side: str,
    position_ticket: int,
    executed_price: float,
    lot: float,
    sl: Optional[float],
    tp: Optional[float],
    units: SLTPUnits,
    enforce_exact: bool,
    tries: int = 3,
    delay: float = 0.25,
) -> bool:
    sltp = _compute_sltp_exact(symbol, side, executed_price, lot, sl, tp, units, enforce_exact)
    req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": int(position_ticket)}
    if sltp.sl is not None: req["sl"] = float(sltp.sl)
    if sltp.tp is not None: req["tp"] = float(sltp.tp)
    last = None
    for i in range(tries):
        res = mt5.order_send(req); last = res
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        time.sleep(delay * (i + 1))
    return False

# ==== Кушодани ордер ====
def open_order(
    symbol: str,
    lot: float,
    side: Literal["buy","sell"] = "buy",
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    sltp_units: SLTPUnits = "points",
    deviation_points: Optional[int] = None,
    enforce_exact: bool = True,
    check_market: bool = True,
    market_policy: Literal["return","raise"] = "return",  
) -> dict:
    if not mt5.initialize():
        return {"ok": False, "error": f"MT5 init failed: {mt5.last_error()}"}
    if not mt5.login(login=LOGIN, password=PASSWORD, server=SERVER):
        return {"ok": False, "error": f"MT5 login failed: {mt5.last_error()}"}

    if not mt5.symbol_select(symbol, True):
        return {"ok": False, "error": f"symbol_select({symbol}) failed: {mt5.last_error()}"}

    if check_market:
        ok, why = is_market_open(symbol)
        if not ok:
            msg = f"Market closed for {symbol}: {why}"
            if market_policy == "raise":
                raise RuntimeError(msg)
            return {"ok": False, "market_closed": True, "reason": why}

    info = mt5.symbol_info(symbol)
    if not info or info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        return {"ok": False, "error": "Symbol trade disabled or no info"}

    lot = _lot_sanitize(symbol, lot)

    tick = mt5.symbol_info_tick(symbol)
    if not tick or not tick.bid or not tick.ask:
        return {"ok": False, "error": "No market tick"}

    entry_price_pre = _get_price_for_side(tick, side)
    try:
        sltp_pre = _compute_sltp_exact(symbol, side, entry_price_pre, lot, sl, tp, sltp_units, enforce_exact)
    except Exception as e:
        return {"ok": False, "error": f"SLTP compute error: {e}"}

    if deviation_points is None:
        deviation_points = max(5, int(info.spread or 10) * 2)

    order_type = mt5.ORDER_TYPE_BUY if side.lower() == "buy" else mt5.ORDER_TYPE_SELL
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "type": order_type,
        "volume": float(lot),
        "price": float(entry_price_pre),
        "deviation": int(deviation_points),
        "magic": 123456,
        "type_time": mt5.ORDER_TIME_GTC,
        "comment": "auto",
    }
    if sltp_pre.sl is not None: req["sl"] = float(sltp_pre.sl)
    if sltp_pre.tp is not None: req["tp"] = float(sltp_pre.tp)

    res = _send_with_fillings(symbol, req)

    if not res or res.retcode != mt5.TRADE_RETCODE_DONE:
        if getattr(res, "retcode", None) == 10018: 
            if market_policy == "raise":
                raise RuntimeError("Market closed (server retcode=10018)")
            return {"ok": False, "market_closed": True, "reason": "server: market closed"}

        req2 = dict(req); req2.pop("sl", None); req2.pop("tp", None)
        res2 = _send_with_fillings(symbol, req2)
        if not res2 or res2.retcode != mt5.TRADE_RETCODE_DONE:
            if getattr(res2, "retcode", None) == 10018:
                if market_policy == "raise":
                    raise RuntimeError("Market closed (server retcode=10018)")
                return {"ok": False, "market_closed": True, "reason": "server: market closed"}
            return {"ok": False, "error": f"Order open failed. retcode={getattr(res2,'retcode',None)} last_error={mt5.last_error()}"}

        executed_price = float(getattr(res2, "price", entry_price_pre))
        time.sleep(0.10)
        pos_ticket = _latest_position_ticket(symbol, side)
        if pos_ticket and (sl is not None or tp is not None):
            ok2 = _apply_sltp_to_position_with_base(
                symbol, side, pos_ticket, executed_price, lot, sl, tp, sltp_units, enforce_exact
            )
            if not ok2:
                return {"ok": False, "error": "Order opened but SL/TP post-modify failed",
                        "executed_price": executed_price, "ticket": pos_ticket}
        return {"ok": True, "mode": "post_modify", "executed_price": executed_price, "ticket": pos_ticket}

    return {"ok": True, "mode": "attached",
            "executed_price": float(getattr(res, "price", entry_price_pre)),
            "deal": getattr(res, "deal", None)}

def close_orders(symbol: Optional[str] = None,
                 side: Optional[Literal["buy","sell"]] = None,
                 include_pendings: bool = False,
                 only_magic: Optional[int] = None,
                 deviation_points: Optional[int] = None,
                 market_policy: Literal["return","raise"] = "return",
                 retries: int = 2,
                 sleep: float = 0.2) -> dict:
    if not mt5.initialize():
        return {"ok": False, "error": f"MT5 init failed: {mt5.last_error()}"}
    if not mt5.login(login=LOGIN, password=PASSWORD, server=SERVER):
        return {"ok": False, "error": f"MT5 login failed: {mt5.last_error()}"}

    results = {"closed_positions": 0, "canceled_pendings": 0, "errors": []}

    # --- Close positions ---
    positions = mt5.positions_get() or []
    for p in positions:
        if symbol and p.symbol != symbol:
            continue
        if side is not None and ((side == "buy"  and p.type != mt5.POSITION_TYPE_BUY) or
                                 (side == "sell" and p.type != mt5.POSITION_TYPE_SELL)):
            continue
        if only_magic is not None and int(getattr(p, "magic", 0)) != int(only_magic):
            continue

        info = mt5.symbol_info(p.symbol)
        tick = mt5.symbol_info_tick(p.symbol)
        if not info or not tick:
            results["errors"].append((int(p.ticket), "no symbol/tick"))
            continue

        opp_type = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_price = float(tick.bid) if opp_type == mt5.ORDER_TYPE_SELL else float(tick.ask)
        dev = deviation_points if deviation_points is not None else max(5, int((info.spread or 10) * 2))

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "type": opp_type,
            "position": int(p.ticket),
            "volume": float(p.volume),
            "price": close_price,
            "deviation": int(dev),
            "magic": 123456,
            "comment": "close_all",
        }

        res = None
        success = False
        for i in range(max(1, retries + 1)):
            res = _send_with_fillings(p.symbol, req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                results["closed_positions"] += 1
                success = True
                break
            if getattr(res, "retcode", None) == 10018: 
                if market_policy == "raise":
                    return {"ok": False, "error": "market closed", "ticket": int(p.ticket)}
                results["errors"].append((int(p.ticket), 10018, "market closed"))
                break
            time.sleep(sleep * (i + 1))

        if not success:
            results["errors"].append((int(p.ticket), getattr(res, "retcode", None)))

    if include_pendings:
        orders = mt5.orders_get() or []
        for o in orders:
            if symbol and o.symbol != symbol:
                continue
            if only_magic is not None and int(getattr(o, "magic", 0)) != int(only_magic):
                continue
            req = {"action": mt5.TRADE_ACTION_REMOVE, "order": int(o.ticket)}
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                results["canceled_pendings"] += 1
            else:
                if getattr(res, "retcode", None) == 10018 and market_policy == "return":
                    results["errors"].append((int(o.ticket), 10018, "market closed"))
                else:
                    results["errors"].append((int(o.ticket), getattr(res, "retcode", None)))

    results["ok"] = True
    return results


if __name__ == "__main__":
    resp = open_order(
        symbol="XAUUSDm",
        lot=0.05,
        side="sell",
        sl=10, tp=10,
        sltp_units="usd",
        enforce_exact=True,
    )
    print(resp)

