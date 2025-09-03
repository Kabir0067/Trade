import MetaTrader5 as mt5
import os, math, time
from datetime import datetime
import pytz

LOGIN    = int(os.getenv("MT5_LOGIN", '248532703'))
PASSWORD = os.getenv("MT5_PASSWORD", "1q2w3e0p$Q")    
SERVER   = os.getenv("MT5_SERVER", 'Exness-MT5Trial')


_INIT_TRIES = (0.5, 1.0, 2.0, 3.0) 
_CONNECTED_ONCE = False          

def _is_connected_ok() -> bool:
    try:
        ti = mt5.terminal_info()
        acc = mt5.account_info()
        return bool(ti and acc and acc.login == LOGIN)
    except:
        return False

def initialize_mt5() -> bool:
    global _CONNECTED_ONCE
    if _is_connected_ok():
        if not _CONNECTED_ONCE:
            acc = mt5.account_info()
            print(f"✅ MT5 connected | balance={acc.balance} equity={acc.equity} margin_free={acc.margin_free}")
            _CONNECTED_ONCE = True
        return True

    try: mt5.shutdown()
    except: pass

    ok = False
    for wait in _INIT_TRIES:
        ok = mt5.initialize(login=LOGIN, server=SERVER, password=PASSWORD)
        if ok and _is_connected_ok():
            acc = mt5.account_info()
            print(f"✅ MT5 connected | balance={acc.balance} equity={acc.equity} margin_free={acc.margin_free}")
            _CONNECTED_ONCE = True
            return True
        code, msg = mt5.last_error()
        print(f"❌ MT5 initialize/login failed: {code} {msg}; retry in {wait}s")
        time.sleep(wait)
        try: mt5.shutdown()
        except: pass
    return False

def shutdown_mt5():
    try:
        mt5.shutdown()
        print("✅ MT5 disconnected")
    except Exception as e:
        print(f"❌ Error during MT5 shutdown: {e}")

def check_autotrading_enabled() -> bool:
    try:
        ti = mt5.terminal_info()
        if not ti:
            print("⚠️ terminal_info() None (идома медиҳем).")
            return False
        if not ti.trade_allowed:
            print("⚠️ Autotrading хомӯш аст дар терминал — идома медиҳем (fallback).")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Ҳангоми тафтиши автотрейдинг: {e} (идома медиҳем)")
        return False

def _round_to_step(value, step, min_v=None, max_v=None):
    if not step or step <= 0: return float(value)
    k = math.floor(float(value) / step)
    v = k * step
    if min_v is not None and v < min_v: v = min_v
    if max_v is not None and v > max_v: v = max_v
    s = f"{step:.10f}".rstrip('0').rstrip('.')
    decimals = len(s.split('.')[-1]) if '.' in s else 0
    return float(f"{v:.{decimals}f}")

def normalize_price(symbol, price):
    if price is None: return None
    info = mt5.symbol_info(symbol)
    return float(round(float(price), info.digits)) if info else float(price)

def ensure_symbol_ready(symbol) -> bool:
    if not mt5.symbol_select(symbol, True):
        print(f"❌ symbol_select({symbol}) failed: {mt5.last_error()}")
        return False
    info = mt5.symbol_info(symbol)
    if info is None:
        print("❌ symbol_info() returned None")
        return False
    if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        print("❌ Symbol trade disabled by broker")
        return False
    return True

def get_tick_retry(symbol, tries=6, delay=0.25):
    for _ in range(tries):
        t = mt5.symbol_info_tick(symbol)
        if t and t.bid and t.ask and t.bid > 0 and t.ask > 0:
            return t
        time.sleep(delay)
    return None

def adjust_lot_size(symbol, lot):
    info = mt5.symbol_info(symbol)
    min_lot = (info.volume_min or 0.01) if info else 0.01
    max_lot = (info.volume_max or 100.0) if info else 100.0
    lot_step = (info.volume_step or 0.01) if info else 0.01
    v = max(min_lot, min(max_lot, float(lot)))
    v = _round_to_step(v, lot_step, min_lot, max_lot)
    print(f"ℹ️ Lot adjusted for {symbol}: {v}")
    return v

def calculate_points_from_usd(symbol, usd_amount, lot):
    info = mt5.symbol_info(symbol)
    if not info:
        print(f"⚠️ No symbol info for {symbol}; skip SL/TP")
        return None
    tick_value = info.trade_tick_value
    tick_size  = info.trade_tick_size
    point      = info.point
    if not tick_value or not tick_size or not point:
        print(f"⚠️ Invalid tick params for {symbol}; skip SL/TP")
        return None
    vpp_1lot = (tick_value / tick_size) * point
    vpp_lot  = vpp_1lot * float(lot)
    if vpp_lot <= 0:
        print("⚠️ Non-positive value per point; skip SL/TP")
        return None
    pts = float(usd_amount) / vpp_lot
    print(f"ℹ️ Points({symbol}): {pts:.6f} for USD={usd_amount}, lot={lot}")
    return pts

def respect_min_stop_distance(symbol, entry_price, sl_price, tp_price, order_type):
    info = mt5.symbol_info(symbol)
    if not info:
        return sl_price, tp_price
    point = info.point or 0.0
    stops_level  = getattr(info, "stops_level", 0) * point
    freeze_level = getattr(info, "freeze_level", 0) * point
    min_dist = max(stops_level, freeze_level)

    def _adj(target, is_tp):
        if target is None: return None
        if order_type == mt5.ORDER_TYPE_BUY:
            if is_tp and target < entry_price + min_dist: return entry_price + min_dist
            if not is_tp and target > entry_price - min_dist: return entry_price - min_dist
        else:
            if is_tp and target > entry_price - min_dist: return entry_price - min_dist
            if not is_tp and target < entry_price + min_dist: return entry_price + min_dist
        return target

    new_sl = normalize_price(symbol, _adj(sl_price, False)) if sl_price is not None else None
    new_tp = normalize_price(symbol, _adj(tp_price, True )) if tp_price is not None else None
    return new_sl, new_tp

def auto_deviation(symbol, mult=2.5, min_dev=5, max_dev=120):
    info = mt5.symbol_info(symbol)
    if not info or info.spread is None: return 50
    dev = int(info.spread * mult)
    return max(min_dev, min(max_dev, dev))


def compose_market_request(order_type, symbol, volume, price, sl=None, tp=None):
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": normalize_price(symbol, float(price)),
        "deviation": auto_deviation(symbol),
        "magic": 123456,
        "comment": "auto",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    if sl is not None: req["sl"] = float(sl)
    if tp is not None: req["tp"] = float(tp)
    return req

def send_with_filling_fallback(base_request):
    last = None
    for mode in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        req = dict(base_request); req["type_filling"] = mode
        print(f"📤 order_send with filling={mode}")
        r = mt5.order_send(req)
        last = r
        if r and r.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Order opened: ticket={r.order}")
            return r
        elif r:
            print(f"❌ Error: retcode={r.retcode}, comment={r.comment}")
        else:
            print(f"❌ order_send=None, last_error={mt5.last_error()}")
    return last

def modify_sl_tp(position_ticket, symbol, sl_price=None, tp_price=None):
    if sl_price is None and tp_price is None:
        return True
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": int(position_ticket),
    }
    if sl_price is not None: req["sl"] = float(sl_price)
    if tp_price is not None: req["tp"] = float(tp_price)
    r = mt5.order_send(req)
    if r and r.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ SL/TP modified for pos={position_ticket}")
        return True
    print(f"❌ SLTP modify failed: {getattr(r,'retcode',None)} - {getattr(r,'comment',None)}")
    return False

def build_order_params(symbol, side, lot, sl_usd=None, tp_usd=None):
    info = mt5.symbol_info(symbol)
    tick = get_tick_retry(symbol)
    if not info or not tick: return None
    order_type = mt5.ORDER_TYPE_BUY if side.lower() == "buy" else mt5.ORDER_TYPE_SELL
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    volume = adjust_lot_size(symbol, lot)

    sl_price = tp_price = None

    def _pts(usd):
        try:
            if usd is None: return None
            u = float(usd)
            if u <= 0: return None
            return calculate_points_from_usd(symbol, u, volume)
        except: return None

    sl_pts = _pts(sl_usd); tp_pts = _pts(tp_usd)
    if sl_pts is not None:
        sl_price = price - sl_pts*info.point if order_type==mt5.ORDER_TYPE_BUY else price + sl_pts*info.point
    if tp_pts is not None:
        tp_price = price + tp_pts*info.point if order_type==mt5.ORDER_TYPE_BUY else price - tp_pts*info.point

    sl_price, tp_price = respect_min_stop_distance(symbol, price, sl_price, tp_price, order_type)
    return {"order_type": order_type, "price": price, "volume": volume, "sl": sl_price, "tp": tp_price}

def open_order(symbol, lot, stop_loss_usd=None, take_profit_usd=None, side="buy", require_autotrading=False):
    print(f"▶️ open_order(symbol={symbol}, side={side}, lot={lot}, SL_USD={stop_loss_usd}, TP_USD={take_profit_usd})")
    if not initialize_mt5(): return False

    enabled = check_autotrading_enabled()
    if require_autotrading and not enabled:
        return False 

    if not ensure_symbol_ready(symbol):
        return False

    params = build_order_params(symbol, side, lot, stop_loss_usd, take_profit_usd)
    if not params:
        print("❌ Could not build params (no tick/info)")
        return False

    req1 = compose_market_request(params["order_type"], symbol, params["volume"], params["price"], sl=params["sl"], tp=params["tp"])
    res = send_with_filling_fallback(req1)

    if not res or res.retcode != mt5.TRADE_RETCODE_DONE:
        req2 = compose_market_request(params["order_type"], symbol, params["volume"], params["price"], sl=None, tp=None)
        res2 = send_with_filling_fallback(req2)
        if res2 and res2.retcode == mt5.TRADE_RETCODE_DONE and (params["sl"] is not None or params["tp"] is not None):
            time.sleep(0.15) 
            side_sign = 0 if params["order_type"] == mt5.ORDER_TYPE_BUY else 1
            pos_ticket = None
            for p in sorted(mt5.positions_get(symbol=symbol) or [], key=lambda x: x.time_update, reverse=True):
                if p.type == side_sign:
                    pos_ticket = p.ticket
                    break
            if pos_ticket:
                modify_sl_tp(pos_ticket, symbol, params["sl"], params["tp"])
        res = res2

    return bool(res and res.retcode == mt5.TRADE_RETCODE_DONE)

def is_market_open(symbol):
    if not mt5.initialize():
        print(f"❌ Ошибка: Не удалось инициализировать MT5: {mt5.last_error()}")
        return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Не удалось получить информацию о символе {symbol}: {mt5.last_error()}")
        return False
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.bid == 0 or tick.ask == 0:
        print(f"❌ Не удалось получить текущие цены для {symbol}: {mt5.last_error()}")
        return False
    
    current_time = datetime.datetime.now(pytz.timezone('Etc/GMT-3'))
    day_of_week = current_time.weekday()
    hour = current_time.hour
    
    if day_of_week == 5 and hour >= 17:
        print(f"❌ Бозор баста аст: {current_time} (Шанбе пас аз 17:00 EST)")
        return False
    if day_of_week == 6:
        print(f"❌ Бозор баста аст: {current_time} (Якшанбе)")
        return False
    
    if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        print(f"❌ Торговля для символа {symbol} отключена")
        return False
    
    return True

def close_all_positions():
    if not initialize_mt5():
        return False

    positions = mt5.positions_get()
    if positions is None:
        print("❌ Ошибка получения позиций: ", mt5.last_error())
        mt5.shutdown()
        return False

    if len(positions) == 0:
        print("✅ Нет открытых позиций для закрытия.")
        mt5.shutdown()
        return True

    all_closed = True
    for pos in positions:
        symbol = pos.symbol
        volume = pos.volume
        ticket = pos.ticket
        price_tick = mt5.symbol_info_tick(symbol)
        if price_tick is None:
            print(f"❌ Не удалось получить цену для {symbol}")
            all_closed = False
            continue

        if pos.type == mt5.POSITION_TYPE_BUY:
            close_price = price_tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        elif pos.type == mt5.POSITION_TYPE_SELL:
            close_price = price_tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        else:
            print(f"❌ Неизвестный тип позиции для тикета {ticket}")
            all_closed = False
            continue

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Закрытие позиции",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Ошибка закрытия позиции {ticket}: {result.retcode if result else 'None'} - {result.comment if result else 'No result'}")
            all_closed = False
        else:
            print(f"✅ Позиция {ticket} закрыта успешно")

    mt5.shutdown()
    return all_closed


# if __name__ == "__main__":
#     print(open_order('GBPUSDm', lot=0.01, stop_loss_usd=1,
#                                     take_profit_usd=2.0, side='buy'))
