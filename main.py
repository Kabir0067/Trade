import os
import time
from datetime import datetime
import MetaTrader5 as mt5
from bstsymbl import choose_once
from conf import open_order
from scalp_analysis import EngineConfig, Engine


CHECK_INTERVAL = float(os.getenv("CHECK_INTERVAL", "2"))   
MAX_WAIT_FLAT = int(os.getenv("MAX_WAIT_FLAT", "0"))       
VOL = float(os.getenv("VOLUME", "0.5"))
SL_POINTS = float(os.getenv("SL_POINTS", "50"))
TP_POINTS = float(os.getenv("TP_POINTS", "10"))
EXECUTE = bool(int(os.getenv("EXECUTE", "0")))
RUN_FOREVER = bool(int(os.getenv("RUN_FOREVER", "1")))
USE_FALLBACK = bool(int(os.getenv("USE_FALLBACK", "0")))  


def ensure_mt5(login: int, password: str, server: str):
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    if login and password and server:
        if not mt5.login(login=login, password=password, server=server):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")


def positions_for_symbol(symbol: str):
    poss = mt5.positions_get(symbol=symbol)
    return poss if poss is not None else []


def is_flat(symbol: str) -> bool:
    return len(positions_for_symbol(symbol)) == 0


def wait_until_flat(symbol: str, check_interval=2.0, max_wait=0) -> bool:
    start = time.time()
    while True:
        if is_flat(symbol):
            return True
        if max_wait and (time.time() - start) > max_wait:
            return False
        time.sleep(check_interval)


def _to_side(signal_obj) -> str | None:
    if signal_obj is None:
        return None
    if isinstance(signal_obj, str):
        s = signal_obj.strip().lower()
        if s in ("buy", "long", "bullish", "up"):
            return "buy"
        if s in ("sell", "short", "bearish", "down"):
            return "sell"
        return None
    if isinstance(signal_obj, dict):
        for k in ("side", "signal", "action", "direction"):
            v = signal_obj.get(k)
            if isinstance(v, str):
                return _to_side(v)
    return None


def _prices_from_points(symbol: str, side: str, sl_points: float, tp_points: float):
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if not info or not tick:
        raise RuntimeError(f"symbol/tick not found for {symbol}, last_error={mt5.last_error()}")
    point = info.point
    if side == "buy":
        price = tick.ask
        sl = round(price - sl_points * point, info.digits)
        tp = round(price + tp_points * point, info.digits)
        order_type = mt5.ORDER_TYPE_BUY
    else:
        price = tick.bid
        sl = round(price + sl_points * point, info.digits)
        tp = round(price - tp_points * point, info.digits)
        order_type = mt5.ORDER_TYPE_SELL
    return price, sl, tp, order_type, info


def send_with_filling_fallback(base_request: dict):
    last = None
    for mode in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        req = dict(base_request)
        req["type_filling"] = mode
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


def place_order(symbol: str, side: str, volume: float, sl_points: float, tp_points: float):
    try:
        res = open_order(symbol, volume, sl_points, tp_points, side=side)
        print("open_order → OK")
        return res
    except Exception as e:
        print(f"open_order Exception: {e}")
        if not USE_FALLBACK:
            raise
        price, sl, tp, order_type, info = _prices_from_points(symbol, side, sl_points, tp_points)
        base = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": int(os.getenv("DEVIATION", "20")),
            "magic": int(os.getenv("MAGIC", "20250902")),
            "comment": f"bot:{side}@{datetime.now().isoformat(timespec='seconds')}",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        return send_with_filling_fallback(base)

def main():
    login = int(os.getenv("MT5_LOGIN", "248532703"))
    password = os.getenv("MT5_PASSWORD", "1q2w3e0p$Q")
    server = os.getenv("MT5_SERVER", "Exness-MT5Trial")

    ensure_mt5(login, password, server)

    try:
        while True:
            sym = os.getenv("SYMBOL") or choose_once()

            cfg = EngineConfig(
                login=login,
                password=password,
                server=server,
                finnhub_key=os.getenv("FINNHUB_KEY", ""),
                symbols=os.getenv("SYMBOLS", sym),
            )
            engine = Engine(cfg)
            engine.start()

            print(f"⏳ Интизор мешавем то позицияҳои {sym} пурра баста шаванд…")
            ok = wait_until_flat(sym, CHECK_INTERVAL, MAX_WAIT_FLAT)
            if not ok:
                print("⚠️ MAX_WAIT_FLAT гузашт, давраро идома медиҳем.")
                continue

            sig = engine.get_signal(sym, execute=EXECUTE)
            side = _to_side(sig)
            if side not in ("buy", "sell"):
                print(f"⚠️ Сигнали номаълум: {sig}. Мегузарем.")
                time.sleep(CHECK_INTERVAL)
                continue

            print(f"🚀 Кушодани ордер: {side.upper()} {sym}")
            place_order(sym, side, VOL, SL_POINTS, TP_POINTS)

            print(f"⏳ Интизор то баста шудани ордер(ҳо) барои {sym}…")
            wait_until_flat(sym, CHECK_INTERVAL, MAX_WAIT_FLAT)

            if not RUN_FOREVER:
                break

    except KeyboardInterrupt:
        print("🛑 Қатъ шуд (Ctrl+C).")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()