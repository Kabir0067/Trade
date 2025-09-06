import os
import time
import MetaTrader5 as mt5
from bstsymbl import choose_once
from conf import open_order, close_orders
from scalp_analysis import EngineConfig, Engine, apply_high_accuracy_mode


def _norm_sig(s: str):
    if s is None: return None
    s = str(s).strip().lower()
    return s if s in ("buy", "sell") else None

def check_direction(symb, sign, n_checks: int = 3, pause_sec: float = 1.0, consecutive: bool = True):
    sign = _norm_sig(sign)
    if sign not in ("buy", "sell"):
        return False
    opposite = "sell" if sign == "buy" else "buy"

    cfg = EngineConfig(
        login=248532703,
        password="1q2w3e0p$Q",
        server="Exness-MT5Trial",
        finnhub_key="",
        symbols=symb
    )
    apply_high_accuracy_mode(cfg, bool(int(os.getenv("HIGH_ACCURACY","0"))))
    engine = Engine(cfg)
    engine.start()
    try:
        hits = 0
        for i in range(1, n_checks + 1):
            res = engine.get_signals(execute=bool(int(os.getenv("EXECUTE","0"))))
            sig = _norm_sig((res.get(symb) or {}).get("signal"))
            print(f"[{symb}] dir-check {i}/{n_checks}: {sig}")
            if sig == opposite:
                hits += 1
            else:
                if consecutive:
                    hits = 0  
            if hits >= n_checks:
                close_orders(symbol=symb, side=sign)
                return True

            time.sleep(pause_sec)
    finally:
        try: engine.stop()
        except: pass
    return False



def check_clos_orders(symb, sign):
    login = 248532703
    password = "1q2w3e0p$Q"
    server = "Exness-MT5Trial"

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    if not mt5.login(login=login, password=password, server=server):
        raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    sign = _norm_sig(sign)

    while True:
        check_direction(symb, sign)

        pos_list = mt5.positions_get(symbol=symb) or []
        ord_list = [o for o in (mt5.orders_get() or []) if getattr(o, "symbol", "") == symb]
        if len(pos_list) == 0 and len(ord_list) == 0:
            break
        time.sleep(1)


def main():
    while True:
        sym = choose_once()
        cfg = EngineConfig(
            login=248532703,
            password="1q2w3e0p$Q",
            server="Exness-MT5Trial",
            finnhub_key="",
            symbols=sym
        )
        apply_high_accuracy_mode(cfg, bool(int(os.getenv("HIGH_ACCURACY","0"))))
        engine = Engine(cfg)
        engine.start()

        cntby = 0
        cntsl = 0

        try:
            i = 1
            while i <= 3:
                res = engine.get_signals(execute=bool(int(os.getenv("EXECUTE","0"))))
                sig = _norm_sig(res[sym]["signal"])
                conf = float(res[sym].get("confidence", 0.0))
                print(f'Best = {sym} Signal = {sig} Conf = {conf}')
                if sig == "buy":
                    cntby += 1
                elif sig == "sell":
                    cntsl += 1
                i += 1
                time.sleep(1)  
        finally:
            try: engine.stop()
            except: pass

        if cntby == 3:
            side = "buy"
        elif cntsl == 3:
            side = "sell"
        else:
            continue
        open_order(
            symbol=sym,
            lot=0.01,
            side=side,
            sl=0.9,
            tp=1.0,
            sltp_units="usd",
            enforce_exact=True,
        )
        check_clos_orders(sym, side)


if __name__ == "__main__":
    main()