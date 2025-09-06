from __future__ import annotations
import os, sys, time, math, signal, random, re
from dataclasses import dataclass, replace
from typing import Dict, Tuple, Optional, List, Callable
from datetime import datetime, time as dtime, timezone, timedelta
import numpy as np
import pandas as pd
try:
    import MetaTrader5 as mt5
    import talib
except Exception as e:
    raise RuntimeError("Лозим: pip install MetaTrader5 TA-Lib (бо бинари системавӣ).") from e

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ------------------- BASES / TARGETS -------------------
TARGET_BASE = ["EURUSD", "XAUUSD", "BTCUSD"]  # тартиби мақсаднок
BASES = tuple(TARGET_BASE)

# Афзалият дар tie-break: XAU > EUR > BTC
PREFERRED_ORDER = ("XAUUSD", "EURUSD", "BTCUSD")
_PREF_RANK = {b: i for i, b in enumerate(PREFERRED_ORDER)}

def _rank_base_of(symbol: str) -> int:
    return _PREF_RANK.get(extract_base(symbol), 99)

def extract_base(name: str) -> str:
    """Аз номи брокер базаи дақиқ (EURUSD/XAUUSD/BTCUSD)-ро мебардорад."""
    u = name.upper()
    for b in BASES:
        if u.startswith(b):
            return b
    m = re.search(r"(EURUSD|XAUUSD|BTCUSD)", u)
    return m.group(1) if m else ""

STYLE   = "Scalping"
LOGIN    = 248532703
PASSWORD = "1q2w3e0p$Q"
SERVER   = "Exness-MT5Trial"

TRADE_TZ = "Asia/Dushanbe"
TRADING_WINDOWS = ""
FX_DAYS = ""

RELAXED = False

MIN_TICK_FRESH_SEC     = 30
MAX_BAR_STALENESS_MULT = 4.0
SPREAD_SECONDS_BACK    = 180
SPREAD_MED_MULT_MAX    = 2.2

VETO_BB_POS_MARGIN     = 0.05
VETO_MIN_RSI_DIST      = 1.8

MIN_SCORE_OK            = 44.0
ADX_MIN                 = 18.0
EMA200_ALIGN_REQUIRED   = False
USE_CONFIRM_TF          = True

STABILITY_CYCLES    = 1
STABILITY_SLEEP_SEC = 1

RUN_FOREVER              = False
LOOP_POLL_SEC            = 3
HYSTERESIS_SCORE_DELTA   = 6.0
MIN_HOLD_SECONDS         = 90

LOG_EVERY_N_LOOPS        = 10

def log(level: str, msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {level:<5} | {msg}", flush=True)

TF = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
}
TF_SECONDS = {"M1":60, "M5":300, "M15":900, "H1":3600, "H4":14400, "D1":86400}

def _tz_now():
    if ZoneInfo:
        try:
            return datetime.now(ZoneInfo(TRADE_TZ))
        except Exception:
            pass
    return datetime.now(timezone.utc)

def _parse_windows(spec: str):
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or "-" not in part: continue
        a, b = part.split("-", 1)
        h1,m1 = map(int, a.split(":"))
        h2,m2 = map(int, b.split(":"))
        out.append((h1, m1, h2, m2))
    return out

_WINDOWS = _parse_windows(TRADING_WINDOWS)

def _in_time_window_local(symbol: str) -> bool:
    now = _tz_now()
    wd = now.isoweekday()
    is_crypto = extract_base(symbol) == "BTCUSD" or symbol.upper().startswith(("BTC", "XRP", "LTC", "ETH"))
    if not is_crypto:
        try:
            lo, hi = FX_DAYS.split("-")
            if not (int(lo) <= wd <= int(hi)):
                return False
        except Exception:
            pass
    if not _WINDOWS:
        return True
    tnow = dtime(now.hour, now.minute)
    for h1,m1,h2,m2 in _WINDOWS:
        if dtime(h1,m1) <= tnow <= dtime(h2,m2):
            return True
    return False

class MT5Watchdog:
    def __init__(self, login: int, password: str, server: str):
        self.login = login; self.password = password; self.server = server
        self.failures = 0
        self._init_ok = False

    def ensure(self) -> bool:
        ti = mt5.terminal_info()
        if ti and ti.connected:
            return True
        return self._reinit()

    def _reinit(self) -> bool:
        backoff = min(60, (2 ** min(self.failures, 6))) + random.uniform(0, 1.0)
        if self.failures > 0:
            log("WARN", f"MT5 reconnect backoff {backoff:.1f}s (failures={self.failures})")
            time.sleep(backoff)
        try:
            mt5.shutdown()
        except Exception:
            pass
        ok = mt5.initialize(login=self.login, password=self.password, server=self.server)
        if ok:
            self.failures = 0
            ti = mt5.terminal_info(); ai = mt5.account_info()
            log("INFO", f"MT5 READY | build={getattr(ti,'build',None)} | login={getattr(ai,'login',None)}")
            self._init_ok = True
            return True
        self.failures += 1
        log("ERROR", f"MT5 initialize failed: {mt5.last_error()}")
        return False

    def shutdown(self):
        try:
            mt5.shutdown()
            log("INFO", "MT5 shutdown")
        except Exception:
            pass

def symbol_is_tradable(si) -> bool:
    mode = getattr(si, "trade_mode", None)
    vis  = getattr(si, "visible", None)
    return (vis in (None, True)) and (mode == mt5.SYMBOL_TRADE_MODE_FULL)

def find_best_symbol_for(base_key: str) -> Optional[str]:
    """Танҳо номҳои мутобиқ ба ҳамин база (EURUSD/XAUUSD/BTCUSD) баррасӣ мешаванд."""
    syms = mt5.symbols_get(f"{base_key}*") or []
    candidates = [s.name for s in syms if extract_base(s.name) == base_key]
    if not candidates:
        alt = mt5.symbols_get(f"{base_key.lower()}*") or []
        candidates = [s.name for s in alt if extract_base(s.name) == base_key]

    best, best_spr = None, 1e12
    for name in candidates:
        si = mt5.symbol_info(name)
        if si is None or not symbol_is_tradable(si): continue
        t = mt5.symbol_info_tick(name)
        if not t or not si.point or t.bid <= 0 or t.ask <= 0: continue
        spr_pts = (t.ask - t.bid) / si.point
        if spr_pts < best_spr:
            best_spr = spr_pts; best = name
    return best

def resolve_symbols(bases: Optional[List[str]] = None) -> List[str]:
    """Ҳар база → як символ (спреди беҳтарин)."""
    bases = bases or list(TARGET_BASE)
    mapping: Dict[str, str] = {}
    for base in bases:
        nm = find_best_symbol_for(base)
        if nm:
            mapping[base] = nm
        else:
            log("WARN", f"Broker symbol for {base} not found")
    out = [mapping[b] for b in bases if b in mapping]
    if not out:
        raise RuntimeError("No symbols resolved. Check server/account.")
    for s in out:
        if not mt5.symbol_select(s, True):
            log("WARN", f"symbol_select({s}) failed")
    log("INFO", f"Symbols map: {mapping}")
    return out

def trim_incomplete_last_bar(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if "time" not in df.columns or not len(df): return df
    tf_sec = TF_SECONDS.get(tf, 60); now = int(time.time())
    if (now - int(df["time"].iloc[-1])) < tf_sec:
        return df.iloc[:-1].copy()
    return df

def get_rates(symbol: str, tf: str, bars: int, retries: int = 3) -> Optional[pd.DataFrame]:
    for i in range(retries):
        try:
            rr = mt5.copy_rates_from_pos(symbol, TF[tf], 0, bars)
        except Exception:
            rr = None
        if rr is not None and len(rr):
            df = pd.DataFrame(rr); df = trim_incomplete_last_bar(df, tf)
            if len(df) >= max(80, bars // 6): return df
        time.sleep(0.2 * (i + 1))
    return None

def get_tick(symbol: str) -> Optional[Dict[str, float]]:
    info = mt5.symbol_info(symbol); t = mt5.symbol_info_tick(symbol)
    if info is None or t is None or info.point == 0: return None
    if getattr(info, "trade_mode", None) != mt5.SYMBOL_TRADE_MODE_FULL: return None
    mid = (t.bid + t.ask) / 2.0; spr = (t.ask - t.bid)
    age = float(max(0, time.time() - getattr(t, "time", time.time())))
    return {"mid": float(mid), "spread": float(spr),
            "spread_points": float(spr / info.point), "tick_age": age}

class TTLCache:
    def __init__(self, ttl_sec: int):
        self.ttl = ttl_sec
        self.store: Dict[str, Tuple[float, Dict[str,float]]] = {}

    def get(self, key: str) -> Optional[Dict[str,float]]:
        now = time.time()
        item = self.store.get(key)
        if not item: return None
        ts, val = item
        if now - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: Dict[str,float]):
        self.store[key] = (time.time(), value)

_spread_cache = TTLCache(ttl_sec=max(30, min(180, SPREAD_SECONDS_BACK//2 or 60)))

def recent_spread_stats(symbol: str) -> Optional[Dict[str, float]]:
    cached = _spread_cache.get(symbol)
    if cached is not None:
        return cached

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(seconds=SPREAD_SECONDS_BACK)
    try:
        ticks = mt5.copy_ticks_range(symbol, start_dt, end_dt, mt5.COPY_TICKS_INFO)
    except Exception:
        ticks = None
    if ticks is None or len(ticks) == 0:
        return None

    df = pd.DataFrame(ticks)
    if not {"ask","bid"}.issubset(df.columns):
        return None

    df = df[(df["ask"] > 0) & (df["bid"] > 0)]
    if df.empty:
        return None

    si = mt5.symbol_info(symbol)
    if si is None or not si.point:
        return None

    sp = (df["ask"] - df["bid"]) / si.point
    sp = sp[np.isfinite(sp)]
    if len(sp) < 10:
        return None

    sp_sorted = np.sort(sp.to_numpy(dtype=float))
    lo = int(0.02 * len(sp_sorted)); hi = int(0.98 * len(sp_sorted))
    trimmed = sp_sorted[lo:hi] if hi > lo else sp_sorted

    med  = float(np.median(trimmed))
    mean = float(np.mean(trimmed))
    p95  = float(np.percentile(sp_sorted, 95))

    out = {"median_pts": med, "mean_pts": mean, "p95_pts": p95}
    _spread_cache.set(symbol, out)
    return out

def ema_slope_pct(series: np.ndarray, period: int = 20, lookback: int = 5) -> float:
    ema = talib.EMA(series, timeperiod=period)
    if ema is None or len(ema) < lookback + 1 or np.isnan(ema[-1]) or np.isnan(ema[-1 - lookback]): return 0.0
    base = series[-1] if series[-1] != 0 else 1e-9
    return float((ema[-1] - ema[-1 - lookback]) / base)

def macd_zscore(close: np.ndarray, fast=12, slow=26, signal=9, window: int = 60) -> float:
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    if macd_hist is None or len(macd_hist) < window + 1 or np.isnan(macd_hist[-1]): return 0.0
    seg = macd_hist[-window:]; mu = np.nanmean(seg); sd = np.nanstd(seg) or 1e-9
    return float((macd_hist[-1] - mu) / sd)

def bbands_feats(close: np.ndarray, period: int = 20, nbdev: float = 2.0, window: int = 120):
    upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev, matype=0)
    if upper is None or np.isnan(upper[-1]) or np.isnan(lower[-1]) or np.isnan(middle[-1]):
        return 0.0, 0.0, 0.5, float(close[-1])
    width = (upper - lower) / (middle + 1e-9)
    seg = width[-window:] if len(width) >= window else width
    mu = np.nanmean(seg); sd = np.nanstd(seg) or 1e-9
    width_z = float((width[-1] - mu) / sd)
    pos = float((close[-1] - lower[-1]) / ((upper[-1] - lower[-1]) + 1e-9))
    return float(width[-1]), width_z, pos, float(middle[-1])

def rsi_val(close: np.ndarray, period: int = 14) -> float:
    r = talib.RSI(close, timeperiod=period)
    return float(r[-1]) if r is not None and not np.isnan(r[-1]) else 50.0

def atr_val(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> float:
    a = talib.ATR(h, l, c, timeperiod=period)
    return float(a[-1]) if a is not None and not np.isnan(a[-1]) else 0.0

def adx_val(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> float:
    a = talib.ADX(h, l, c, timeperiod=period)
    return float(a[-1]) if a is not None and not np.isnan(a[-1]) else 0.0

def ema_rel200(c: np.ndarray) -> float:
    e = talib.EMA(c, timeperiod=200)
    if e is None or np.isnan(e[-1]) or np.isnan(c[-1]): return 0.0
    return float((c[-1] - e[-1]) / (c[-1] + 1e-9))

def tf_features(df: pd.DataFrame) -> Dict[str, float]:
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)

    a = atr_val(h, l, c, period=14)
    a_series = talib.ATR(h, l, c, timeperiod=14)
    a20 = float(np.nanmean(a_series[-20:])) if a_series is not None and len(a_series) >= 20 else (a or 1e-9)
    atr_ratio = (a / (a20 + 1e-9)) if a20 > 0 else 0.0
    mz = macd_zscore(c, window=min(120, max(60, len(c)//3)))
    r = rsi_val(c, period=14)
    bb_w, bb_z, bb_pos, bb_mid = bbands_feats(c, period=20, nbdev=2.0, window=min(200, max(80, len(c)//2)))
    sl = ema_slope_pct(c, period=20, lookback=5)
    ema20 = talib.EMA(c, timeperiod=20)
    tr20  = talib.ATR(h, l, c, timeperiod=20)
    if ema20 is None or tr20 is None or np.isnan(ema20[-1]) or np.isnan(tr20[-1]):
        squeeze_on = False
    else:
        kel_w = 3.0 * tr20[-1]
        bb_w_abs = abs((bb_w * (bb_mid + 1e-9)))
        squeeze_on = bool(bb_w_abs < kel_w)

    tv = None
    for col in ("tick_volume", "real_volume", "volume"):
        if col in df.columns:
            tv = df[col].to_numpy(dtype=float); break
    if tv is None or tv.size == 0:
        tickv_ratio = 1.0
    else:
        recent_n = min(3, len(tv))
        base_n = min(max(20, len(tv)//2), len(tv))
        tickv_recent = float(np.nanmean(tv[-recent_n:]))
        tickv_avg    = float(np.nanmean(tv[-base_n:])) or 1.0
        tickv_ratio  = tickv_recent / (tickv_avg + 1e-9)

    adx = adx_val(h, l, c, period=14)
    ema_rel = ema_rel200(c)

    return {
        "atr": a, "atr_ratio": atr_ratio,
        "macd_z": mz, "rsi": r,
        "bb_width": bb_w, "bb_z": bb_z, "bb_pos": bb_pos,
        "slope": sl, "tickv_ratio": tickv_ratio,
        "squeeze_on": float(squeeze_on),
        "adx": adx, "ema_rel200": ema_rel
    }

def _bars_fresh_enough(df: pd.DataFrame, tf: str) -> bool:
    if df is None or not len(df): return False
    now = int(time.time())
    tf_sec = TF_SECONDS.get(tf, 60)
    last_open = int(df["time"].iloc[-1])
    last_close = last_open + tf_sec
    limit = int(tf_sec * MAX_BAR_STALENESS_MULT)
    return (now - last_close) <= limit

# ------------------- CONFIGS -------------------
@dataclass
class StyleConfig:
    tfs: Tuple[str, str]
    bars_tf1: int
    bars_tf2: int
    spread_to_atr_max: float
    atr_floor_frac: float
    bb_width_max_z: float
    min_confluence: float
    min_tickv_ratio: float
    spread_to_atr_ok: float
    macd_min_z: float
    rsi_trend_lo: float
    rsi_trend_hi: float
    bb_width_min_z: float
    atr_ceiling_frac: float
    confirm_tf: Optional[str] = None
    bars_tf3: int = 0
    adx_min: float = ADX_MIN

def base_configs() -> Dict[str, StyleConfig]:
    return {
        "Scalping": StyleConfig(
            tfs=("M5","M15"),
            bars_tf1=500, bars_tf2=500,
            spread_to_atr_max=0.26, atr_floor_frac=0.80,
            bb_width_max_z=4.5, min_confluence=48.0,
            min_tickv_ratio=0.45, spread_to_atr_ok=0.12,
            macd_min_z=0.55, rsi_trend_lo=46, rsi_trend_hi=54,
            bb_width_min_z=0.28, atr_ceiling_frac=2.9,
            confirm_tf="H1", bars_tf3=400, adx_min=18.0
        ),
        "Intraday": StyleConfig(
            tfs=("M15","H1"),
            bars_tf1=700, bars_tf2=500,
            spread_to_atr_max=0.22, atr_floor_frac=0.88,
            bb_width_max_z=3.8, min_confluence=55.0,
            min_tickv_ratio=0.65, spread_to_atr_ok=0.13,
            macd_min_z=0.65, rsi_trend_lo=47, rsi_trend_hi=53,
            bb_width_min_z=0.30, atr_ceiling_frac=2.6,
            confirm_tf="H4", bars_tf3=400, adx_min=20.0
        ),
    }

SYMBOL_OVERRIDES: Dict[str, Dict[str, Dict[str, float]]] = {
    "Scalping": {
        "EURUSD": {"spread_to_atr_max": 0.26, "spread_to_atr_ok": 0.12,
                   "min_tickv_ratio": 0.45, "bb_width_max_z": 4.2,
                   "bb_width_min_z": 0.28, "atr_ceiling_frac": 2.9, "atr_floor_frac": 0.80},
        "XAUUSD": {"atr_floor_frac": 0.78, "spread_to_atr_max": 0.28,
                   "spread_to_atr_ok": 0.14, "min_tickv_ratio": 0.55,
                   "bb_width_max_z": 5.5, "bb_width_min_z": 0.28,
                   "atr_ceiling_frac": 2.9},
        "BTCUSD": {"atr_floor_frac": 0.80, "spread_to_atr_max": 0.34,
                   "spread_to_atr_ok": 0.18, "min_tickv_ratio": 0.60,
                   "bb_width_max_z": 6.5, "bb_width_min_z": 0.30,
                   "atr_ceiling_frac": 3.2},
    },
    "Intraday": {
        "EURUSD": {"spread_to_atr_max":0.18, "spread_to_atr_ok":0.10, "min_tickv_ratio":0.55,
                   "bb_width_max_z":3.2, "bb_width_min_z":0.28, "atr_ceiling_frac":2.3, "atr_floor_frac":0.90},
        "XAUUSD": {"spread_to_atr_max":0.22, "spread_to_atr_ok":0.13, "min_tickv_ratio":0.65,
                   "bb_width_max_z":3.8, "bb_width_min_z":0.30, "atr_ceiling_frac":2.6, "atr_floor_frac":0.88},
        "BTCUSD": {"spread_to_atr_max":0.26, "spread_to_atr_ok":0.16, "min_tickv_ratio":0.70,
                   "bb_width_max_z":4.2, "bb_width_min_z":0.32, "atr_ceiling_frac":3.0, "atr_floor_frac":0.85},
    }
}

def apply_overrides(base: StyleConfig, base_key: str, style: str) -> StyleConfig:
    ov = SYMBOL_OVERRIDES.get(style, {}).get(base_key, {})
    return replace(
        base,
        spread_to_atr_max=ov.get("spread_to_atr_max", base.spread_to_atr_max),
        spread_to_atr_ok =ov.get("spread_to_atr_ok",  base.spread_to_atr_ok),
        min_tickv_ratio  =ov.get("min_tickv_ratio",  base.min_tickv_ratio),
        bb_width_max_z   =ov.get("bb_width_max_z",   base.bb_width_max_z),
        bb_width_min_z   =ov.get("bb_width_min_z",   base.bb_width_min_z),
        atr_ceiling_frac =ov.get("atr_ceiling_frac", base.atr_ceiling_frac),
        atr_floor_frac   =ov.get("atr_floor_frac",   base.atr_floor_frac),
        min_confluence   =ov.get("min_confluence",   base.min_confluence),
    )

@dataclass
class MicroProfile:
    min_tps: float
    max_tps: float
    spread_med_x: float
    max_flips: int
    tstat_min: float

MICRO_OVERRIDES: Dict[str, MicroProfile] = {
    "EURUSD": MicroProfile(2.0, 25.0, 1.8, 12, 0.45),
    "XAUUSD": MicroProfile(2.5, 28.0, 1.7, 16, 0.50),
    "BTCUSD": MicroProfile(1.0, 35.0, 2.0, 22, 0.35),
}

def _base_key_for(symbol: str) -> str:
    """Базаи дақиқ аз номи символ; фоллбэк EURUSD."""
    b = extract_base(symbol)
    return b if b else "EURUSD"

def micro_ok(symbol: str,
             window_sec: int = 4,
             min_tps: Optional[float] = None,
             max_tps: Optional[float] = None,
             spread_med_x: Optional[float] = None,
             max_flips: Optional[int] = None,
             tstat_min: Optional[float] = None) -> Tuple[bool, str, float]:

    prof = MICRO_OVERRIDES.get(_base_key_for(symbol), MICRO_OVERRIDES["EURUSD"])
    min_tps = prof.min_tps if min_tps is None else min_tps
    max_tps = prof.max_tps if max_tps is None else max_tps
    spread_med_x = prof.spread_med_x if spread_med_x is None else spread_med_x
    max_flips = prof.max_flips if max_flips is None else max_flips
    tstat_min = prof.tstat_min if tstat_min is None else tstat_min

    try:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(seconds=window_sec)
        ticks = mt5.copy_ticks_range(symbol, start_dt, end_dt, mt5.COPY_TICKS_INFO)
        if ticks is None or len(ticks) < max(4, window_sec*2):
            return True, "", 0.0  # ҳеҷ маълумоти кифоя — иҷозат

        b = np.array([t["bid"] for t in ticks if t["bid"] > 0], dtype=float)
        a = np.array([t["ask"] for t in ticks if t["ask"] > 0], dtype=float)
        if b.size == 0 or a.size == 0:
            return False, "no_quotes", 0.0

        tps = len(ticks) / max(1.0, window_sec)
        if tps < min_tps:
            return False, "thin_tape", 0.0
        if tps > max_tps:
            return False, "storm_tape", 0.0

        spr = a - b
        cur = float(spr[-1]); med = float(np.nanmedian(spr))
        if med > 0 and cur > spread_med_x * med:
            return False, "spread_spike", 0.0

        flips = 0; last = None
        for i in range(1, len(a)):
            side = 1 if (a[i]-a[i-1]) > (b[i]-b[i-1]) else -1
            if last is not None and side != last:
                flips += 1
            last = side
        if flips > max_flips:
            return False, "quote_flips", 0.0

        mid = (a + b) / 2.0
        rets = np.diff(np.log(mid + 1e-9))
        if rets.size < 2:
            return True, "", 0.0
        mu = float(np.mean(rets))
        sd = float(np.std(rets, ddof=1)) or 1e-9
        tstat = abs(mu) / (sd / math.sqrt(max(1, rets.size)))
        if tstat < tstat_min:
            return False, "tstat_weak", float(tstat)

        return True, "", float(tstat)
    except Exception:
        return True, "", 0.0

def _slope_threshold(symbol: str) -> float:
    base = _base_key_for(symbol)
    return 0.00006 if base in ("EURUSD","XAUUSD") else 0.00012

def veto_and_confluence(
    cfg: StyleConfig,
    f1: Dict[str, float], f2: Dict[str, float],
    tk: Dict[str, float], sym: str,
    extra: Optional[Dict[str, float]] = None,
    f3: Optional[Dict[str, float]] = None,
    relaxed: bool = False
) -> Tuple[bool, float, List[str]]:
    reasons: List[str] = []

    if not _in_time_window_local(sym):
        return False, 0.0, ["time_window_off"]

    if tk.get("tick_age", 0.0) > MIN_TICK_FRESH_SEC:
        return False, 0.0, ["tick_stale"]

    if extra:
        if not extra.get("fresh_tf1", True): return False, 0.0, ["bars_stale_tf1"]
        if not extra.get("fresh_tf2", True): return False, 0.0, ["bars_stale_tf2"]

    spstats = recent_spread_stats(sym)
    if spstats is not None:
        curr_pts = tk["spread_points"]
        if not relaxed:
            if curr_pts > SPREAD_MED_MULT_MAX * spstats["median_pts"]:
                return False, 0.0, ["spread_spike_vs_median"]
            if curr_pts > 1.10 * spstats["p95_pts"]:
                return False, 0.0, ["spread_spike_p95"]
        else:
            if extra is not None:
                extra["spread_p95_ratio"] = curr_pts / max(1e-9, spstats["p95_pts"])
                extra["spread_med_ratio"] = curr_pts / max(1e-9, spstats["median_pts"])

    spread_to_atr = tk["spread"] / (f1["atr"] + 1e-9) if f1["atr"] > 0 else 1e9
    if (not relaxed and spread_to_atr > cfg.spread_to_atr_max) or \
       (relaxed and spread_to_atr > cfg.spread_to_atr_max * 1.6):
        return False, 0.0, ["spread_to_atr_high"]

    if (not relaxed and (f1["atr_ratio"] < cfg.atr_floor_frac or f2["atr_ratio"] < cfg.atr_floor_frac)) or \
       (relaxed and (f1["atr_ratio"] < 0.75 * cfg.atr_floor_frac and f2["atr_ratio"] < 0.75 * cfg.atr_floor_frac)):
        return False, 0.0, ["atr_ratio_low"]

    if (not relaxed and f1["tickv_ratio"] < cfg.min_tickv_ratio) or (relaxed and f1["tickv_ratio"] < 0.20):
        return False, 0.0, ["low_liquidity_tf1"]

    if not relaxed:
        min_slope = _slope_threshold(sym)
        if abs(f1["slope"]) < min_slope or abs(f2["slope"]) < min_slope:
            return False, 0.0, ["slope_weak"]
        rsi_margin = 1.6 if cfg.tfs[0] == "M5" else 1.8
        if (abs(f1["rsi"] - 50.0) < rsi_margin) or (abs(f2["rsi"] - 50.0) < rsi_margin):
            return False, 0.0, ["rsi_near_50"]

        adx1 = extra.get("adx1", f1.get("adx", 0.0)) if extra else f1.get("adx", 0.0)
        adx2 = extra.get("adx2", f2.get("adx", 0.0)) if extra else f2.get("adx", 0.0)
        if (adx1 < cfg.adx_min) or (adx2 < cfg.adx_min):
            return False, 0.0, ["adx_weak"]

        if EMA200_ALIGN_REQUIRED and extra and ("ema1" in extra) and ("ema2" in extra):
            s1 = np.sign(extra["ema1"]); s2 = np.sign(extra["ema2"])
            if s1 == 0 or s2 == 0 or s1 != s2:
                return False, 0.0, ["ema200_misaligned"]
            if f3 is not None and "ema3" in extra:
                s3 = np.sign(extra["ema3"])
                if s3 == 0 or s3 != s1:
                    return False, 0.0, ["ema200_misaligned_tf3"]

    same_dir = np.sign(f1["macd_z"]) == np.sign(f2["macd_z"]) and np.sign(f1["macd_z"]) != 0
    conf = 0.0
    if abs(f1["macd_z"]) >= cfg.macd_min_z: conf += 20.0
    if abs(f2["macd_z"]) >= cfg.macd_min_z: conf += 20.0
    if same_dir: conf += 15.0

    def rsi_pts(rv: float) -> float:
        if rv >= cfg.rsi_trend_hi or rv <= cfg.rsi_trend_lo: return 10.0
        return max(0.0, 10.0 * (abs(rv - 50.0) / 10.0))
    conf += rsi_pts(f1["rsi"]) + rsi_pts(f2["rsi"])

    if same_dir:
        if f1["macd_z"] > 0 and f1["rsi"] >= cfg.rsi_trend_hi and f2["rsi"] >= cfg.rsi_trend_hi: conf += 5.0
        if f1["macd_z"] < 0 and f1["rsi"] <= cfg.rsi_trend_lo and f2["rsi"] <= cfg.rsi_trend_lo: conf += 5.0

    if f1["bb_z"] >= cfg.bb_width_min_z: conf += 5.0
    if f2["bb_z"] >= cfg.bb_width_min_z: conf += 5.0
    if np.sign(f1["slope"]) == np.sign(f2["slope"]) and np.sign(f1["slope"]) != 0:
        conf += 10.0

    dyn_min = cfg.min_confluence
    adx_avg = (extra.get("adx1", 0.0) + extra.get("adx2", 0.0)) / 2.0 if extra else 0.0
    if adx_avg >= (cfg.adx_min + 8.0):      dyn_min -= 6.0
    if f1["bb_z"] >= (cfg.bb_width_min_z + 1.0): dyn_min -= 4.0
    if bool(f1.get("squeeze_on", 0.0)):   dyn_min += 4.0
    dyn_min = max(30.0, dyn_min)

    if not relaxed and conf < dyn_min:
        return False, conf, ["low_confluence"]

    return True, conf, []

def scored(
    cfg: StyleConfig,
    f1: Dict[str, float], f2: Dict[str, float],
    tk: Dict[str, float],
    extra: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    score, parts = 0.0, {}

    spread_to_atr = tk["spread"] / (f1["atr"] + 1e-9) if f1["atr"] > 0 else 1e9
    s_sp = 15.0 if spread_to_atr <= cfg.spread_to_atr_ok else max(
        0.0,
        15.0 * ((cfg.spread_to_atr_max - spread_to_atr) / (cfg.spread_to_atr_max - cfg.spread_to_atr_ok + 1e-9))
    )
    if extra:
        med_ratio = extra.get("spread_med_ratio", 1.0)
        p95_ratio = extra.get("spread_p95_ratio", 1.0)
        s_sp -= max(0.0, min(10.0, 5.0*(med_ratio-1.0) + 5.0*(p95_ratio-1.0)))
    s_sp = max(0.0, s_sp)
    score += s_sp; parts["spread"] = s_sp; parts["spread_to_atr"] = spread_to_atr

    s_atr = 10.0 * min(2.0, (f1["atr_ratio"] + f2["atr_ratio"]) / 2.0)
    if f1["atr_ratio"] > cfg.atr_ceiling_frac or f2["atr_ratio"] > cfg.atr_ceiling_frac: s_atr -= 5.0
    s_atr = max(0.0, min(15.0, s_atr)); score += s_atr; parts["atr"] = s_atr

    same_dir = np.sign(f1["macd_z"]) == np.sign(f2["macd_z"]) and np.sign(f1["macd_z"]) != 0
    s_macd = 0.0
    if abs(f1["macd_z"]) >= cfg.macd_min_z: s_macd += 10.0
    if abs(f2["macd_z"]) >= cfg.macd_min_z: s_macd += 10.0
    if same_dir: s_macd += 10.0
    score += s_macd; parts["macd"] = s_macd

    def rsi_pts(rv: float) -> float:
        if rv >= cfg.rsi_trend_hi or rv <= cfg.rsi_trend_lo: return 6.0
        return max(0.0, 6.0 * (abs(rv - 50.0) / 10.0))
    s_rsi = rsi_pts(f1["rsi"]) + rsi_pts(f2["rsi"])
    if same_dir:
        if f1["macd_z"] > 0 and f1["rsi"] >= cfg.rsi_trend_hi and f2["rsi"] >= cfg.rsi_trend_hi: s_rsi += 4.0
        if f1["macd_z"] < 0 and f1["rsi"] <= cfg.rsi_trend_lo and f2["rsi"] <= cfg.rsi_trend_lo: s_rsi += 4.0
    score += s_rsi; parts["rsi"] = s_rsi

    s_bb = 0.0
    if f1["bb_z"] >= cfg.bb_width_min_z: s_bb += 6.0
    if f2["bb_z"] >= cfg.bb_width_min_z: s_bb += 6.0
    if same_dir:
        if f1["macd_z"] > 0: s_bb += 4.0 * float(f1["bb_pos"] > 0.6) + 4.0 * float(f2["bb_pos"] > 0.6)
        if f1["macd_z"] < 0: s_bb += 4.0 * float(f1["bb_pos"] < 0.4) + 4.0 * float(f2["bb_pos"] < 0.4)
    s_bb = min(12.0, s_bb); score += s_bb; parts["bb"] = s_bb

    s_sl = 0.0
    for sl in (f1["slope"], f2["slope"]):
        s_sl += max(0.0, min(10.0, abs(sl) * 400.0))
    if same_dir and np.sign(f1["slope"]) == np.sign(f2["slope"]) and np.sign(f1["slope"]) != 0:
        s_sl += 4.0
    score += s_sl; parts["slope"] = s_sl

    s_liq = min(10.0, 10.0 * f1.get("tickv_ratio", 1.0))
    score += s_liq; parts["liquidity"] = s_liq

    if extra:
        if "adx1" in extra and "adx2" in extra:
            s_adx = min(10.0, (extra["adx1"] + extra["adx2"]) / 6.0)
            score += s_adx; parts["adx_bonus"] = s_adx
        if ("ema1" in extra) and ("ema2" in extra):
            s1, s2 = np.sign(extra["ema1"]), np.sign(extra["ema2"])
            ema_bonus = 6.0 * float(s1 != 0 and s1 == s2)
            score += ema_bonus; parts["ema_bonus"] = ema_bonus

    score = max(-100.0, min(100.0, score))
    return score, parts

# ------------------- SELECTOR -------------------
class SymbolSelector:
    def __init__(self, style: str, symbols: List[str], strict: bool = True, relaxed: Optional[bool] = None):
        assert style in ("Scalping", "Intraday")
        self.style = style
        self.symbols = symbols
        self.relaxed = RELAXED if relaxed is None else relaxed

        cfg = base_configs()[style]
        if strict and not self.relaxed:
            cfg = replace(cfg, min_confluence=cfg.min_confluence + 8.0)
        if self.relaxed:
            cfg = replace(cfg,
                min_confluence=max(0.0, cfg.min_confluence - 8.0),
                adx_min=max(8.0, cfg.adx_min - 5.0)
            )
        self.cfg_base = cfg

        # Ислоҳ: база барои ҳар символ аз extract_base
        self.base_key_map = {s: (extract_base(s) or "EURUSD") for s in symbols}

        # override-ҳои дақиқ барои EURUSD/XAUUSD/BTCUSD
        self.cfg_by_symbol = {s: apply_overrides(cfg, self.base_key_map[s], style) for s in symbols}

    def _calc_one(self, s: str) -> Dict[str, float]:
        tk = get_tick(s)
        if tk is None or tk["mid"] <= 0:
            return {"score": -1e9, "reason":"no_tick_or_disabled", "veto_ok": 0.0}

        ok_micro, why_micro, tstat = micro_ok(s)
        if not ok_micro and not self.relaxed:
            return {"score": -1e9, "reason": f"micro:{why_micro}", "veto_ok": 0.0}

        cfg = self.cfg_by_symbol[s]
        tf1, tf2 = cfg.tfs
        df1 = get_rates(s, tf1, cfg.bars_tf1); df2 = get_rates(s, tf2, cfg.bars_tf2)
        if df1 is None or df2 is None:
            return {"score": -1e9, "reason":"no_bars", "veto_ok": 0.0}

        fresh_tf1 = _bars_fresh_enough(df1, tf1)
        fresh_tf2 = _bars_fresh_enough(df2, tf2)
        if not fresh_tf1 or not fresh_tf2:
            return {"score": -1e9, "reason":"bars_stale", "veto_ok": 0.0}

        df3 = None; f3_feats = None
        if USE_CONFIRM_TF and getattr(cfg, "confirm_tf", None):
            tf3 = cfg.confirm_tf
            if tf3:
                df3 = get_rates(s, tf3, cfg.bars_tf3)

        f1 = tf_features(df1); f2 = tf_features(df2)

        c1,h1,l1 = df1["close"].to_numpy(float), df1["high"].to_numpy(float), df1["low"].to_numpy(float)
        c2,h2,l2 = df2["close"].to_numpy(float), df2["high"].to_numpy(float), df2["low"].to_numpy(float)
        adx1 = adx_val(h1,l1,c1); adx2 = adx_val(h2,l2,c2)
        ema1 = ema_rel200(c1);   ema2 = ema_rel200(c2)

        extra = {
            "fresh_tf1": fresh_tf1, "fresh_tf2": fresh_tf2,
            "adx1": adx1, "adx2": adx2, "ema1": ema1, "ema2": ema2,
            "micro_tstat": tstat
        }

        if df3 is not None and len(df3):
            c3,h3,l3 = df3["close"].to_numpy(float), df3["high"].to_numpy(float), df3["low"].to_numpy(float)
            f3_feats = tf_features(df3)
            extra["adx3"] = adx_val(h3,l3,c3)
            extra["ema3"] = ema_rel200(c3)

        spstats = recent_spread_stats(s)
        if spstats is not None:
            extra["spread_p95_breach"] = (tk["spread_points"] > 1.10 * spstats["p95_pts"])

        ok, conf, reasons = veto_and_confluence(cfg, f1, f2, tk, s, extra=extra, f3=f3_feats, relaxed=self.relaxed)
        out: Dict[str, float] = {
            "veto_ok": float(ok), "confluence": conf,
            "spread_to_atr": (tk["spread"] / (f1["atr"] + 1e-9)) if f1["atr"]>0 else 1e9,
            "f1_atr_ratio": f1["atr_ratio"], "f2_atr_ratio": f2["atr_ratio"],
            "f1_macd_z": f1["macd_z"], "f2_macd_z": f2["macd_z"],
            "f1_rsi": f1["rsi"], "f2_rsi": f2["rsi"],
            "f1_bb_z": f1["bb_z"], "f2_bb_z": f2["bb_z"],
            "f1_bb_pos": f1["bb_pos"], "f2_bb_pos": f2["bb_pos"],
            "f1_slope": f1["slope"], "f2_slope": f2["slope"],
            "f1_tickv_ratio": f1["tickv_ratio"],
            "adx1": adx1, "adx2": adx2, "ema1": ema1, "ema2": ema2,
            "tick_age": tk.get("tick_age", 0.0),
            "micro_tstat": tstat
        }
        if f3_feats:
            out.update({"f3_bb_z": f3_feats["bb_z"], "f3_rsi": f3_feats["rsi"], "f3_macd_z": f3_feats["macd_z"]})
            out["adx3"] = extra.get("adx3", 0.0); out["ema3"] = extra.get("ema3", 0.0)

        if not ok and not self.relaxed:
            out["score"] = -1e9; out["veto_reasons"] = ";".join(reasons); out["reason"]=out["veto_reasons"]; return out

        sc, parts = scored(cfg, f1, f2, tk, extra=extra)
        if tstat >= 0.6:
            sc = min(100.0, sc * (1.0 + min(0.06, 0.02*tstat)))

        out["score"] = sc
        out.update({f"part_{k}": v for k, v in parts.items()})
        if not ok and self.relaxed:
            out["reason"] = ";".join(reasons)
        else:
            out["reason"] = out.get("reason","")
        return out

    def _stable_pass(self, s: str) -> Optional[Dict[str,float]]:
        info_last = None
        for _ in range(STABILITY_CYCLES):
            res = self._calc_one(s)
            info_last = res
            if res.get("veto_ok") != 1.0 or (res.get("score",-1e9) < MIN_SCORE_OK):
                return None
            time.sleep(STABILITY_SLEEP_SEC)
        return info_last

    def choose(self, return_details: bool = False):
        raw_results: Dict[str, Dict[str, float]] = {}
        for s in self.symbols:
            try: raw_results[s] = self._calc_one(s)
            except Exception as e: raw_results[s] = {"score": -1e9, "reason": f"exception:{e}", "veto_ok": 0.0}

        if self.relaxed:
            valid = {s:r for s,r in raw_results.items() if r.get("veto_ok",0.0)==1.0}
            source = valid if valid else raw_results
            best = sorted(
                source.items(),
                key=lambda kv: (
                    -kv[1].get("score",-1e9),
                    _rank_base_of(kv[0]),                         # афзалият: XAU > EUR > BTC
                    kv[1].get("spread_to_atr",1e9),
                    -kv[1].get("confluence",0.0),
                    kv[0]
                )
            )[0][0]
            return (best, raw_results) if return_details else best

        candidates = [s for s,r in raw_results.items() if r.get("veto_ok",0.0)==1.0 and r.get("score",-1e9)>=MIN_SCORE_OK]
        if not candidates:
            best = "NO_TRADE"
            return (best, raw_results) if return_details else best

        stable_results: Dict[str, Dict[str,float]] = {}
        for s in candidates:
            st = self._stable_pass(s)
            if st is not None: stable_results[s] = st

        if not stable_results:
            best = "NO_TRADE"
            return (best, raw_results) if return_details else best

        best = sorted(stable_results.items(),
                      key=lambda kv: (
                          -kv[1]["score"],
                          _rank_base_of(kv[0]),                     # афзалият: XAU > EUR > BTC
                          kv[1]["spread_to_atr"],
                          -kv[1]["confluence"],
                          kv[0]
                      ))[0][0]
        if return_details:
            merged = {**raw_results}
            for k,v in stable_results.items(): merged[k] = v
            return best, merged
        return best

# ------------------- API -------------------
def ensure_initialized() -> MT5Watchdog:
    wd = MT5Watchdog(LOGIN, PASSWORD, SERVER)
    if not wd.ensure():
        log("WARN", "Initial MT5 connect failed; service will keep retrying.")
    return wd

def get_best_symbol(style: str = STYLE, symbols: Optional[List[str]] = None,
                    block_until_tradeable: Optional[bool] = None, poll_sec: int = 2) -> str:
    wd = ensure_initialized()
    try:
        if not wd.ensure():
            return "NO_TRADE"
        if not symbols:
            symbols = resolve_symbols(bases=TARGET_BASE)  # танҳо 3 база
        else:
            for s in symbols: mt5.symbol_select(s, True)

        use_block = (False if block_until_tradeable is None else block_until_tradeable) if RELAXED else \
                    (True if block_until_tradeable is None else block_until_tradeable)

        selector = SymbolSelector(style, symbols, strict=not RELAXED, relaxed=RELAXED)

        if RELAXED or not use_block:
            best, info = selector.choose(return_details=True)
            # for s, v in info.items():
            #     log("INFO", f"{s} | veto_ok={v.get('veto_ok')} score={round(v.get('score',0),2)} "
            #                 f"spr/ATR={round(v.get('spread_to_atr',0),3)} "
            #                 f"ATRr={round(v.get('f1_atr_ratio',0),2)}/{round(v.get('f2_atr_ratio',0),2)} "
            #                 f"BBz={round(v.get('f1_bb_z',0),2)}/{round(v.get('f2_bb_z',0),2)} "
            #                 f"BBpos={round(v.get('f1_bb_pos',0),2)}/{round(v.get('f2_bb_pos',0),2)} "
            #                 f"slope={round(v.get('f1_slope',0),6)}/{round(v.get('f2_slope',0),6)} "
            #                 f"ADX={round(v.get('adx1',0),1)}/{round(v.get('adx2',0),1)} "
            #                 f"EMA200={round(v.get('ema1',0),4)}/{round(v.get('ema2',0),4)} "
            #                 f"tick_age={round(v.get('tick_age',0),1)}s")
            # log("INFO", f"Best: {best}")
            return best

        while True:
            if not wd.ensure():
                time.sleep(poll_sec)
                continue
            best, info = selector.choose(return_details=True)
            if best != "NO_TRADE": return best
            time.sleep(poll_sec)
    finally:
        try: mt5.shutdown()
        except Exception: pass

def choose_once() -> str:
    return get_best_symbol(style=STYLE, symbols=None, block_until_tradeable=False)

# ------------------- SERVICE -------------------
class SymbolPickerService:
    def __init__(self,
                 style: str = STYLE,
                 symbols: Optional[List[str]] = None,
                 poll_sec: float = LOOP_POLL_SEC,
                 on_update: Optional[Callable[[str, Dict[str,Dict[str,float]]], None]] = None):
        self.style = style
        self.symbols = symbols
        self.poll_sec = poll_sec
        self.on_update = on_update
        self.wd = ensure_initialized()
        self.selector: Optional[SymbolSelector] = None
        self._stop = False
        self._last_choice: Optional[str] = None
        self._last_choice_time: float = 0.0
        self._loops = 0

    def stop(self, *_):
        self._stop = True
        log("INFO", "Stop signal received")

    def _should_switch(self, best: str, info: Dict[str,Dict[str,float]]) -> bool:
        if self._last_choice is None:
            return True
        if best == self._last_choice:
            return False
        now = time.time()
        if now - self._last_choice_time < MIN_HOLD_SECONDS:
            return False
        cur_score = info.get(self._last_choice, {}).get("score", -1e9)
        new_score = info.get(best, {}).get("score", -1e9)
        return (new_score - cur_score) >= HYSTERESIS_SCORE_DELTA

    def run_forever(self):
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        if not self.wd.ensure():
            log("WARN", "Waiting for MT5 connect...")
        if not self.symbols:
            while not self._stop and not self.wd.ensure():
                time.sleep(1.0)
            if self._stop: return
            self.symbols = resolve_symbols(bases=TARGET_BASE)
        else:
            for s in self.symbols: mt5.symbol_select(s, True)

        self.selector = SymbolSelector(self.style, self.symbols, strict=not RELAXED, relaxed=RELAXED)

        while not self._stop:
            self._loops += 1
            if not self.wd.ensure():
                time.sleep(self.poll_sec)
                continue
            best, info = self.selector.choose(return_details=True)
            if self._loops % LOG_EVERY_N_LOOPS == 1:
                for s, v in info.items():
                    log("INFO", f"{s} | veto_ok={v.get('veto_ok')} score={round(v.get('score',0),2)} "
                                f"spr/ATR={round(v.get('spread_to_atr',0),3)} conf={round(v.get('confluence',0),1)}")
                log("INFO", f"Candidate best: {best}")

            if best != "NO_TRADE" and self._should_switch(best, info):
                prev = self._last_choice
                self._last_choice = best
                self._last_choice_time = time.time()
                log("INFO", f"SWITCH => {best} (prev={prev})")
                if self.on_update:
                    try:
                        self.on_update(best, info)
                    except Exception as e:
                        log("WARN", f"on_update error: {e}")

            time.sleep(self.poll_sec)

        self.wd.shutdown()

# ------------------- MAIN -------------------
if __name__ == "__main__":
    if RUN_FOREVER:
        def print_update(best: str, info: Dict[str,Dict[str,float]]):
            log("INFO", f"[CALLBACK] Best symbol now: {best}")
        svc = SymbolPickerService(on_update=print_update)
        svc.run_forever()
    else:
        print(choose_once())
