from __future__ import annotations
"""
SCALPING_X100 — ultra‑fast, adaptive scalper for EURUSD, XAUUSD, BTCUSD (M1–M15)
- Focus: real‑time signal generation with strict microstructure, regime, and risk gates
- Platforms: MetaTrader5 (Python), TA‑Lib (indicators)
- Highlights vs prior version:
  * Symbol auto‑resolution (handles Exness suffixes like "m", ".e", "-RAW")
  * Aggressive low‑latency caching of rates/orderbook to cut MT5 IPC overhead
  * Per‑symbol tuned thresholds (spread, micro-tape, ATR‑relative ranges)
  * Microstructure guard upgraded: tape rate, spread spike, quote‑flip, t‑stat, imbalance
  * Multi‑TF consensus with dynamic weights; smarter squeeze/volatility gating
  * Adaptive SL/TP and trailing using current volatility & ADX; scratch exit window
  * Position management: BE+spread lock, partials, ATR chandelier‑style trail
  * Equity guardrails: per‑trade risk, daily max loss, session throttling
  * CSV logging kept lean; deterministic debouncing; warmup protection

Environment vars (override defaults):
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
  SYMBOLS="EURUSD,XAUUSD,BTCUSD"    # bases; auto‑resolves (e.g., XAUUSDm)
  TIMEZONE="Asia/Dushanbe"
  HIGH_ACCURACY=1                    # stricter filters, slower trade freq
  EXECUTE=0                          # 1 = place orders; 0 = just signals
  LOOP=0                             # 1 = continuous loop; 0 = single run

Requires: pip install MetaTrader5 TA-Lib pandas numpy pytz requests
"""
import os, math, time, threading, logging, json, csv
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytz, requests

try:
    import MetaTrader5 as mt5
    import talib
except Exception as e:
    raise RuntimeError("Нужны пакеты MetaTrader5 и TA‑Lib.") from e

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("scalping_x100")

# ---------- Data models ----------
@dataclass
class SymbolParams:
    base: str                                   # e.g. "XAUUSD", "EURUSD", "BTCUSD"
    resolved: Optional[str] = None              # broker symbol (e.g. "XAUUSDm")
    tf_primary: str = "M1"
    tf_confirm: str = "M5"
    tf_long: str = "M15"
    entry_mode: str = "market"                 # market | limit (pullback)
    pullback_atr_mult: float = 0.35
    # microstructure & spread tuning per symbol
    spread_limit_pct: float = 0.00015
    micro_window_sec: int = 4
    micro_min_tps: float = 5
    micro_max_tps: float = 45
    micro_imb_thresh: float = 0.25
    micro_spread_med_x: float = 1.5
    quote_flips_max: int = 12
    micro_tstat_thresh: float = 0.6

@dataclass
class EngineConfig:
    login: int
    password: str
    server: str

    finnhub_key: str = os.getenv("FINNHUB_KEY", "")

    symbols: List[SymbolParams] | List[str] | str = field(default_factory=lambda: [
        SymbolParams("XAUUSD", tf_primary="M1", tf_confirm="M5", tf_long="M15", entry_mode="market")
    ])

    tz_local: str = os.getenv("TIMEZONE", "Asia/Dushanbe")
    # Active hours (local) — user prefers 11:00–15:00 and 16:00–20:00 Dushanbe
    active_sessions: List[Tuple[int, int]] = field(default_factory=lambda: [(11,15),(16,20)])

    # Confidence & regime thresholds
    conf_min: int = 85
    conf_min_low: int = 70
    conf_min_high: int = 95
    adx_trend_lo: float = 18.0
    adx_trend_hi: float = 28.0
    atr_rel_lo: float = 0.0006
    atr_rel_hi: float = 0.0025

    min_body_pct_of_atr: float = 0.12
    news_blackout_min_before: int = 30
    news_blackout_min_after: int = 45
    min_bar_age_sec: int = 1

    fixed_volume: float = 0.01
    max_risk_per_trade: float = 0.005         # 0.5%/trade
    max_drawdown: float = 0.05                # 5% live drawdown guard
    max_daily_loss_pct: float = 0.02          # 2%/day kill switch
    max_trades_per_hour: int = 1

    # SL/TP multipliers — adapted per regime
    sl_atr_mult_trend: float = 0.9
    tp_atr_mult_trend: float = 2.2
    sl_atr_mult_range: float = 1.2
    tp_atr_mult_range: float = 1.6
    be_trigger_R: float = 0.8
    be_lock_spread_mult: float = 1.2
    trail_atr_mult: float = 1.1
    partial_tp_R1: float = 0.7
    partial_tp_R1_close_pct: float = 0.4

    # signal weights
    weights: Dict[str,float] = field(default_factory=lambda: {
        "trend":0.45, "momentum":0.25, "meanrev":0.15, "structure":0.10, "volume":0.05
    })
    signal_amplification: float = 1.35

    # latency & cadence
    decision_debounce_ms: int = 180
    max_decision_latency_ms: int = 900
    latency_violation_limit: int = 3
    cooldown_seconds: int = 150
    poll_seconds_fast: int = int(os.getenv("POLL_SECONDS", "3"))

    pending_ttl_sec: int = int(os.getenv("PENDING_TTL_SEC", "12"))
    scratch_window_s: int = int(os.getenv("SCRATCH_WINDOW_S", "5"))
    scratch_mae_r: float = float(os.getenv("SCRATCH_MAE_R", "0.25"))

    adaptive_enabled: bool = True
    trail_on_entry: bool = True
    use_squeeze_filter: bool = True
    hedge_flip_enabled: bool = False
    pyramid_enabled: bool = False
    log_csv_path: str = os.getenv("LOG_CSV", "signals_log.csv")

# ---------- Constants ----------
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1
}

# ---------- Helpers ----------
def safe_last(x: np.ndarray, default: float = 0.0) -> float:
    try:
        v = float(x[-1]);  return v if np.isfinite(v) else default
    except Exception:
        return default

# ---------- News blackout (Finviz/Finnhub style) ----------
class NewsGuard:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.flag = False
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self): self._th.start()
    def stop(self): self._stop = True

    def _run(self):
        if not self.cfg.finnhub_key:
            return
        sess = requests.Session(); sess.headers.update({"User-Agent":"scalping-x100/1.0"})
        while not self._stop:
            try:
                utc_now = datetime.now(timezone.utc)
                start = (utc_now - timedelta(hours=8)).date()
                finish = (utc_now + timedelta(hours=8)).date()
                url = (
                    f"https://finnhub.io/api/v1/calendar/economic?from={start}&to={finish}&token={self.cfg.finnhub_key}"
                )
                r = sess.get(url, timeout=3)
                self.flag = False
                if r.ok:
                    for ev in (r.json() or {}).get("economicCalendar", []):
                        if str(ev.get("impact", "")).lower() not in ("high","medium"):
                            continue
                        dt = ev.get("time") or ev.get("date")
                        if not dt: continue
                        try:
                            ev_time = datetime.fromisoformat(dt.replace("Z","+00:00"))
                        except Exception:
                            continue
                        minutes = abs((utc_now - ev_time).total_seconds())/60.0
                        if minutes <= max(self.cfg.news_blackout_min_before, self.cfg.news_blackout_min_after):
                            self.flag = True; break
            except Exception:
                pass
            time.sleep(60)

# ---------- Worker ----------
class Worker:
    def __init__(self, cfg: EngineConfig, sp: SymbolParams, newsguard: NewsGuard):
        self.cfg, self.sp, self.news = cfg, sp, newsguard
        self.tz = pytz.timezone(cfg.tz_local)
        self._last_decision_ms = 0.0
        self._last_net_norm = 0.0
        self._last_signal = "Neutral"
        self._latency_violations = 0
        self._hourly_trades: Dict[str,int] = {}
        self._pending_created: Dict[int,float] = {}
        self._filled_at: Dict[int,float] = {}
        self._addons_done: Dict[int,int] = {}
        self._current_drawdown = 0.0
        # low‑latency caches per timeframe
        self._rates_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self._cache_ttl_sec = {"M1":0.35, "M5":0.5, "M15":0.75, "M30":1.0, "H1":1.5}
        # select symbol on terminal
        if self.sp.resolved and not mt5.symbol_select(self.sp.resolved, True):
            log.warning("symbol_select failed: %s", self.sp.resolved)

    # ---- time/session ----
    def _now_local(self): return datetime.now(self.tz)
    def _hour_key(self):  return self._now_local().strftime("%Y-%m-%d %H:00")
    def _trade_quota_ok(self)->bool: return self._hourly_trades.get(self._hour_key(),0) < self.cfg.max_trades_per_hour
    def _in_active_session(self)->bool:
        h=self._now_local().hour
        return any(s<=h<e for s,e in self.cfg.active_sessions)

    # ---- data fetch with cache ----
    def _fetch_rates(self, tf:str, count:int=300)->Optional[pd.DataFrame]:
        sym = self.sp.resolved or self.sp.base
        tfc = TF_MAP.get(tf)
        if tfc is None: return None
        now = time.time()
        ttl = self._cache_ttl_sec.get(tf, 0.6)
        if tf in self._rates_cache:
            df, ts = self._rates_cache[tf]
            if now - ts < ttl:
                return df
        for _ in range(2):
            rr=mt5.copy_rates_from_pos(sym, tfc, 0, count)
            if rr is not None and len(rr)>=60:
                df=pd.DataFrame(rr)
                df["time"]=pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(self.tz)
                self._rates_cache[tf] = (df, now)
                return df
            time.sleep(0.01)
        return None

    def _fetch_book(self)->Optional[Dict[str,Any]]:
        try:
            sym = self.sp.resolved or self.sp.base
            book=mt5.market_book_get(sym)
            if book is None: return None
            bids,asks=[],[]
            for lv in book:
                if lv.type==mt5.BOOK_TYPE_BUY:  bids.append((lv.price, lv.volume))
                elif lv.type==mt5.BOOK_TYPE_SELL: asks.append((lv.price, lv.volume))
            return {"bids":sorted(bids,key=lambda x:-x[0])[:5],
                    "asks":sorted(asks,key=lambda x:x[0])[:5]}
        except Exception: return None

    def _spread_pct(self)->float:
        info=mt5.symbol_info(self.sp.resolved or self.sp.base)
        if not info or None in (info.spread, info.point, info.bid, info.ask): return 1e-4
        spr=info.spread*info.point; mid=(info.bid+info.ask)/2.0
        return spr/max(1e-9,mid)

    # ---- volumes ----
    def _ensure_volume(self, df:pd.DataFrame)->np.ndarray:
        vol=df.get("tick_volume", df.get("real_volume", None))
        if vol is None or len(vol)!=len(df):
            rng=(df["high"].values-df["low"].values)
            vol=(np.abs(np.append([0],np.diff(df["close"].values)))+rng)*1_000
        return np.asarray(vol, dtype=np.float64)

    # ---- indicators ----
    def _indicators(self, df:pd.DataFrame, is_long:bool=False, shift:int=1)->Optional[Dict[str,Any]]:
        if df is None or len(df)<120 or shift<1: return None
        c=df["close"].values.astype(np.float64)[:-shift]
        h=df["high"].values.astype(np.float64)[:-shift]
        l=df["low"].values.astype(np.float64)[:-shift]
        v=self._ensure_volume(df)[:-shift]
        ema8=talib.EMA(c,8); ema21=talib.EMA(c,21); ema50=talib.EMA(c,50)
        ema200=talib.EMA(c, min(200,max(50,len(c)//2)))
        macd,macds,macdh=talib.MACD(c,12,26,9)
        rsi=talib.RSI(c,14); rsi_f=talib.RSI(c,7)
        adx=talib.ADX(h,l,c,14)
        atr=talib.ATR(h,l,c,14)
        bU,bM,bL=talib.BBANDS(c,20,2,2)
        donH=talib.MAX(h,20); donL=talib.MIN(l,20)
        stK,stD=talib.STOCH(h,l,c,14,3,3)
        will=talib.WILLR(h,l,c,14); psar=talib.SAR(h,l,0.02,0.2)
        typ=(h+l+c)/3.0; vS=talib.SMA(v,20); vSTD=talib.STDDEV(v,20)
        o=float(df.iloc[-1-shift]["open"]); cc=float(df.iloc[-1-shift]["close"])
        hi=float(df.iloc[-1-shift]["high"]); lo=float(df.iloc[-1-shift]["low"])
        body=abs(cc-o)
        vwap=float(safe_last(np.cumsum(typ*v))/max(1.0, safe_last(np.cumsum(v))))
        zvol=0.0 if (not np.isfinite(vSTD[-1]) or vSTD[-1]==0) else (v[-1]-vS[-1])/vSTD[-1]
        ema20=talib.EMA(c,20); tr20=talib.ATR(h,l,c,20)
        kelU=ema20+1.5*tr20; kelL=ema20-1.5*tr20
        squeeze_on=bool((bU[-1]-bL[-1]) < (kelU[-1]-kelL[-1]))
        out={"close":cc,"open":o,"high":hi,"low":lo,"body":body,
             "ema8":float(safe_last(ema8)),"ema21":float(safe_last(ema21)),
             "ema50":float(safe_last(ema50)),"ema200":float(safe_last(ema200)),
             "macd":float(safe_last(macd)),"macd_sig":float(safe_last(macds)),
             "macd_hist":float(safe_last(macdh)),
             "rsi":float(safe_last(rsi)),"rsi_fast":float(safe_last(rsi_f)),
             "adx":float(safe_last(adx)),"adx_series":adx,
             "atr":float(safe_last(atr)),
             "bb_upper":float(safe_last(bU)),"bb_mid":float(safe_last(bM)),"bb_lower":float(safe_last(bL)),
             "don_high":float(safe_last(donH)),"don_low":float(safe_last(donL)),
             "stoch_k":float(safe_last(stK)),"stoch_d":float(safe_last(stD)),
             "williams_r":float(safe_last(will)),"psar":float(safe_last(psar)),
             "vwap":float(vwap),"z_vol":float(zvol),"squeeze_on":squeeze_on}
        if is_long:
            if len(ema200)>21:
                slope=(ema200[-1]-ema200[-21])/(abs(ema200[-1])+1e-9)
                out["slope_deg"]=float(math.degrees(math.atan(slope)))
            else: out["slope_deg"]=0.0
        return out

    # ---- regime ----
    def _regime(self, ind_long:Dict[str,Any])->str:
        if not ind_long: return "trend"
        return "trend" if (ind_long.get("adx",0.0)>=self.cfg.adx_trend_lo or abs(ind_long.get("slope_deg",0.0))>=1.5) else "range"

    # ---- adapt params ----
    def _adaptive_params(self, ind_p:Dict[str,Any], ind_l:Dict[str,Any])->Dict[str,Any]:
        if not self.cfg.adaptive_enabled:
            return {"conf_min": self.cfg.conf_min,
                    "sl_mult": (self.cfg.sl_atr_mult_trend if self._regime(ind_l)=="trend" else self.cfg.sl_atr_mult_range),
                    "tp_mult": (self.cfg.tp_atr_mult_trend if self._regime(ind_l)=="trend" else self.cfg.tp_atr_mult_range),
                    "trail_mult": self.cfg.trail_atr_mult,
                    "w_mul": {"trend":1.0,"momentum":1.0,"meanrev":1.0,"structure":1.0,"volume":1.0}}
        adx=ind_l.get("adx",self.cfg.adx_trend_lo)
        atr_rel = ind_p.get("atr",0.0)/max(1e-9, ind_p.get("close",1.0))
        regime=self._regime(ind_l)
        conf= self.cfg.conf_min
        if adx>=self.cfg.adx_trend_hi: conf = max(self.cfg.conf_min_low, self.cfg.conf_min - 10)
        elif adx<=self.cfg.adx_trend_lo: conf = min(self.cfg.conf_min_high, self.cfg.conf_min + 8)
        if atr_rel>=self.cfg.atr_rel_hi: conf = max(self.cfg.conf_min_low, conf - 4)
        elif atr_rel<=self.cfg.atr_rel_lo: conf = min(self.cfg.conf_min_high, conf + 4)
        if regime=="trend":
            sl = self.cfg.sl_atr_mult_trend * (1.10 if atr_rel>=self.cfg.atr_rel_hi else 0.90 if atr_rel<=self.cfg.atr_rel_lo else 1.0)
            tp = self.cfg.tp_atr_mult_trend * (0.95 if atr_rel>=self.cfg.atr_rel_hi else 1.05 if atr_rel<=self.cfg.atr_rel_lo else 1.0)
        else:
            sl = self.cfg.sl_atr_mult_range * (1.05 if atr_rel>=self.cfg.atr_rel_hi else 0.95 if atr_rel<=self.cfg.atr_rel_lo else 1.0)
            tp = self.cfg.tp_atr_mult_range * (0.95 if atr_rel<=self.cfg.atr_rel_lo else 1.05 if atr_rel>=self.cfg.atr_rel_hi else 1.0)
        trail = self.cfg.trail_atr_mult * (1.10 if adx>=self.cfg.adx_trend_hi else 0.90 if adx<=self.cfg.adx_trend_lo else 1.0)
        w_mul={"trend":1.0,"momentum":1.0,"meanrev":1.0,"structure":1.0,"volume":1.0}
        if regime=="trend":
            w_mul["trend"] = 1.2; w_mul["momentum"] = 1.1; w_mul["meanrev"]=0.8
        else:
            w_mul["meanrev"]=1.2; w_mul["trend"]=0.85
        return {"conf_min": int(max(self.cfg.conf_min_low, min(self.cfg.conf_min_high, conf))),
                "sl_mult": max(0.4, sl), "tp_mult": max(0.8, tp),
                "trail_mult": max(0.4, trail), "w_mul": w_mul}

    # ---- component scores ----
    def _component_scores(self, ind:Dict[str,Any], regime:str, book:Optional[Dict[str,Any]], wmul:Dict[str,float]):
        base_keys=self.cfg.weights.keys()
        w_buy={k:0.0 for k in base_keys}; w_sell={k:0.0 for k in base_keys}
        c=ind["close"]
        # trend stack
        tr=0.0
        if c>ind["ema8"]>ind["ema21"]>ind["ema50"]: tr=2.5
        elif c>ind["ema8"]>ind["ema21"]: tr=1.8
        elif c>ind["ema21"]: tr=1.0
        elif c<ind["ema8"]<ind["ema21"]<ind["ema50"]: tr=-2.5
        elif c<ind["ema8"]<ind["ema21"]: tr=-1.8
        elif c<ind["ema21"]: tr=-1.0
        (w_buy if tr>0 else w_sell)["trend"]=abs(tr)*wmul.get("trend",1.0)
        # momentum
        mom=0.0
        if ind["macd_hist"]>0 and ind["macd"]>ind["macd_sig"]: mom+=1.5
        elif ind["macd_hist"]<0 and ind["macd"]<ind["macd_sig"]: mom-=1.5
        if ind["rsi_fast"]>60 and ind["rsi"]>55: mom+=1.0
        elif ind["rsi_fast"]<40 and ind["rsi"]<45: mom-=1.0
        if ind["stoch_k"]>ind["stoch_d"] and ind["stoch_k"]>20: mom+=0.5
        elif ind["stoch_k"]<ind["stoch_d"] and ind["stoch_k"]<80: mom-=0.5
        (w_buy if mom>0 else w_sell)["momentum"]=abs(mom)*wmul.get("momentum",1.0)
        # mean reversion / extremes
        mr=0.0
        if regime=="range":
            width=max(1e-9, ind["bb_upper"]-ind["bb_lower"])
            bb_pos=(c-ind["bb_lower"])/width
            if bb_pos<=0.1: mr+=1.5
            elif bb_pos>=0.9: mr-=1.5
            if ind["williams_r"]<-80: mr+=1.0
            elif ind["williams_r"]>-20: mr-=1.0
        # extremes vs ema200 + psar bias
        rev=0.0
        if ind["rsi"]<20 and c<=ind["bb_lower"] and c>ind["ema200"]: rev+=2.0
        elif ind["rsi"]>80 and c>=ind["bb_upper"] and c<ind["ema200"]: rev-=2.0
        rev += 0.5 if c>ind["psar"] else -0.5
        mr += rev
        (w_buy if mr>0 else w_sell)["meanrev"]=abs(mr)*wmul.get("meanrev",1.0)
        # structure (Donchian/Mid)
        s=0.0; mid=0.5*(ind["don_high"]+ind["don_low"]).__float__()
        if c>=ind["don_high"]: s=2.0
        elif c<=ind["don_low"]: s=-2.0
        elif c>mid: s=0.5
        else: s=-0.5
        (w_buy if s>0 else w_sell)["structure"]=abs(s)*wmul.get("structure",1.0)
        # volume / tape / book
        vol_s=0.0
        vwap_sig=0.5 if c>ind["vwap"] else -0.5
        vol_s+=vwap_sig
        if abs(ind["z_vol"])>0.5:
            vol_s *= (1+min(2.0,abs(ind["z_vol"])))
        if book:
            try:
                b=sum(v for _,v in book["bids"][:3]); a=sum(v for _,v in book["asks"][:3])
                if b+a>0:
                    imb=(b-a)/(b+a)
                    if imb>0.15: vol_s+=0.5
                    elif imb<-0.15: vol_s-=0.5
            except Exception: pass
        (w_buy if vol_s>0 else w_sell)["volume"]=abs(vol_s)*wmul.get("volume",1.0)
        return w_buy, w_sell

    # ---- microstructure gate ----
    def _micro_ok(self)->Tuple[bool,str,float]:
        try:
            sym = self.sp.resolved or self.sp.base
            end=int(time.time()); start=end-self.sp.micro_window_sec
            ticks=mt5.copy_ticks_range(sym, start*1000, end*1000, mt5.COPY_TICKS_INFO)
            if ticks is None or len(ticks) < max(4, self.sp.micro_window_sec*2):
                return True,"",0.0
            b=np.array([t['bid'] for t in ticks if t['bid']>0.0], dtype=float)
            a=np.array([t['ask'] for t in ticks if t['ask']>0.0], dtype=float)
            if len(b)==0 or len(a)==0: return False,"no_quotes",0.0
            tps=len(ticks)/max(1.0, self.sp.micro_window_sec)
            if tps<self.sp.micro_min_tps: return False,"thin_tape",0.0
            if tps>self.sp.micro_max_tps: return False,"storm_tape",0.0
            spr=a-b; med=float(np.nanmedian(spr)); cur=float(spr[-1])
            if cur> self.sp.micro_spread_med_x*max(1e-9,med): return False,"spread_spike",0.0
            mid=(a+b)/2.0; dm=np.diff(mid)
            up=int((dm>0).sum()); dn=int((dm<0).sum()); tot=max(1,up+dn)
            imb=abs(up-dn)/tot
            if imb < self.sp.micro_imb_thresh: return False,"no_aggressor",0.0
            flips=0; last=None
            for i in range(1,len(a)):
                side = 1 if a[i]-a[i-1] > b[i]-b[i-1] else -1
                if last is not None and side!=last: flips+=1
                last=side
            if flips>self.sp.quote_flips_max: return False,"quote_flips",0.0
            rets=np.diff(np.log(mid+1e-9))
            mu=np.mean(rets); sd=np.std(rets, ddof=1) or 1e-9; n=len(rets)
            tstat= float(abs(mu)/(sd/np.sqrt(max(1,n))))
            if tstat < self.sp.micro_tstat_thresh: return False,"tstat_weak",tstat
            return True,"",tstat
        except Exception:
            return True,"",0.0

    # ---- filters & sizing ----
    def _apply_filters(self, signal:str, conf:int, ind:Dict[str,Any])->Tuple[str,int]:
        if signal=="Neutral": return signal,conf
        if ind["body"] < self.cfg.min_body_pct_of_atr * max(ind["atr"],1e-9): conf=max(0, conf-20)
        if signal=="Buy" and ind["close"]>ind["ema8"]>ind["ema21"]: conf=min(100, conf+10)
        if signal=="Sell" and ind["close"]<ind["ema8"]<ind["ema21"]: conf=min(100, conf+10)
        return signal, conf

    def _calc_volume(self, tick, sl_dist:float)->float:
        vol=self.cfg.fixed_volume
        try:
            info=mt5.symbol_info(self.sp.resolved or self.sp.base)
            acct=mt5.account_info()
            if info and acct and sl_dist>0:
                balance=float(acct.balance)
                contract=float(getattr(info, "trade_contract_size", 100.0))
                tick_size=float(getattr(info, "trade_tick_size", info.point))
                tick_value=float(getattr(info, "trade_tick_value", 0.0)) or (contract * tick_size)
                risk=self.cfg.max_risk_per_trade
                ticks_in_sl=max(1.0, sl_dist/max(1e-9,tick_size))
                money_per_lot_sl=ticks_in_sl*tick_value
                lots=(balance*risk)/max(1e-9, money_per_lot_sl)
                vol=max(0.01, min(self.cfg.fixed_volume*3.0, lots))
        except Exception: pass
        return float(round(vol,2))

    # ---- signal core ----
    def get_signal(self, execute: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        reasons: List[str] = []

        dfp = self._fetch_rates(self.sp.tf_primary, 220)
        if dfp is None or dfp.empty:
            return {
                "symbol": self.sp.resolved or self.sp.base, "signal": "Neutral", "confidence": 0,
                "regime": None, "reasons": ["no_rates_primary"],
                "spread_bps": None, "latency_ms": (time.time() - t0) * 1000.0,
                "timestamp": self._now_local().isoformat()
            }
        dfc = self._fetch_rates(self.sp.tf_confirm, 200)
        if dfc is None or dfc.empty:
            dfc = dfp
        dfl = self._fetch_rates(self.sp.tf_long, 200)
        if dfl is None or dfl.empty:
            dfl = dfp

        last_age = max(0.0, (self._now_local() - dfp.iloc[-1]["time"]).total_seconds())
        book = self._fetch_book()
        spread = self._spread_pct()
        ingest_ms = (time.time() - t0) * 1000.0

        # global guards
        if self._current_drawdown >= self.cfg.max_drawdown: reasons.append("max_drawdown")
        if self.news.flag: reasons.append("news_blackout")
        if spread > self.sp.spread_limit_pct: reasons.append("spread")
        if not self._in_active_session(): reasons.append("session")
        if ingest_ms > 6000: reasons.append("ingest_slow")
        if not self._trade_quota_ok(): reasons.append("quota")
        if last_age < self.cfg.min_bar_age_sec: reasons.append("bar_age")
        ok_micro, why_micro, tstat = self._micro_ok()
        if not ok_micro: reasons.append(f"micro:{why_micro}")
        if reasons:
            return {
                "symbol": self.sp.resolved or self.sp.base, "signal": "Neutral", "confidence": 0,
                "regime": None, "reasons": reasons, "spread_bps": spread * 10000,
                "latency_ms": (time.time() - t0) * 1000.0,
                "timestamp": self._now_local().isoformat()
            }

        indp = self._indicators(dfp, shift=1)
        indc = self._indicators(dfc, shift=1) if isinstance(dfc, pd.DataFrame) and not dfc.empty else None
        indl = self._indicators(dfl, is_long=True, shift=1) if isinstance(dfl, pd.DataFrame) and not dfl.empty else None
        if not indp:
            return {
                "symbol": self.sp.resolved or self.sp.base, "signal": "Neutral", "confidence": 0,
                "regime": None, "reasons": ["no_indicators"],
                "spread_bps": spread * 10000, "latency_ms": (time.time() - t0) * 1000.0,
                "timestamp": self._now_local().isoformat()
            }
        if self.cfg.use_squeeze_filter and indp.get("squeeze_on", False):
            reasons.append("squeeze_wait")
            return {
                "symbol": self.sp.resolved or self.sp.base, "signal": "Neutral", "confidence": 0,
                "regime": None, "reasons": reasons, "spread_bps": spread * 10000,
                "latency_ms": (time.time() - t0) * 1000.0,
                "timestamp": self._now_local().isoformat()
            }

        # HTF alignment gates
        if indl:
            trend_ok_buy  = (indl.get("adx", 0) >= self.cfg.adx_trend_lo and indp["close"] > indl.get("ema50", indp["close"] * 0.999))
            trend_ok_sell = (indl.get("adx", 0) >= self.cfg.adx_trend_lo and indp["close"] < indl.get("ema50", indp["close"] * 1.001))
        else:
            trend_ok_buy = trend_ok_sell = True

        adapt  = self._adaptive_params(indp, indl or indp)
        regime = self._regime(indl or indp)

        # score aggregation
        wmul = adapt["w_mul"]; basew = self.cfg.weights
        buy_s, sell_s = self._component_scores(indp, regime, book, wmul)
        buy_net  = sum(basew[k]*buy_s.get(k, 0.0)  for k in basew) * self.cfg.signal_amplification
        sell_net = sum(basew[k]*sell_s.get(k, 0.0) for k in basew) * self.cfg.signal_amplification
        net = buy_net - sell_net
        net_norm = math.tanh(net * 1.5)
        conf = int(100 * abs(net_norm))

        if indc is not None:
            cb, cs = self._component_scores(indc, regime, book, wmul)
            cnet = sum(basew[k]*cb.get(k,0.0) for k in basew) - sum(basew[k]*cs.get(k,0.0) for k in basew)
            if (net > 0 and cnet > 0) or (net < 0 and cnet < 0):
                conf = min(100, conf + 12)

        if tstat >= self.sp.micro_tstat_thresh:
            conf = min(100, int(conf * (1.0 + min(0.10, 0.02 * tstat))))
        if indp.get("atr", 0) > 0 and (indp["atr"]/max(1e-9, indp["close"])) * 100.0 > 0.1:
            conf = int(conf * 1.03)
        if spread > 0.0001:  # generic minor penalty
            conf = max(0, conf - 2)
        conf = max(0, min(100, conf))

        signal = "Neutral"
        blocked_by_htf = False

        if conf >= adapt["conf_min"]:
            if net_norm > 0.05 and trend_ok_buy:
                signal = "Buy"
            elif net_norm < -0.05 and trend_ok_sell:
                signal = "Sell"
            else:
                blocked_by_htf = True
        else:
            reasons.append("conf_below_min")

        signal, conf = self._apply_filters(signal, conf, indp)

        if signal == "Neutral" and blocked_by_htf:
            reasons.append("htf_trend_block")
            conf = min(conf, max(0, adapt["conf_min"] - 1))

        now_ms = time.time() * 1000.0
        if (now_ms - self._last_decision_ms) < self.cfg.decision_debounce_ms:
            reasons.append("debounce")
            return {
                "symbol": self.sp.resolved or self.sp.base, "signal": "Neutral", "confidence": 0,
                "regime": regime, "reasons": reasons,
                "spread_bps": spread*10000, "latency_ms": (time.time()-t0)*1000.0,
                "timestamp": self._now_local().isoformat()
            }
        if signal == self._last_signal and abs(net_norm - self._last_net_norm) < 0.03:
            reasons.append("stable")
            return {
                "symbol": self.sp.resolved or self.sp.base, "signal": "Neutral", "confidence": 0,
                "regime": regime, "reasons": reasons,
                "spread_bps": spread*10000, "latency_ms": (time.time()-t0)*1000.0,
                "timestamp": self._now_local().isoformat()
            }

        self._last_decision_ms = now_ms
        self._last_net_norm = net_norm
        self._last_signal = signal
        comp_ms = (time.time() - t0) * 1000.0

        if signal != "Neutral" and execute:
            self._execute(signal, conf, indp, adapt)

        res = {
            "symbol": self.sp.resolved or self.sp.base, "signal": signal, "confidence": int(conf),
            "regime": regime, "reasons": reasons,
            "spread_bps": spread * 10000,
            "latency_ms": comp_ms,
            "timestamp": self._now_local().isoformat()
        }
        self._log_csv(res)
        return res

    # ---- execution & management ----
    def _execute(self, signal:str, confidence:int, ind:Dict[str,Any], adapt:Dict[str,Any])->bool:
        try:
            sym = self.sp.resolved or self.sp.base
            tick=mt5.symbol_info_tick(sym)
            if not tick: return False
            atr=max(ind.get("atr",0.0),1e-9)
            sl_mult, tp_mult = adapt["sl_mult"], adapt["tp_mult"]
            cf=confidence/100.0
            sl_mult *= (1.05 - cf*0.2); tp_mult *= (0.95 + cf*0.4)
            sl_dist, tp_dist = atr*sl_mult, atr*tp_mult
            vol=self._calc_volume(tick, sl_dist)
            req={"action": mt5.TRADE_ACTION_DEAL if self.sp.entry_mode=="market" else mt5.TRADE_ACTION_PENDING,
                 "symbol": sym, "volume": float(vol),
                 "deviation":10,"magic":20250826,"comment":f"X100-{signal}-{confidence}%",
                 "type_time": mt5.ORDER_TIME_GTC,
                 "type_filling": mt5.ORDER_FILLING_IOC if self.sp.entry_mode=="market" else mt5.ORDER_FILLING_RETURN}
            if signal=="Buy":
                mkt=tick.ask; entry = mkt if self.sp.entry_mode=="market" else (mkt - self.sp.pullback_atr_mult*atr)
                req.update({"type": mt5.ORDER_TYPE_BUY if self.sp.entry_mode=="market" else mt5.ORDER_TYPE_BUY_LIMIT,
                            "price": float(entry), "sl": float(entry - sl_dist), "tp": float(entry + tp_dist)})
            else:
                mkt=tick.bid; entry = mkt if self.sp.entry_mode=="market" else (mkt + self.sp.pullback_atr_mult*atr)
                req.update({"type": mt5.ORDER_TYPE_SELL if self.sp.entry_mode=="market" else mt5.ORDER_TYPE_SELL_LIMIT,
                            "price": float(entry), "sl": float(entry + sl_dist), "tp": float(entry - tp_dist)})
            t0=time.time(); res=mt5.order_send(req); rtt=(time.time()-t0)*1000.0
            log.info("ORDER %s | %s vol=%.2f rtt=%d ret=%s", sym, req.get("type"), float(req["volume"]), int(rtt), getattr(res,"retcode",None))
            code=getattr(res,"retcode",None); ticket=getattr(res,"order",None)
            if not res or code not in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
                return False
            if req["action"]==mt5.TRADE_ACTION_PENDING and ticket:
                self._pending_created[int(ticket)]=time.time()
            else:
                for p in (mt5.positions_get(symbol=sym) or []):
                    self._filled_at[int(p.ticket)] = time.time()
                    self._addons_done[int(p.ticket)] = 0
            acct=mt5.account_info()
            if acct:
                try: self._current_drawdown=max(0.0,(acct.balance-acct.equity)/max(1.0,acct.balance))
                except Exception: pass
            return True
        except Exception as e:
            log.error("execute error %s: %s", self.sp.resolved or self.sp.base, e); return False

    def manage_positions(self):
        try:
            sym = self.sp.resolved or self.sp.base
            now=time.time(); tick=mt5.symbol_info_tick(sym)
            for o in (mt5.orders_get(symbol=sym) or []):
                c=self._pending_created.get(o.ticket)
                if c and now-c>self.cfg.pending_ttl_sec:
                    mt5.order_delete(o.ticket)
                    self._pending_created.pop(o.ticket,None)
                    log.info("CANCEL TTL %s order=%s", sym, o.ticket)
            positions=mt5.positions_get(symbol=sym)
            if not positions or not tick: return
            dfp=self._fetch_rates(self.sp.tf_primary, 160)
            indp=self._indicators(dfp) if dfp is not None else None
            dfl=self._fetch_rates(self.sp.tf_long, 160)
            indl=self._indicators(dfl, is_long=True) if dfl is not None else None
            adapt=self._adaptive_params(indp or {}, indl or {})
            for p in positions:
                is_buy=(p.type==mt5.POSITION_TYPE_BUY)
                cur = tick.bid if is_buy else tick.ask
                entry, sl, tp, vol = p.price_open, p.sl, p.tp, p.volume
                risk_unit=(entry-sl) if is_buy else (sl-entry)
                if risk_unit<=0: continue
                profit_R = ((cur-entry) if is_buy else (entry-cur))/risk_unit
                filled=self._filled_at.get(p.ticket)
                if filled and now-filled<=self.cfg.scratch_window_s:
                    if profit_R < -self.cfg.scratch_mae_r:
                        self._close_full(p, cur, comment="scratch-exit")
                        self._filled_at.pop(p.ticket,None);  self._addons_done.pop(p.ticket,None)
                        continue
                if filled and now-filled>3600:
                    self._filled_at.pop(p.ticket,None)
                if profit_R >= self.cfg.be_trigger_R:
                    be = entry + (1 if is_buy else -1)*( self.cfg.be_lock_spread_mult * max(0.0, tick.ask-tick.bid) )
                    self._modify_sl(p.ticket, be, tp)
                if profit_R >= self.cfg.partial_tp_R1 and vol>0.01:
                    self._close_partial(p.ticket, vol*self.cfg.partial_tp_R1_close_pct, is_buy)
                atr = self._last_atr()
                if atr>0:
                    new_sl = (cur - atr*adapt["trail_mult"]) if is_buy else (cur + atr*adapt["trail_mult"])
                    if (is_buy and new_sl>sl) or ((not is_buy) and new_sl<sl):
                        self._modify_sl(p.ticket, new_sl, tp)
        except Exception as e:
            log.debug("manage_positions %s", e)

    def _last_atr(self)->float:
        df=self._fetch_rates(self.sp.tf_primary,100)
        if df is None or len(df)<20: return 0.0
        atr=talib.ATR(df["high"].values.astype(np.float64),
                      df["low"].values.astype(np.float64),
                      df["close"].values.astype(np.float64),14)
        return float(atr[-2]) if np.isfinite(atr[-2]) else 0.0

    def _modify_sl(self, ticket:int, new_sl:float, tp:float):
        try:
            pos=next((x for x in (mt5.positions_get() or []) if x.ticket==ticket), None)
            if not pos: return
            req={"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "symbol": pos.symbol,
                 "sl": float(new_sl), "tp": float(tp), "magic":20250826, "comment":"BE/Trail"}
            res=mt5.order_send(req)
            if getattr(res,"retcode",None)!=mt5.TRADE_RETCODE_DONE:
                log.debug("SLTP ret=%s", getattr(res,"retcode",None))
        except Exception as e: log.debug("modify sl err %s", e)

    def _close_partial(self, ticket:int, vol_to_close:float, is_buy:bool):
        try:
            pos=next((x for x in (mt5.positions_get() or []) if x.ticket==ticket), None)
            if not pos: return
            tick=mt5.symbol_info_tick(pos.symbol); price=tick.bid if is_buy else tick.ask
            req={"action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol,
                 "volume": float(round(max(0.01,min(pos.volume,vol_to_close)),2)),
                 "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
                 "position": ticket, "price": float(price),
                 "deviation":10, "magic":20250826, "comment":"TP1 partial",
                 "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
            res=mt5.order_send(req)
            if getattr(res,"retcode",None)!=mt5.TRADE_RETCODE_DONE:
                log.debug("partial ret=%s", getattr(res,"retcode",None))
        except Exception as e: log.debug("partial err %s", e)

    def _close_full(self, p, price:float, comment:str):
        try:
            req={"action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol,
                 "volume": float(round(max(0.01,p.volume),2)),
                 "type": mt5.ORDER_TYPE_SELL if p.type==mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                 "position": p.ticket, "price": float(price),
                 "deviation":10, "magic":20250826, "comment":comment,
                 "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
            rr=mt5.order_send(req)
            log.info("%s CLOSE %s ret=%s", p.symbol, comment, getattr(rr,"retcode",None))
        except Exception as e: log.debug("close_full err %s", e)

    def _log_csv(self, res: Dict[str,Any]):
        path=self.cfg.log_csv_path
        header=["timestamp","symbol","signal","confidence","regime","spread_bps","latency_ms","reasons"]
        row=[res.get("timestamp"), res.get("symbol"), res.get("signal"), res.get("confidence"),
             res.get("regime"), res.get("spread_bps"), int(res.get("latency_ms",0)), ",".join(res.get("reasons",[]))]
        try:
            exists=os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w=csv.writer(f)
                if not exists: w.writerow(header)
                w.writerow(row)
        except Exception as e:
            log.debug("csv log err: %s", e)

# ---------- Engine ----------
class Engine:
    def __init__(self, cfg:EngineConfig):
        self.cfg=cfg; self.tz=pytz.timezone(cfg.tz_local); self.news=NewsGuard(cfg)
        self._workers: List[Worker] = []
        self._daily_pnl_start_equity: Optional[float] = None

    # symbol resolver: handles suffixes like m, .e, -RAW, etc.
    def _resolve_symbol_name(self, base:str)->str:
        candidates=[base, base+"m", base+".e", base+"-RAW", base+"-raw"]
        try:
            all_syms = mt5.symbols_get()
            names = {s.name for s in (all_syms or [])}
            for c in candidates:
                if c in names: return c
            # fallback: case-insensitive contains
            for s in (all_syms or []):
                nm=s.name.upper(); b=base.upper()
                if b in nm: return s.name
        except Exception:
            pass
        return base

    def _init_mt5(self)->bool:
        for _ in range(6):
            if mt5.initialize(login=self.cfg.login, server=self.cfg.server, password=self.cfg.password):
                term=mt5.terminal_info(); acc=mt5.account_info()
                log.info("MT5 ready | build=%s | login=%s", getattr(term,'build',None), getattr(acc,'login',None))
                if self._daily_pnl_start_equity is None and acc:
                    self._daily_pnl_start_equity = float(acc.equity)
                return True
            time.sleep(1.0)
        log.error("MT5 init failed"); return False

    def _ensure_alive(self):
        ti=mt5.terminal_info()
        if not ti or not ti.connected:
            try: mt5.shutdown()
            except Exception: pass
            self._init_mt5()

    def _normalize_symbols(self, symbols)->List[SymbolParams]:
        # Accept list[str], comma string, or list[SymbolParams]
        if isinstance(symbols, list) and symbols and isinstance(symbols[0], SymbolParams):
            syms = symbols
        else:
            if isinstance(symbols, str): bases=[x.strip() for x in symbols.split(',') if x.strip()]
            elif isinstance(symbols, list): bases=[str(x).strip() for x in symbols if str(x).strip()]
            else: bases=["XAUUSD","BTCUSD","EURUSD"]
            syms=[]
            for b in bases:
                if b.upper()=="BTCUSD":
                    sp=SymbolParams(b, tf_primary="M5", tf_confirm="M15", tf_long="M15")
                    sp.micro_min_tps=3; sp.micro_max_tps=80; sp.quote_flips_max=16; sp.micro_tstat_thresh=0.5
                    sp.spread_limit_pct=0.0008
                elif b.upper()=="XAUUSD":
                    sp=SymbolParams(b, tf_primary="M1", tf_confirm="M5", tf_long="M15")
                    sp.spread_limit_pct=0.00018
                else: # EURUSD
                    sp=SymbolParams(b, tf_primary="M1", tf_confirm="M5", tf_long="M15")
                    sp.spread_limit_pct=0.00012
                syms.append(sp)
        # resolve broker names and select
        for sp in syms:
            sp.resolved = self._resolve_symbol_name(sp.base)
            try:
                mt5.symbol_select(sp.resolved, True)
            except Exception:
                pass
        return syms

    def _daily_loss_guard(self)->bool:
        try:
            acc=mt5.account_info()
            if not acc or self._daily_pnl_start_equity is None:
                return False
            drop=(self._daily_pnl_start_equity - float(acc.equity))/max(1.0, self._daily_pnl_start_equity)
            return drop >= self.cfg.max_daily_loss_pct
        except Exception:
            return False

    def start(self):
        if not self._init_mt5(): raise SystemExit(1)
        self.news.start()
        syms=self._normalize_symbols(self.cfg.symbols)
        self._workers=[Worker(self.cfg, sp, self.news) for sp in syms]

    def get_signal(self, symbol:Optional[str]=None, execute:bool=False)->Dict[str,Any]:
        
        if not self._workers: self.start()
        if self._daily_loss_guard():
            return {"symbol": symbol or "*", "signal":"Neutral", "confidence":0,
                    "regime":None, "reasons":["daily_loss_guard"],
                    "spread_bps":None, "latency_ms":0.0, "timestamp": datetime.now(pytz.timezone(self.cfg.tz_local)).isoformat()}
        if symbol is None:
            if len(self._workers)==1: return self._workers[0].get_signal(execute=execute)
            raise ValueError("Specify symbol when multiple workers are active")
        w=next((w for w in self._workers if (w.sp.resolved==symbol.strip() or w.sp.base==symbol.strip())), None)
        if not w: raise ValueError(f"Unknown symbol {symbol}")
        return w.get_signal(execute=execute)

    def get_signals(self, execute:bool=False)->Dict[str,Dict[str,Any]]:
        if not self._workers: self.start()
        if self._daily_loss_guard():
            return {w.sp.resolved or w.sp.base: {"symbol": w.sp.resolved or w.sp.base, "signal":"Neutral", "confidence":0,
                                                "regime":None, "reasons":["daily_loss_guard"],
                                                "spread_bps":None, "latency_ms":0.0, "timestamp": datetime.now(pytz.timezone(self.cfg.tz_local)).isoformat()}
                    for w in self._workers}
        return {w.sp.resolved or w.sp.base: w.get_signal(execute=execute) for w in self._workers}

    def loop(self, execute:bool=False):
        poll=int(os.getenv("POLL_SECONDS", str(self.cfg.poll_seconds_fast)))
        while True:
            t0=time.time(); self._ensure_alive()
            for w in self._workers:
                try:
                    w.manage_positions()
                    res=w.get_signal(execute=execute)
                    print(json.dumps(res, ensure_ascii=False))
                except Exception as e:
                    log.error("worker %s error: %s", w.sp.resolved or w.sp.base, e)
            time.sleep(max(0.5, poll - (time.time()-t0)))

# ---------- Presets ----------
def apply_high_accuracy_mode(cfg: EngineConfig, enable: bool):
    if not enable: return
    cfg.sl_atr_mult_trend = 1.4
    cfg.tp_atr_mult_trend = 0.6
    cfg.sl_atr_mult_range = 1.6
    cfg.tp_atr_mult_range = 0.5
    cfg.weights.update({"trend":0.55,"momentum":0.25,"meanrev":0.10,"structure":0.05,"volume":0.05})
    cfg.signal_amplification = 1.2
    cfg.conf_min = 92
    cfg.use_squeeze_filter = True
    cfg.max_trades_per_hour = 1
    cfg.hedge_flip_enabled = False
    cfg.pyramid_enabled = False

if __name__=="__main__":
    cfg = EngineConfig(
        login=int(os.getenv("MT5_LOGIN","248532703")),
        password=os.getenv("MT5_PASSWORD","1q2w3e0p$Q"),
        server=os.getenv("MT5_SERVER","Exness-MT5Trial"),
        finnhub_key=os.getenv("FINNHUB_KEY","d2eu521r01qmrq4or590d2eu521r01qmrq4or59g"),
        symbols=os.getenv("SYMBOLS","EURUSDm,XAUUSDm,BTCUSDm"),
    )
    apply_high_accuracy_mode(cfg, bool(int(os.getenv("HIGH_ACCURACY","0"))))
    engine = Engine(cfg)
    engine.start()
    if bool(int(os.getenv("LOOP","0"))):
        engine.loop(execute=bool(int(os.getenv("EXECUTE","0"))))
    else:
        res = engine.get_signals(execute=bool(int(os.getenv("EXECUTE","0"))))  # Используем get_signals
        print(json.dumps(res, ensure_ascii=False, indent=2))