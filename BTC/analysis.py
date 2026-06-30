import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import talib
import MetaTrader5 as mt5

import config


# ───────────────────────────────────────────────────────────────────────────
#  HELPERS
# ───────────────────────────────────────────────────────────────────────────
def num(v, default=0.0):
    """NaN/None-safe scalar. (NaN is truthy in Python, so `x or 0` is a trap.)"""
    if v is None:
        return default
    try:
        if isinstance(v, float) and math.isnan(v):
            return default
    except (TypeError, ValueError):
        return default
    return v


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


_TF_MAP = {
    "D1": mt5.TIMEFRAME_D1, "H4": mt5.TIMEFRAME_H4, "H1": mt5.TIMEFRAME_H1,
    "M15": mt5.TIMEFRAME_M15, "M5": mt5.TIMEFRAME_M5, "M1": mt5.TIMEFRAME_M1,
}
MASTER_TIMEFRAMES = [(_TF_MAP[name], w, name) for name, w in config.TF_WEIGHTS.items()]
ANCHOR_TF_CONSTS = {_TF_MAP[n] for n in config.ANCHOR_TFS}


# ═══════════════════════════════════════════════════════════════════════════
#  INDICATOR ENGINE  —  Optimized for Bitcoin/Crypto Volatility
# ═══════════════════════════════════════════════════════════════════════════
class InstitutionalIndicatorEngine:
    def __init__(self, df):
        self.df = df.copy()

    def compute_all(self):
        df = self.df
        # ── self-contained candle geometry ──
        df['direction'] = np.where(df['close'] > df['open'], 1,
                                   np.where(df['close'] < df['open'], -1, 0))
        df['body_size'] = (df['close'] - df['open']).abs()
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

        high, low, close, opn = df['high'], df['low'], df['close'], df['open']
        volume = df.get('tick_volume')
        if volume is None:
            volume = pd.Series(np.ones(len(df)), index=df.index)
        volume = volume.astype(float).replace(0, np.nan).ffill().fillna(1.0)

        # ── Core indicators ──
        df['atr'] = talib.ATR(high, low, close, timeperiod=14)
        df['rsi'] = talib.RSI(close, timeperiod=14)
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close, 12, 26, 9)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close, 20, 2.0, 2.0, 0)
        df['ema_9'] = talib.EMA(close, 9)
        df['ema_21'] = talib.EMA(close, 21)
        df['ema_50'] = talib.EMA(close, 50)
        df['ema_200'] = talib.EMA(close, 200)
        stoch_k, stoch_d = talib.STOCHRSI(close, 14, 3, 3, 0)
        df['stoch_k'], df['stoch_d'] = stoch_k, stoch_d

        # ── Candle patterns ──
        df['engulfing'] = talib.CDLENGULFING(opn, high, low, close)
        df['hammer'] = talib.CDLHAMMER(opn, high, low, close)
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(opn, high, low, close)
        df['doji'] = talib.CDLDOJI(opn, high, low, close)
        df['pin_bar_bull'] = (df['lower_wick'] > df['body_size'] * 2) & (df['direction'] == 1)
        df['pin_bar_bear'] = (df['upper_wick'] > df['body_size'] * 2) & (df['direction'] == -1)
        df['rej_bull'] = (df['lower_wick'] > df['body_size'] * 1.5) & (df['lower_wick'] > df['upper_wick'])
        df['rej_bear'] = (df['upper_wick'] > df['body_size'] * 1.5) & (df['upper_wick'] > df['lower_wick'])

        # ── Overextension filter (ADAPTED FOR BITCOIN: 5.0% instead of 2.5%, RSI wider) ──
        df['dist_from_ema200'] = (close - df['ema_200']) / df['ema_200'] * 100
        df['overextended_up'] = (df['rsi'] > 85) & (df['dist_from_ema200'] > 5.0)
        df['overextended_down'] = (df['rsi'] < 15) & (df['dist_from_ema200'] < -5.0)

        # ── RSI divergence ──
        prev_price_low = low.shift(1).rolling(10).min()
        prev_rsi_low = df['rsi'].shift(1).rolling(10).min()
        prev_price_high = high.shift(1).rolling(10).max()
        prev_rsi_high = df['rsi'].shift(1).rolling(10).max()
        df['bullish_divergence'] = (low < prev_price_low) & (df['rsi'] > prev_rsi_low)
        df['bearish_divergence'] = (high > prev_price_high) & (df['rsi'] < prev_rsi_high)

        # ── Order flow: CVD + divergence ──
        df['typical_price'] = (high + low + close) / 3
        df['vol_delta'] = np.where(close > opn, volume, np.where(close < opn, -volume, 0.0))
        df['cvd'] = df['vol_delta'].cumsum()
        df['cvd_ema'] = df['cvd'].ewm(span=14, adjust=False).mean()
        df['cvd_rising'] = df['cvd'] > df['cvd_ema']
        df['cvd_falling'] = df['cvd'] < df['cvd_ema']
        df['vol_sma'] = volume.rolling(20).mean()
        df['high_volume'] = volume > df['vol_sma'] * 2.0  # Crypto spikes are sharper
        prev_cvd_high = df['cvd'].shift(1).rolling(14).max()
        prev_cvd_low = df['cvd'].shift(1).rolling(14).min()
        df['cvd_bear_div'] = (high > prev_price_high) & (df['cvd'] < prev_cvd_high)
        df['cvd_bull_div'] = (low < prev_price_low) & (df['cvd'] > prev_cvd_low)

        # ── Rolling VWAP + σ-bands ──
        w = 20
        vp = (df['typical_price'] * volume).rolling(w).sum()
        vv = volume.rolling(w).sum().replace(0, np.nan)
        df['vwap'] = vp / vv
        dev = df['typical_price'] - df['vwap']
        vstd = dev.rolling(w).std()
        df['vwap_upper'] = df['vwap'] + 2 * vstd
        df['vwap_lower'] = df['vwap'] - 2 * vstd
        df['vwap_slope'] = df['vwap'].diff()

        # ── Volume Profile POC ──
        df['poc_level'] = vp / vv
        df['va_high'] = df['poc_level'] + vstd
        df['va_low'] = df['poc_level'] - vstd

        # ── Regime: Kaufman Efficiency Ratio + ADX ──
        n = 10
        chg = (close - close.shift(n)).abs()
        vol_path = close.diff().abs().rolling(n).sum().replace(0, np.nan)
        df['efficiency'] = (chg / vol_path).fillna(0)
        adx_f = df['adx'].fillna(0)
        df['regime'] = np.where((df['efficiency'] > 0.45) | (adx_f > 25), 'trend',
                                np.where((df['efficiency'] < 0.30) & (adx_f < 20), 'range', 'mixed'))

        # ── Volatility regime + squeeze ──
        df['atr_rank'] = df['atr'].rolling(100, min_periods=20).rank(pct=True)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, np.nan)
        df['bb_width_rank'] = df['bb_width'].rolling(100, min_periods=20).rank(pct=True)
        df['squeeze'] = df['bb_width_rank'] < 0.20

        # ── Z-score mean reversion ──
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std().replace(0, np.nan)
        df['zscore'] = (close - sma20) / std20

        # ── SMC: swing structure ──
        window = 5
        df['is_swing_high'] = (high == high.rolling(window).max())
        df['is_swing_low'] = (low == low.rolling(window).min())
        df['confirmed_swing_high'] = df['is_swing_high'].shift(2)
        df['confirmed_swing_low'] = df['is_swing_low'].shift(2)
        df['last_swing_high'] = high.shift(2).where(df['confirmed_swing_high']).ffill()
        df['last_swing_low'] = low.shift(2).where(df['confirmed_swing_low']).ffill()
        prev_sh = df['last_swing_high'].shift(1)
        prev_sl = df['last_swing_low'].shift(1)

        df['trend'] = np.where(prev_sh.notna() & (close > prev_sh), 1,
                               np.where(prev_sl.notna() & (close < prev_sl), -1, np.nan))
        df['trend'] = df['trend'].ffill().fillna(0)
        prev_trend = df['trend'].shift(1)

        df['bullish_bos'] = prev_sh.notna() & (close > prev_sh) & (prev_trend == 1)
        df['bearish_bos'] = prev_sl.notna() & (close < prev_sl) & (prev_trend == -1)
        tol = df['atr'] * 0.1
        df['eqh'] = prev_sh.notna() & ((high - prev_sh).abs() < tol) & ~df['is_swing_high']
        df['eql'] = prev_sl.notna() & ((low - prev_sl).abs() < tol) & ~df['is_swing_low']
        df['liquidity_sweep_high'] = prev_sh.notna() & (high > prev_sh) & (close < prev_sh)
        df['liquidity_sweep_low'] = prev_sl.notna() & (low < prev_sl) & (close > prev_sl)
        df['bullish_fvg'] = low > high.shift(2)
        df['bearish_fvg'] = high < low.shift(2)

        # ── SMC: Order Blocks (institutional demand / supply zones) ──
        # Bullish OB = the last DOWN candle right before a strong bullish
        # displacement (a big body vs ATR that also leaves a bullish FVG or breaks
        # structure). Price returning INTO that zone and reacting up is high-prob
        # demand; mirror logic for the bearish OB (supply). A zone stays valid for
        # OB_MAX_AGE bars. Purely additive — only writes new columns.
        _ob_i = np.arange(len(df))
        _big_body = df['body_size'] > (df['atr'] * config.OB_DISPLACE_ATR)
        _disp_bull = (df['direction'] == 1) & _big_body & (df['bullish_fvg'] | df['bullish_bos'])
        _disp_bear = (df['direction'] == -1) & _big_body & (df['bearish_fvg'] | df['bearish_bos'])
        _ob_bull = _disp_bull & (df['direction'].shift(1) == -1)   # prior candle = down
        _ob_bear = _disp_bear & (df['direction'].shift(1) == 1)    # prior candle = up
        df['bull_ob_high'] = high.shift(1).where(_ob_bull).ffill()
        df['bull_ob_low'] = low.shift(1).where(_ob_bull).ffill()
        df['bear_ob_high'] = high.shift(1).where(_ob_bear).ffill()
        df['bear_ob_low'] = low.shift(1).where(_ob_bear).ffill()
        _bull_age = _ob_i - pd.Series(np.where(_ob_bull, _ob_i, np.nan), index=df.index).ffill()
        _bear_age = _ob_i - pd.Series(np.where(_ob_bear, _ob_i, np.nan), index=df.index).ffill()
        df['in_bull_ob'] = (((_bull_age <= config.OB_MAX_AGE) & (low <= df['bull_ob_high'])
                             & (close >= df['bull_ob_low']) & (close > opn)).fillna(False))
        df['in_bear_ob'] = (((_bear_age <= config.OB_MAX_AGE) & (high >= df['bear_ob_low'])
                             & (close <= df['bear_ob_high']) & (close < opn)).fillna(False))

        df['ema_bull_stack'] = (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
        df['ema_bear_stack'] = (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])

        self.df = df
        return self.df


# ═══════════════════════════════════════════════════════════════════════════
#  CRYPTO 24/7 SESSION TIMING (No Weekend Blocks!)
# ═══════════════════════════════════════════════════════════════════════════
try:
    from zoneinfo import ZoneInfo
    _TZ_NY = ZoneInfo("America/New_York")
    _TZ_TOKYO = ZoneInfo("Asia/Tokyo")
    _ZONES_OK = True
except Exception:
    _ZONES_OK = False


def _in_window(dt, sh, sm, eh, em):
    """Crypto is 24/7. Removed the weekend block that was killing Saturday/Sunday."""
    minutes = dt.hour * 60 + dt.minute
    return sh * 60 + sm <= minutes < eh * 60 + em


def get_session_info(now=None):
    """
    Crypto-optimized session detection. Bitcoin reacts to US market open (ETFs)
    and Asian session, but NEVER closes on weekends.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    hour, minute = now.hour, now.minute

    is_weekend = now.weekday() >= 5

    if _ZONES_OK:
        ny = now.astimezone(_TZ_NY)
        tok = now.astimezone(_TZ_TOKYO)
        newyork = _in_window(ny, 8, 0, 17, 0)
        asian = _in_window(tok, 9, 0, 15, 0)
        
        if is_weekend:
            session, power = "CRYPTO_WEEKEND", 0.95
        elif newyork:
            session, power = "US_SESSION", 1.15   # Highest volatility for BTC ETFs
        elif asian:
            session, power = "ASIAN_SESSION", 1.05
        else:
            session, power = "CRYPTO_BASE", 1.00
    else:
        if is_weekend:
            session, power = "CRYPTO_WEEKEND", 0.95
        elif 13 <= hour < 21:
            session, power = "US_SESSION", 1.15
        elif 0 <= hour < 8:
            session, power = "ASIAN_SESSION", 1.05
        else:
            session, power = "CRYPTO_BASE", 1.00

    h4_fresh = ((hour % 4) * 60 + minute) <= 10
    h1_fresh = minute <= 5
    m15_since = minute % 15
    m15_fresh = m15_since <= 3
    m15_ending = m15_since >= 13

    bonus = (5 if h4_fresh else 0) + (3 if h1_fresh else 0) + (2 if m15_fresh else 0)
    return {
        "session": session, "session_power": power,
        "h4_fresh": h4_fresh, "h1_fresh": h1_fresh, "m15_fresh": m15_fresh,
        "m15_ending": m15_ending, "candle_bonus": bonus,
        "hour_utc": hour, "minute_utc": minute,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PER-TIMEFRAME SCORING
# ═══════════════════════════════════════════════════════════════════════════
def _score_timeframe(c, p):
    # 1. SMC structure
    smc_b = smc_s = 0
    if c.get('bullish_bos'):
        smc_b += 10
    if c.get('liquidity_sweep_low') or c.get('eql'):
        smc_b += 8
    if c.get('bullish_fvg'):
        smc_b += 4
    if c.get('bearish_bos'):
        smc_s += 10
    if c.get('liquidity_sweep_high') or c.get('eqh'):
        smc_s += 8
    if c.get('bearish_fvg'):
        smc_s += 4
    smc_b, smc_s = min(smc_b, 20), min(smc_s, 20)
    # Order-block confluence: a clean bonus ON TOP of the base SMC score, so
    # existing setups score exactly as before and only a fresh institutional-zone
    # reaction in the signal's direction is rewarded (sharper, not louder).
    if config.OB_ENABLED:
        if c.get('in_bull_ob'):
            smc_b = min(smc_b + config.OB_SCORE, 20 + config.OB_SCORE)
        if c.get('in_bear_ob'):
            smc_s = min(smc_s + config.OB_SCORE, 20 + config.OB_SCORE)

    # 2. Momentum
    mom_b = mom_s = 0
    adx = num(c.get('adx'))
    if adx > 20:
        if num(c.get('plus_di')) > num(c.get('minus_di')):
            mom_b += 6
        if num(c.get('minus_di')) > num(c.get('plus_di')):
            mom_s += 6
    mh, pmh = num(c.get('macd_hist')), num(p.get('macd_hist'))
    if mh > 0 and mh > pmh:
        mom_b += 6
    if mh < 0 and mh < pmh:
        mom_s += 6
    if num(c.get('vwap_slope')) > 0:
        mom_b += 3
    elif num(c.get('vwap_slope')) < 0:
        mom_s += 3
    mom_b, mom_s = min(mom_b, 15), min(mom_s, 15)

    # 3. Mean reversion (Adjusted for Crypto Z-Score tolerance)
    rev_b = rev_s = 0
    rsi = num(c.get('rsi'), 50)
    if 20 < rsi < 40:
        rev_b += 5
    if 60 < rsi < 80:
        rev_s += 5
    low_v, bb_low = num(c.get('low')), num(c.get('bb_lower'))
    high_v, bb_up = num(c.get('high')), num(c.get('bb_upper'))
    if bb_low and low_v <= bb_low:
        rev_b += 4
    if bb_up and high_v >= bb_up:
        rev_s += 4
    sk = num(c.get('stoch_k'), 50)
    if sk < 20:
        rev_b += 3
    elif sk > 80:
        rev_s += 3
    
    # Z-score adjusted from 2.0 to 2.5 for Bitcoin volatility
    z = num(c.get('zscore'))
    if z < -2.5:
        rev_b += 3
    elif z > 2.5:
        rev_s += 3
    
    vwl, vwu = num(c.get('vwap_lower')), num(c.get('vwap_upper'))
    if vwl and low_v <= vwl:
        rev_b += 2
    if vwu and high_v >= vwu:
        rev_s += 2
    rev_b, rev_s = min(rev_b, 15), min(rev_s, 15)

    # 4. Confirmation
    conf_b = conf_s = 0
    if (num(c.get('engulfing')) > 0 or num(c.get('hammer')) > 0 or c.get('pin_bar_bull') or c.get('rej_bull')):
        conf_b += 5
    if c.get('bullish_divergence') or c.get('cvd_bull_div'):
        conf_b += 5
    if c.get('ema_bull_stack'):
        conf_b += 5
    if (num(c.get('engulfing')) < 0 or num(c.get('shooting_star')) > 0 or c.get('pin_bar_bear') or c.get('rej_bear')):
        conf_s += 5
    if c.get('bearish_divergence') or c.get('cvd_bear_div'):
        conf_s += 5
    if c.get('ema_bear_stack'):
        conf_s += 5
    conf_b, conf_s = min(conf_b, 15), min(conf_s, 15)

    # 5. Volume / order flow
    vol_b = vol_s = 0
    hv = c.get('high_volume')
    if c.get('cvd_rising'):
        vol_b += 10 if hv else 6
    if c.get('cvd_falling'):
        vol_s += 10 if hv else 6
    poc, price, atr = num(c.get('poc_level')), num(c.get('close')), num(c.get('atr'), 1)
    if poc > 0 and abs(price - poc) < atr * 0.5:
        if c.get('cvd_rising'):
            vol_b += 5
        elif c.get('cvd_falling'):
            vol_s += 5
    vol_b, vol_s = min(vol_b, 15), min(vol_s, 15)

    # ── REGIME WEIGHTS: Crypto trends harder, trust momentum more ──
    regime = c.get('regime', 'mixed')
    trend_w = 1.30 if regime == 'trend' else (0.70 if regime == 'range' else 1.0)
    rev_w = 1.25 if regime == 'range' else (0.50 if regime == 'trend' else 1.0)

    smc_b, smc_s = smc_b * trend_w, smc_s * trend_w
    mom_b, mom_s = mom_b * trend_w, mom_s * trend_w
    rev_b, rev_s = rev_b * rev_w, rev_s * rev_w

    buy = smc_b + mom_b + rev_b + conf_b + vol_b
    sell = smc_s + mom_s + rev_s + conf_s + vol_s

    if c.get('overextended_up'):
        buy = max(0, buy - 15)
    if c.get('overextended_down'):
        sell = max(0, sell - 15)

    breakdown = {
        "smc": round(smc_b - smc_s, 1), "momentum": round(mom_b - mom_s, 1),
        "reversion": round(rev_b - rev_s, 1), "confirm": conf_b - conf_s,
        "volume": vol_b - vol_s,
    }
    return round(buy, 1), round(sell, 1), breakdown


# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-TIMEFRAME SIGNAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════
class MultiTimeframeSignalEngine:
    def __init__(self, data_engine):
        self.data_engine = data_engine
        self._prev_signal_key = None
        self._last_dir_ts = {"BUY": -1e18, "SELL": -1e18}
        self._pending = None

    @staticmethod
    def _live_spread():
        try:
            tick = mt5.symbol_info_tick(config.SYMBOL)
            if tick:
                return max(0.0, float(tick.ask - tick.bid))
        except Exception:
            pass
        return None

    # ── scalp-sized SL/TP from real structure (CRYPTO SPREAD OPTIMIZED) ──
    def _compute_smart_sl_tp(self, signal_dir, exec_df, m15_df, price, atr):
        if not (np.isfinite(price) and np.isfinite(atr) and atr > 0):
            return None

        # Bitcoin spread adaptation: Cap the spread buffer so massive spikes
        # don't ruin the SL calculation, but keep it wide enough for safety.
        try:
            tick = mt5.symbol_info_tick(config.SYMBOL)
            spread = (tick.ask - tick.bid) if tick else 0.0
        except Exception:
            spread = 0.0
        
        spread_buffer = min(max(spread * 0.5, atr * 0.05), atr * 0.4)

        sl_c, tp_c = [], []
        for df in [exec_df, m15_df]:
            if df is None or len(df) < 20:
                continue
            recent = df.iloc[-50:].reset_index(drop=True)
            nn = len(recent)
            highs, lows = recent['high'].values, recent['low'].values
            opens, closes = recent['open'].values, recent['close'].values

            # swing pivots — placed like a disciplined trader: SL a touch BEYOND the
            # level (a stop-hunt wick won't take it), TP a touch BEFORE the level (it
            # fills before the crowd reverses price right at the obvious high/low).
            # BTC wicks are violent, so the buffers are a bit wider than gold's.
            sl_buf, tp_buf = atr * 0.15, atr * 0.10
            for i in range(3, nn - 3):
                if highs[i] >= max(highs[i-1], highs[i-2], highs[i-3], highs[i+1], highs[i+2]):
                    if signal_dir == 'BUY':
                        tp_c.append((highs[i] - tp_buf, 3))     # TP just before resistance
                    else:
                        sl_c.append((highs[i] + sl_buf, 3))     # SL just beyond the swing high
                if lows[i] <= min(lows[i-1], lows[i-2], lows[i-3], lows[i+1], lows[i+2]):
                    if signal_dir == 'BUY':
                        sl_c.append((lows[i] - sl_buf, 3))      # SL just beyond the swing low
                    else:
                        tp_c.append((lows[i] + tp_buf, 3))      # TP just before support
            for i in range(2, nn - 1):
                body = abs(closes[i] - opens[i])
                if closes[i] > opens[i] and body > atr * 0.5 and closes[i-1] < opens[i-1]:
                    if signal_dir == 'BUY' and lows[i-1] < price:
                        sl_c.append((lows[i-1] - atr * 0.1, 5))
                if closes[i] < opens[i] and body > atr * 0.5 and closes[i-1] > opens[i-1]:
                    if signal_dir == 'SELL' and highs[i-1] > price:
                        sl_c.append((highs[i-1] + atr * 0.1, 5))
            for i in range(2, nn):
                if lows[i] > highs[i-2] and signal_dir == 'BUY' and highs[i-2] < price:
                    sl_c.append((highs[i-2] - atr * 0.1, 3))
                if highs[i] < lows[i-2] and signal_dir == 'SELL' and lows[i-2] > price:
                    sl_c.append((lows[i-2] + atr * 0.1, 3))

        def best_sl():
            fb = price - atr if signal_dir == 'BUY' else price + atr
            if signal_dir == 'BUY':
                v = [(lvl, w) for lvl, w in sl_c if lvl < price - atr * 0.1]
                if not v:
                    return fb
                v.sort(key=lambda x: (-x[1], -x[0]))
            else:
                v = [(lvl, w) for lvl, w in sl_c if lvl > price + atr * 0.1]
                if not v:
                    return fb
                v.sort(key=lambda x: (-x[1], x[0]))
            return v[0][0]

        def best_tp(risk):
            band = config.TP_ATR_MULT_MAX * atr
            if signal_dir == 'BUY':
                v = [lvl for lvl, w in tp_c if price + risk * config.MIN_RR <= lvl <= price + band]
                if v:
                    return min(v)
                return price + max(risk * config.TARGET_RR, atr)
            else:
                v = [lvl for lvl, w in tp_c if price - band <= lvl <= price - risk * config.MIN_RR]
                if v:
                    return max(v)
                return price - max(risk * config.TARGET_RR, atr)

        sl = best_sl()
        if signal_dir == 'BUY':
            sl -= spread_buffer
        else:
            sl += spread_buffer
        
        risk = abs(price - sl)
        max_sl = config.SL_ATR_MULT_MAX * atr
        min_sl = config.SL_ATR_MULT_MIN * atr
        
        if risk > max_sl or risk <= 0:
            sl = price - max_sl if signal_dir == 'BUY' else price + max_sl
            risk = max_sl
        elif risk < min_sl:
            sl = price - min_sl if signal_dir == 'BUY' else price + min_sl
            risk = min_sl
            
        tp = best_tp(risk)
        reward = abs(tp - price)
        if reward > config.TP_ATR_MULT_MAX * atr:
            tp = price + config.TP_ATR_MULT_MAX * atr if signal_dir == 'BUY' else price - config.TP_ATR_MULT_MAX * atr
            reward = config.TP_ATR_MULT_MAX * atr
        if reward < risk * config.MIN_RR:
            tp = price + risk * config.TARGET_RR if signal_dir == 'BUY' else price - risk * config.TARGET_RR
            
        return round(sl, 2), round(tp, 2)

    def _dedupe_key(self, signal, price, atr):
        bucket = round(price / max(atr * 0.25, 0.01))
        return f"{signal}_{bucket}"

    def _aggregate(self, per_tf):
        total_buy = total_sell = 0.0
        tf_buy = tf_sell = 0
        tf_results = []
        tf_scores = {}
        agg = {"smc": 0.0, "momentum": 0.0, "reversion": 0.0, "confirm": 0.0, "volume": 0.0}
        anchor_dir, anchor_score, anchor_name = None, 0, ""

        for it in per_tf:
            tf, weight, name, c, p = it['tf'], it['weight'], it['name'], it['c'], it['p']
            buy_s, sell_s, brk = _score_timeframe(c, p)
            tf_scores[name] = (buy_s, sell_s)

            if buy_s > sell_s and buy_s >= config.TF_AGREE_MIN_SCORE:
                tf_buy += 1
            elif sell_s > buy_s and sell_s >= config.TF_AGREE_MIN_SCORE:
                tf_sell += 1

            if tf in ANCHOR_TF_CONSTS:
                diff = buy_s - sell_s
                qualifies = ((diff > 0 and buy_s >= config.ANCHOR_MIN_SCORE)
                             or (diff < 0 and sell_s >= config.ANCHOR_MIN_SCORE))
                if qualifies and abs(diff) > anchor_score:
                    anchor_score = abs(diff)
                    anchor_name = name
                    anchor_dir = "BUY" if diff > 0 else "SELL"

            total_buy += buy_s * weight / 100
            total_sell += sell_s * weight / 100
            for k in agg:
                agg[k] += brk[k] * weight / 100

            d = "BUY" if buy_s > sell_s else ("SELL" if sell_s > buy_s else "—")
            mark = " 🔑" if tf in ANCHOR_TF_CONSTS else ""
            tf_results.append(f"{name}: {d} ({buy_s:.0f}B/{sell_s:.0f}S){mark}")

        return {"total_buy": total_buy, "total_sell": total_sell, "tf_buy": tf_buy,
                "tf_sell": tf_sell, "agg": agg, "anchor_dir": anchor_dir,
                "anchor_name": anchor_name, "tf_results": tf_results, "tf_scores": tf_scores}

    def _decide(self, aggr, shared):
        last_close, last_atr = shared['last_close'], shared['last_atr']
        sl_tp_atr, exec_df, m15_df = shared['sl_tp_atr'], shared['exec_df'], shared['m15_df']
        timing = shared['timing']
        spread = shared.get('spread')
        spread_atr = shared.get('spread_atr')

        total_buy, total_sell = aggr['total_buy'], aggr['total_sell']
        tf_buy, tf_sell, agg = aggr['tf_buy'], aggr['tf_sell'], aggr['agg']
        anchor_dir, anchor_name = aggr['anchor_dir'], aggr['anchor_name']

        lead = "BUY" if total_buy >= total_sell else "SELL"
        final_buy, final_sell = total_buy, total_sell
        if timing['candle_bonus'] > 0:
            if lead == "BUY":
                final_buy += timing['candle_bonus']
            else:
                final_sell += timing['candle_bonus']

        sp = timing['session_power']
        conf_buy = min(100, int(final_buy / config.CONF_SCALE * 100 * sp))
        conf_sell = min(100, int(final_sell / config.CONF_SCALE * 100 * sp))

        def stars(cf):
            if cf >= 90:
                return "⭐⭐⭐⭐⭐"
            if cf >= 80:
                return "⭐⭐⭐⭐"
            if cf >= 70:
                return "⭐⭐⭐"
            return "⭐⭐"

        nz = max(abs(v) for v in agg.values()) if agg else 0
        dominant = max(agg, key=lambda k: abs(agg[k])) if nz > 0 else "none"

        sess_map = {"US_SESSION": "🇺🇸 US Crypto", "ASIAN_SESSION": "🌙 Asian",
                    "CRYPTO_WEEKEND": "🔥 Weekend 24/7", "CRYPTO_BASE": "⚡ Crypto Base"}
        fresh = " ".join(t for t, on in [("H4🆕", timing['h4_fresh']),
                                         ("H1🆕", timing['h1_fresh']),
                                         ("M15🆕", timing['m15_fresh'])] if on) or "—"

        base = {
            "close_price": last_close, "atr": last_atr, "tf_details": aggr['tf_results'],
            "breakdown": agg, "dominant": dominant, "buy_conf": conf_buy, "sell_conf": conf_sell,
            "tf_agree_buy": tf_buy, "tf_agree_sell": tf_sell, "anchor": anchor_name or "—",
            "anchor_dir": anchor_dir or "NONE",
            "session": sess_map.get(timing['session'], timing['session']),
            "session_power": sp, "candle_fresh": fresh,
            "spread": spread, "spread_atr": spread_atr,
        }

        signal_dir, score = None, 0
        edge = abs(conf_buy - conf_sell)
        if (conf_buy > conf_sell and conf_buy >= config.MIN_CONFIDENCE
                and tf_buy >= config.MIN_TF_AGREE and anchor_dir == "BUY"
                and edge >= config.MIN_CONF_EDGE):
            signal_dir, score = "BUY", conf_buy
        elif (conf_sell > conf_buy and conf_sell >= config.MIN_CONFIDENCE
                and tf_sell >= config.MIN_TF_AGREE and anchor_dir == "SELL"
                and edge >= config.MIN_CONF_EDGE):
            signal_dir, score = "SELL", conf_sell

        # ── Higher-timeframe context guard (D1 macro + H1 intraday) ──────────────
        # Execution TFs lead the 10-40 min scalp, but taking a trade straight into a
        # strong higher-TF trend is the main losing pattern. A higher TF "opposes" only
        # when its OWN raw score on the other side is ≥ TF_AGREE_MIN_SCORE (i.e. it would
        # independently vote the opposite direction). One opposing TF → confidence
        # penalty (re-gated against MIN_CONFIDENCE); both D1+H1 opposing → veto.
        if signal_dir is not None and config.HTF_GUARD_ENABLED:
            tf_scores = aggr.get('tf_scores', {})

            def _htf_opposes(tf_name):
                bs, ss = tf_scores.get(tf_name, (0.0, 0.0))
                opp = ss if signal_dir == "BUY" else bs   # score on the side AGAINST us
                own = bs if signal_dir == "BUY" else ss
                return opp >= config.TF_AGREE_MIN_SCORE and opp > own

            against = [t for t in config.HTF_GUARD_TFS if _htf_opposes(t)]
            if len(against) >= 2:
                return {**base, "signal": "NEUTRAL",
                        "reason": f"HTF veto: {'+'.join(against)} oppose {signal_dir}"}
            if len(against) == 1:
                score = int(score * config.HTF_PENALTY)
                if signal_dir == "BUY":
                    base["buy_conf"] = score
                else:
                    base["sell_conf"] = score
                if score < config.MIN_CONFIDENCE:
                    return {**base, "signal": "NEUTRAL",
                            "reason": f"HTF conflict: {against[0]} opposes {signal_dir} — conf {score}% < {config.MIN_CONFIDENCE}%"}

        if signal_dir is None:
            reasons = []
            if anchor_dir is None:
                reasons.append("Anchor weak")
            elif ((conf_buy >= config.MIN_CONFIDENCE and anchor_dir != "BUY")
                  or (conf_sell >= config.MIN_CONFIDENCE and anchor_dir != "SELL")):
                reasons.append(f"Anchor({anchor_name})={anchor_dir} mismatch")
            if tf_buy < config.MIN_TF_AGREE and tf_sell < config.MIN_TF_AGREE:
                reasons.append(f"TF agree B={tf_buy} S={tf_sell} (need {config.MIN_TF_AGREE})")
            if conf_buy < config.MIN_CONFIDENCE and conf_sell < config.MIN_CONFIDENCE:
                reasons.append(f"Conf low B:{conf_buy}% S:{conf_sell}%")
            if edge < config.MIN_CONF_EDGE and max(conf_buy, conf_sell) >= config.MIN_CONFIDENCE:
                reasons.append(f"Edge thin ({edge}%, need {config.MIN_CONF_EDGE}%)")
            return {**base, "signal": "NEUTRAL", "reason": " | ".join(reasons) or "No confluence"}

        ext = config.OVEREXT_Z
        exec_z = shared.get('exec_z', 0.0)
        if signal_dir == "BUY" and exec_z >= ext:
            return {**base, "signal": "NEUTRAL", "reason": f"Overextended +{exec_z:.1f}σ — not chasing the top"}
        if signal_dir == "SELL" and exec_z <= -ext:
            return {**base, "signal": "NEUTRAL", "reason": f"Overextended {exec_z:.1f}σ — not chasing the bottom"}

        exec_body_atr = shared.get('exec_body_atr', 0.0)
        if (score < config.STRONG_SIGNAL_IGNORE_BODY_SCORE
                and not shared.get('exec_high_volume')
                and exec_body_atr < config.MIN_TRIGGER_BODY_ATR):
            return {**base, "signal": "NEUTRAL",
                    "reason": f"Trigger candle too weak ({exec_body_atr:.2f} ATR)"}

        # Anti-stop-hunt: stand aside on a VIOLENT manipulation/news spike candle
        # (range >> ATR). Those wick one way, take stops, then reverse — entering
        # there is exactly how the M5 stop gets hunted. Disable with MAX_TRIGGER_RANGE_ATR=99.
        exec_range_atr = shared.get('exec_range_atr', 0.0)
        if exec_range_atr > config.MAX_TRIGGER_RANGE_ATR:
            return {**base, "signal": "NEUTRAL",
                    "reason": f"Volatility spike ({exec_range_atr:.1f}×ATR) — not chasing manipulation"}

        sltp = self._compute_smart_sl_tp(signal_dir, exec_df, m15_df, last_close, sl_tp_atr)
        if sltp is None:
            return {**base, "signal": "NEUTRAL", "reason": "SL/TP could not be computed"}
        sl, tp = sltp

        return {
            **base, "signal": signal_dir, "score": score, "stars": stars(score),
            "sl": sl, "tp": tp,
            "risk_pts": round(abs(last_close - sl), 2),
            "reward_pts": round(abs(tp - last_close), 2),
        }

    @staticmethod
    def _ab_view(d):
        return {
            "signal": d.get("signal", "NEUTRAL"), "score": d.get("score", 0),
            "close_price": d.get("close_price"), "sl": d.get("sl"), "tp": d.get("tp"),
            "atr": d.get("atr"), "risk_pts": d.get("risk_pts"),
            "reward_pts": d.get("reward_pts"), "session": d.get("session"),
            "buy_conf": d.get("buy_conf", 0), "sell_conf": d.get("sell_conf", 0),
            "spread": d.get("spread"), "spread_atr": d.get("spread_atr"),
        }

    def generate_signal(self, now=None):
        if hasattr(self.data_engine, 'last_bar_age_seconds'):
            try:
                age = self.data_engine.last_bar_age_seconds(mt5.TIMEFRAME_M1)
                if age is not None and age > config.MAX_DATA_AGE_SEC:
                    return {"signal": "NEUTRAL", "reason": f"Stale feed ({int(age)}s, no new M1 bar)"}
            except Exception:
                pass

        per_tf = []
        last_close = last_atr = exec_atr = None
        exec_df = m15_df = None
        exec_z = 0.0
        exec_body_atr = 0.0
        exec_range_atr = 0.0
        exec_high_volume = False
        for tf, weight, name in MASTER_TIMEFRAMES:
            try:
                raw = self.data_engine.get_data(specific_tf=tf)
                df = self.data_engine.to_dataframe(raw)
                if df is None or len(df) < 60:
                    continue
                analyzed = InstitutionalIndicatorEngine(df).compute_all()
                c = analyzed.iloc[-2].to_dict()
                p = analyzed.iloc[-3].to_dict()
                if tf == mt5.TIMEFRAME_M5:
                    exec_df, exec_atr = analyzed, num(c.get('atr'))
                    exec_z = num(c.get('zscore'))
                    if exec_atr and exec_atr > 0:
                        exec_body_atr = num(c.get('body_size')) / exec_atr
                        exec_range_atr = (num(c.get('high')) - num(c.get('low'))) / exec_atr
                    exec_high_volume = bool(c.get('high_volume'))
                if tf == mt5.TIMEFRAME_M15:
                    m15_df, last_close, last_atr = analyzed, c.get('close'), num(c.get('atr'))
                per_tf.append({"tf": tf, "weight": weight, "name": name, "c": c, "p": p})
            except Exception:
                continue

        if last_close is None or last_atr is None:
            return {"signal": "NEUTRAL", "reason": "No data available"}
        if not (np.isfinite(last_close) and np.isfinite(last_atr) and last_atr > 0):
            return {"signal": "NEUTRAL", "reason": "Invalid ATR/price"}

        sl_tp_atr = exec_atr if (exec_atr and np.isfinite(exec_atr) and exec_atr > 0) else last_atr
        spread = self._live_spread()
        spread_atr = (spread / sl_tp_atr) if (spread is not None and sl_tp_atr > 0) else None
        if spread is not None:
            if spread > config.MAX_SPREAD_USD:
                return {"signal": "NEUTRAL",
                        "reason": f"Spread too wide ({spread:.2f} > {config.MAX_SPREAD_USD})"}
            if spread_atr is not None and spread_atr > config.MAX_SPREAD_ATR_PCT:
                return {"signal": "NEUTRAL",
                        "reason": f"Spread/ATR too high ({spread_atr:.2%})"}

        # Session detection anchored to the BROKER clock (robust on any server,
        # even with a skewed OS clock). An explicit `now` (tests) still overrides.
        session_now = now
        if session_now is None and hasattr(self.data_engine, 'broker_utc_now'):
            try:
                session_now = self.data_engine.broker_utc_now()
            except Exception:
                session_now = None
        timing = get_session_info(now=session_now)
        if timing['m15_ending']:
            return {"signal": "NEUTRAL",
                    "reason": f"M15 candle ending ({timing['minute_utc']}min) — waiting",
                    "session": timing['session']}

        shared = {"last_close": last_close, "last_atr": last_atr, "sl_tp_atr": sl_tp_atr,
                  "exec_df": exec_df, "m15_df": m15_df, "timing": timing,
                  "exec_z": exec_z, "exec_body_atr": exec_body_atr,
                  "exec_range_atr": exec_range_atr,
                  "exec_high_volume": exec_high_volume,
                  "spread": spread, "spread_atr": spread_atr}

        decision = self._decide(self._aggregate(per_tf), shared)
        s = dict(decision)

        self._pending = None
        if s.get("signal") in ("BUY", "SELL"):
            now_dt = now if now is not None else datetime.now(timezone.utc)
            now_ts = now_dt.timestamp()
            d = s["signal"]
            key = self._dedupe_key(d, last_close, sl_tp_atr)
            elapsed = now_ts - self._last_dir_ts.get(d, -1e18)
            cooled = 0 <= elapsed < config.SAME_DIR_COOLDOWN_MIN * 60
            suppress = None
            if key == self._prev_signal_key:
                suppress = "Duplicate (same price zone)"
            elif cooled:
                mins = config.SAME_DIR_COOLDOWN_MIN
                suppress = f"{d} cooldown ({mins}min) — one signal per setup"
            if suppress:
                s["signal"] = "NEUTRAL"
                s["reason"] = suppress
                for k in ("score", "stars", "sl", "tp", "risk_pts", "reward_pts"):
                    s.pop(k, None)
            else:
                self._pending = (key, d, now_ts)

        s["engine_version"] = "live"
        s["ab"] = {"live": self._ab_view(decision)}
        return s

    def commit_pending(self):
        if self._pending:
            key, d, ts = self._pending
            self._prev_signal_key = key
            self._last_dir_ts[d] = ts
            self._pending = None
