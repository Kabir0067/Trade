from info import initialize_mt5, TIME_ZONES
from typing import Tuple, Optional
from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import talib
import time



def get_rates(symbol, timeframe, count, max_retries=3, retry_delay=1.0):
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Symbol {symbol} not found or not enabled")
        return None
    for attempt in range(max_retries):
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                print(f"❌ Attempt {attempt + 1}/{max_retries}: No data received for {symbol}")
            elif len(rates) < count:
                print(f"❌ Attempt {attempt + 1}/{max_retries}: Insufficient data for {symbol}: "
                      f"{len(rates)}/{count} bars received")
            else:
                print(f"✅ Successfully received {len(rates)} bars for {symbol}")
                return rates
        except Exception as e:
            print(f"❌ Attempt {attempt + 1}/{max_retries}: Error occurred: {str(e)}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    print(f"❌ Failed to get sufficient data for {symbol} after {max_retries} attempts")
    return None

def get_ohlc_data(symbol: str, timeframe: int, count: int, 
                 max_retries: int = 3, retry_delay: float = 1.0) -> Optional[np.ndarray]:
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Symbol {symbol} not found or not enabled")
        return None

    for attempt in range(max_retries):
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                print(f"❌ Attempt {attempt + 1}/{max_retries}: No data for {symbol}")
            elif len(rates) < count:
                print(f"❌ Attempt {attempt + 1}/{max_retries}: Insufficient data: {len(rates)}/{count}")
            else:
                print(f"✅ Successfully received {len(rates)} bars for {symbol}")
                return rates
            time.sleep(retry_delay)
        except Exception as e:
            print(f"❌ Attempt {attempt + 1}/{max_retries}: Error: {e}")
    print(f"❌ Failed to get data for {symbol} after {max_retries} attempts")
    return None

def rsi_analysis(symbol, timeframe=mt5.TIMEFRAME_M5, rsi_period=14, rsi_oversold=30, rsi_overbought=70):
    global order_buy, order_sel
    try:
        timeframes = [timeframe, timeframe*3, timeframe*6] 
        rsi_values = []
        price_trends = []
        
        for tf in timeframes:
            rates = get_rates(symbol, tf, 200)  
            if rates is None or len(rates) < rsi_period + 20:  
                continue
                
            closes = np.array([r['close'] for r in rates], dtype=np.float64)
            rsi = talib.RSI(closes, timeperiod=rsi_period)
            if len(rsi) < 2 or np.any(np.isnan(rsi[-2:])):
                continue
                
            ema_fast = talib.EMA(closes, timeperiod=9)
            ema_slow = talib.EMA(closes, timeperiod=21)
            trend_up = ema_fast[-1] > ema_slow[-1] if not (np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1])) else True
            
            rsi_values.append({
                'value': rsi[-1],
                'prev': rsi[-2],
                'trend': trend_up,
                'weight': 1.0 if tf == timeframe else 0.7  
            })
            price_trends.append(trend_up)
        
        if not rsi_values:
            order_sel[0] = 1
            return False
            
        total_weight = sum(r['weight'] for r in rsi_values)
        weighted_rsi = sum(r['value'] * r['weight'] for r in rsi_values) / total_weight
        avg_trend = sum(1 if r['trend'] else -1 for r in rsi_values) / len(rsi_values) > 0
        
        price_highs = np.array([r['high'] for r in rates[-20:]], dtype=np.float64)
        price_lows = np.array([r['low'] for r in rates[-20:]], dtype=np.float64)
        rsi_highs = rsi[-20:]
        
        price_low_idx = np.argmin(price_lows)
        rsi_low_idx = np.argmin(rsi_highs)
        bullish_div = (price_low_idx < len(price_lows) - 3 and 
                      rsi_low_idx > price_low_idx and 
                      price_lows[price_low_idx] < price_lows[price_low_idx-1] and
                      rsi_highs[rsi_low_idx] > rsi_highs[rsi_low_idx-1])
        
        price_high_idx = np.argmax(price_highs)
        rsi_high_idx = np.argmax(rsi_highs)
        bearish_div = (price_high_idx < len(price_highs) - 3 and 
                      rsi_high_idx > price_high_idx and 
                      price_highs[price_high_idx] > price_highs[price_high_idx-1] and
                      rsi_highs[rsi_high_idx] < rsi_highs[rsi_high_idx-1])
        
        score = 1
        
        if weighted_rsi < rsi_oversold:
            score = min(6, int((rsi_oversold - weighted_rsi) / 2) + 2)
        elif weighted_rsi > rsi_overbought:
            score = min(6, int((weighted_rsi - rsi_overbought) / 2) + 2)
        
        if (weighted_rsi < 50 and avg_trend) or (weighted_rsi > 50 and not avg_trend):
            score = max(1, score - 1)
            
        if bullish_div and not bearish_div:
            score = min(6, score + 2)
        elif bearish_div and not bullish_div:
            score = min(6, score + 2)
        
        buy_condition = (weighted_rsi < rsi_oversold or (weighted_rsi < 50 and avg_trend) or bullish_div)
        sell_condition = (weighted_rsi > rsi_overbought or (weighted_rsi > 50 and not avg_trend) or bearish_div)
        
        order_buy[0] = score if buy_condition and not sell_condition else 0
        order_sel[0] = score if sell_condition and not buy_condition else 1 
        
        if order_buy[0] == 0 and order_sel[0] == 0:
            order_sel[0] = 1  
            
        signal = f"{'BUY' if order_buy[0] > 0 else 'SELL'} (RSI: {weighted_rsi:.1f}, Score: {score}/6, Trend: {'Up' if avg_trend else 'Down'})"
        print(f"✅ RSI Analysis | {symbol}: {signal}")
        return True
        
    except Exception as e:
        print(f"❌ RSI Analysis Error for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        order_sel[0] = 1  
        return False

def macd_analysis(symbol, timeframe=mt5.TIMEFRAME_M5, fast_period=12, slow_period=26, signal_period=9):
    global order_buy, order_sel
    try:
        timeframes = [timeframe, timeframe*3]  
        macd_signals = []
        
        for tf in timeframes:
            rates = get_rates(symbol, tf, 200)
            if rates is None or len(rates) < slow_period + signal_period + 20:
                continue
                
            closes = np.array([r['close'] for r in rates], dtype=np.float64)
            highs = np.array([r['high'] for r in rates], dtype=np.float64)
            lows = np.array([r['low'] for r in rates], dtype=np.float64)
            
            macd, signal_line, hist = talib.MACD(
                closes, 
                fastperiod=fast_period, 
                slowperiod=slow_period, 
                signalperiod=signal_period
            )
            
            if np.isnan(hist[-1]) or np.isnan(macd[-1]) or np.isnan(signal_line[-1]):
                continue
                
            ema_fast = talib.EMA(closes, timeperiod=9)
            ema_slow = talib.EMA(closes, timeperiod=21)
            trend_up = ema_fast[-1] > ema_slow[-1] if not (np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1])) else True
            
            current_macd, prev_macd = macd[-1], macd[-2] if len(macd) > 1 else macd[-1]
            current_signal, prev_signal = signal_line[-1], signal_line[-2] if len(signal_line) > 1 else signal_line[-1]
            current_hist, prev_hist = hist[-1], hist[-2] if len(hist) > 1 else hist[-1]
            current_price, prev_price = closes[-1], closes[-2] if len(closes) > 1 else closes[-1]
            
            bullish_cross = (prev_macd <= prev_signal and current_macd > current_signal)
            bearish_cross = (prev_macd >= prev_signal and current_macd < current_signal)
            
            lookback = min(20, len(closes) // 2) 
            price_highs = highs[-lookback:]
            price_lows = lows[-lookback:]
            hist_highs = hist[-lookback:]
            
            def find_extremes(prices, lookback=5):
                highs, lows = [], []
                for i in range(lookback, len(prices)-lookback):
                    window = prices[i-lookback:i+lookback+1]
                    if prices[i] == max(window):
                        highs.append((i, prices[i]))
                    elif prices[i] == min(window):
                        lows.append((i, prices[i]))
                return highs, lows
            
            price_highs_idx, price_lows_idx = find_extremes(price_highs)
            hist_highs_idx, hist_lows_idx = find_extremes(hist_highs)
            
            bullish_div = False
            bearish_div = False
            
            if len(price_lows_idx) >= 2 and len(hist_lows_idx) >= 2:
                if (price_lows_idx[-1][1] < price_lows_idx[-2][1] and 
                    hist_lows_idx[-1][1] > hist_lows_idx[-2][1]):
                    bullish_div = True
                elif (price_lows_idx[-1][1] > price_lows_idx[-2][1] and 
                      hist_lows_idx[-1][1] < hist_lows_idx[-2][1]):
                    bullish_div = True
            
            if len(price_highs_idx) >= 2 and len(hist_highs_idx) >= 2:
                if (price_highs_idx[-1][1] > price_highs_idx[-2][1] and 
                    hist_highs_idx[-1][1] < hist_highs_idx[-2][1]):
                    bearish_div = True
                elif (price_highs_idx[-1][1] < price_highs_idx[-2][1] and 
                      hist_highs_idx[-1][1] > hist_highs_idx[-2][1]):
                    bearish_div = True
            
            hist_momentum = current_hist - prev_hist
            hist_acceleration = hist_momentum - (prev_hist - (hist[-3] if len(hist) > 2 else prev_hist))
            
            score = 1
            
            hist_strength = abs(current_hist) / np.mean(np.abs(hist[-20:])) if np.mean(np.abs(hist[-20:])) > 0 else 1.0
            score = min(7, int(hist_strength * 2) + 1)
            
            if hist_momentum > 0 and hist_acceleration > 0:
                score = min(7, score + 2)
            elif hist_momentum < 0 and hist_acceleration < 0:
                score = min(7, score + 2)
            
            if bullish_cross or bullish_div:
                score = min(7, score + 2)
            if bearish_cross or bearish_div:
                score = min(7, score + 2)
            
            buy_signal = (bullish_cross or bullish_div) and (trend_up or current_hist > 0)
            sell_signal = (bearish_cross or bearish_div) and (not trend_up or current_hist < 0)
            
            macd_signals.append({
                'timeframe': tf,
                'score': score,
                'buy': buy_signal,
                'sell': sell_signal,
                'hist': current_hist,
                'weight': 1.0 if tf == timeframe else 0.7 
            })
        
        if not macd_signals:
            order_sel[1] = 1
            return False
        
        total_weight = sum(s['weight'] for s in macd_signals)
        weighted_buy_score = sum(s['score'] * s['weight'] for s in macd_signals if s['buy']) / total_weight
        weighted_sell_score = sum(s['score'] * s['weight'] for s in macd_signals if s['sell']) / total_weight
        
        buy_confirmations = sum(1 for s in macd_signals if s['buy'])
        sell_confirmations = sum(1 for s in macd_signals if s['sell'])
        
        min_confirmations = 1
        
        if buy_confirmations >= min_confirmations and weighted_buy_score > weighted_sell_score:
            order_buy[1] = int(weighted_buy_score)
            order_sel[1] = 0
            signal_dir = 'BUY'
            final_score = order_buy[1]
        elif sell_confirmations >= min_confirmations and weighted_sell_score > weighted_buy_score:
            order_sel[1] = int(weighted_sell_score)
            order_buy[1] = 0
            signal_dir = 'SELL'
            final_score = order_sel[1]
        else:
            order_sel[1] = 1
            order_buy[1] = 0
            signal_dir = 'SELL'
            final_score = 1
        
        hist_avg = np.mean([abs(s['hist']) for s in macd_signals])
        signal_info = (
            f"{signal_dir} (Score: {final_score}/7, "
            f"Buy Conf: {buy_confirmations}/{len(macd_signals)}, "
            f"Sell Conf: {sell_confirmations}/{len(macd_signals)}, "
            f"Hist: {macd_signals[0]['hist']:.6f})"
        )
        print(f"✅ MACD Analysis | {symbol}: {signal_info}")
        return True
        
    except Exception as e:
        print(f"❌ MACD Analysis Error for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        order_sel[1] = 1 
        return False

def fibonacci_analysis(symbol, timeframe=mt5.TIMEFRAME_M5, lookback=200, fib_levels=[0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.14, 4.236]):
    global order_buy, order_sel
    try:
        timeframes = [timeframe, timeframe*2, timeframe*4]  
        fib_signals = []
        
        for tf in timeframes:
            rates = get_rates(symbol, tf, lookback + 100)
            if rates is None or len(rates) < 100: 
                continue
                
            highs = np.array([r['high'] for r in rates], dtype=np.float64)
            lows = np.array([r['low'] for r in rates], dtype=np.float64)
            closes = np.array([r['close'] for r in rates], dtype=np.float64)
            current_price = closes[-1]
            
            ema_fast = talib.EMA(closes, timeperiod=9)
            ema_slow = talib.EMA(closes, timeperiod=21)
            trend_up = ema_fast[-1] > ema_slow[-1] if not (np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1])) else True
            
            def find_swing_points(highs, lows, window=5):
                high_points, low_points = [], []
                
                for i in range(window, len(highs) - window):
                    is_high = True
                    for j in range(1, window + 1):
                        if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                            is_high = False
                            break
                    if is_high:
                        high_points.append((i, highs[i]))
                    
                    is_low = True
                    for j in range(1, window + 1):
                        if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                            is_low = False
                            break
                    if is_low:
                        low_points.append((i, lows[i]))
                
                if len(high_points) > 0 and len(low_points) > 0:
                    max_high = max(h[1] for h in high_points)
                    min_low = min(l[1] for l in low_points)
                    
                    price_range = max_high - min_low
                    threshold = price_range * 0.1  
                    
                    high_points = [h for h in high_points if h[1] > min_low + threshold]
                    low_points = [l for l in low_points if l[1] < max_high - threshold]
                
                return high_points, low_points
            
            window = max(3, min(10, len(highs) // 20))  
            high_points, low_points = find_swing_points(highs, lows, window)
            
            if len(high_points) < 2 or len(low_points) < 2:
                continue  
            
            recent_highs = sorted(high_points, key=lambda x: x[0], reverse=True)[:3]
            recent_lows = sorted(low_points, key=lambda x: x[0], reverse=True)[:3]
            
            if recent_highs[0][0] > recent_lows[0][0]: 
                trend = "downtrend"
                swing_high = recent_highs[0][1]
                swing_low = min(l[1] for l in low_points if l[0] < recent_highs[0][0])
            else: 
                trend = "uptrend"
                swing_low = recent_lows[0][1]
                swing_high = max(h[1] for h in high_points if h[0] < recent_lows[0][0])
            
            fib_range = swing_high - swing_low
            if fib_range == 0: 
                continue
                
            fib_prices = {}
            for level in fib_levels:
                if trend == "uptrend":
                    price = swing_high - (fib_range * level)
                else:  
                    price = swing_low + (fib_range * level)
                fib_prices[level] = price
            
            min_distance = float('inf')
            closest_level = 0
            for level, price in fib_prices.items():
                distance = abs(current_price - price) / current_price  
                if distance < min_distance:
                    min_distance = distance
                    closest_level = level
            
            score = 1
            
            if closest_level in [0.5, 0.618, 0.786, 1.0, 1.272, 1.618]:
                score = min(4, score + 2)
            else:
                score = min(4, score + 1)
            
            if (trend == "uptrend" and current_price > swing_low) or \
               (trend == "downtrend" and current_price < swing_high):
                score = min(4, score + 1)
            
            def check_price_action(rates, trend):
                if len(rates) < 3:
                    return False
                
                current = rates[-1]
                prev = rates[-2]
                prev2 = rates[-3]
                
                if (trend == "uptrend" and 
                    current['close'] > current['open'] and 
                    prev['close'] < prev['open'] and 
                    current['close'] > prev['open'] and 
                    current['open'] < prev['close'] and
                    (current['close'] - current['open']) > (prev['open'] - prev['close'])): 
                    return True
                
                elif (trend == "downtrend" and 
                      current['close'] < current['open'] and  
                      prev['close'] > prev['open'] and 
                      current['close'] < prev['open'] and 
                      current['open'] > prev['close'] and
                      (current['open'] - current['close']) > (prev['close'] - prev['open'])):  
                    return True
                
                body_size = abs(current['close'] - current['open'])
                upper_wick = current['high'] - max(current['open'], current['close'])
                lower_wick = min(current['open'], current['close']) - current['low']
                
                if (trend == "uptrend" and 
                    lower_wick > 2 * body_size and 
                    upper_wick < body_size * 0.5 and
                    current['close'] > current['open']):  
                    return True
                
                elif (trend == "downtrend" and 
                      upper_wick > 2 * body_size and 
                      lower_wick < body_size * 0.5 and
                      current['close'] < current['open']): 
                    return True
                
                return False
            
            price_action_confirmation = check_price_action(rates, trend)
            
            if price_action_confirmation:
                score = min(4, score + 1)
            
            if trend == "uptrend" and current_price > swing_low and (price_action_confirmation or trend_up):
                signal = "BUY"
            elif trend == "downtrend" and current_price < swing_high and (price_action_confirmation or not trend_up):
                signal = "SELL"
            else:
                signal = None
            
            if signal:
                fib_signals.append({
                    'timeframe': tf,
                    'score': score,
                    'signal': signal,
                    'level': closest_level,
                    'price': fib_prices[closest_level],
                    'weight': 1.0 if tf == timeframe else 0.6  
                })
        
        if not fib_signals:
            order_sel[2] = 1
            return False
        
        total_weight = sum(s['weight'] for s in fib_signals)
        buy_score = sum(s['score'] * s['weight'] for s in fib_signals if s['signal'] == 'BUY') / total_weight
        sell_score = sum(s['score'] * s['weight'] for s in fib_signals if s['signal'] == 'SELL') / total_weight
        
        buy_confirmations = sum(1 for s in fib_signals if s['signal'] == 'BUY')
        sell_confirmations = sum(1 for s in fib_signals if s['signal'] == 'SELL')
        
        min_confirmations = 1  
        
        if buy_confirmations >= min_confirmations and buy_score > sell_score:
            order_buy[2] = int(buy_score)
            order_sel[2] = 0
            signal_dir = 'BUY'
            final_score = order_buy[2]
            level = next((s['level'] for s in fib_signals if s['signal'] == 'BUY'), 0)
        elif sell_confirmations >= min_confirmations and sell_score > buy_score:
            order_sel[2] = int(sell_score)
            order_buy[2] = 0
            signal_dir = 'SELL'
            final_score = order_sel[2]
            level = next((s['level'] for s in fib_signals if s['signal'] == 'SELL'), 0)
        else:
            order_sel[2] = 1
            order_buy[2] = 0
            signal_dir = 'SELL'
            final_score = 1
            level = 0
        
        signal_info = (
            f"{signal_dir} (Score: {final_score}/4, "
            f"Level: {level*100:.1f}%, "
            f"Buy Conf: {buy_confirmations}/{len(fib_signals)}, "
            f"Sell Conf: {sell_confirmations}/{len(fib_signals)})"
        )
        print(f"✅ Fibonacci Analysis | {symbol}: {signal_info}")
        return True
        
    except Exception as e:
        print(f"❌ Fibonacci Analysis Error for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        order_sel[2] = 1  
        return False

def moving_averages(symbol, timeframe=mt5.TIMEFRAME_M5, sma_short_period=50, sma_long_period=200):
    global order_buy, order_sel
    try:
        rates = get_rates(symbol, timeframe, sma_long_period + 10)
        if rates is None:
            order_sel[3] = 1
            return False
        closes = np.array([r['close'] for r in rates], dtype=np.float64)
        if len(closes) < sma_long_period:
            order_sel[3] = 1
            return False
        sma_short = talib.SMA(closes, timeperiod=sma_short_period)
        sma_long = talib.SMA(closes, timeperiod=sma_long_period)
        if np.isnan(sma_short[-1]) or np.isnan(sma_long[-1]):
            order_sel[3] = 1
            return False
        sma_short_current, sma_short_prev = sma_short[-1], sma_short[-2] if len(sma_short) > 1 else sma_short[-1]
        sma_long_current, sma_long_prev = sma_long[-1], sma_long[-2] if len(sma_long) > 1 else sma_long[-1]
        current_close = closes[-1]

        bullish_crossover = sma_short_prev <= sma_long_prev and sma_short_current > sma_long_current
        bearish_crossover = sma_short_prev >= sma_long_prev and sma_short_current < sma_long_current
        distance_to_sma_short = abs(current_close - sma_short_current) / sma_short_current
        score = max(1, min(5, int(distance_to_sma_short / 0.005) + 1))
        if bullish_crossover:
            score = min(5, score + 2)
        elif bearish_crossover:
            score = min(5, score + 2)

        order_buy[3] = score if bullish_crossover or current_close > sma_short_current else 0
        order_sel[3] = score if bearish_crossover or current_close < sma_short_current else 0
        signal = f"{'BUY' if order_buy[3] > 0 else 'SELL'} (SMA{sma_short_period}: {sma_short_current:.5f}, SMA{sma_long_period}: {sma_long_current:.5f}, Score: {score}/5)"
        print(f"Moving Averages Analysis | {symbol}: {signal}")
        return True
    except Exception as e:
        print(f"❌ Moving Averages Analysis Error for {symbol}: {str(e)}")
        order_sel[3] = 1
        return False
        
def ATR_Average_True_Range(symbol, timeframe=mt5.TIMEFRAME_M5, atr_period=14, lookback=100):
    global order_buy, order_sel
    try:
        rates = get_rates(symbol, timeframe, lookback)
        if rates is None or len(rates) < atr_period * 2: 
            print(f"❌ Insufficient data for ATR calculation on {symbol}")
            order_sel[4] = 1
            return False
            
        high = np.array([r['high'] for r in rates], dtype=np.float64)
        low = np.array([r['low'] for r in rates], dtype=np.float64)
        close = np.array([r['close'] for r in rates], dtype=np.float64)
        
        atr = talib.ATR(high, low, close, timeperiod=atr_period)
        if np.any(np.isnan(atr[-5:])):  
            print(f"❌ Invalid ATR values for {symbol}")
            order_sel[4] = 1
            return False
            
        sma = talib.SMA(close, timeperiod=20)
        if np.isnan(sma[-1]):
            print(f"❌ Invalid SMA values for {symbol}")
            order_sel[4] = 1
            return False
            
        trend_up = close[-1] > sma[-1]
        current_atr = atr[-1]
        prev_atr = atr[-2] if len(atr) > 1 else current_atr
        
        atr_rising = current_atr > prev_atr
        price_volatility = current_atr / close[-1] 
        
        score = min(7, max(1, int(price_volatility * 1000))) 
        
        if trend_up and atr_rising:
            score = min(7, score + 2) 
        elif not trend_up and atr_rising:
            score = min(7, score + 1)  
            
        if trend_up and atr_rising:
            order_buy[4] = score
            order_sel[4] = 0
        elif not trend_up and atr_rising:
            order_buy[4] = 0
            order_sel[4] = score
        else:
            order_buy[4] = 0
            order_sel[4] = 1
            
        signal = f"{'BUY' if order_buy[5] > 0 else 'SELL'} (ATR: {current_atr:.5f}, Score: {score}/7, Trend: {'Up' if trend_up else 'Down'})"
        print(f"✅ ATR Analysis | {symbol}: {signal}")
        return True
        
    except Exception as e:
        print(f"❌ ATR Analysis Error for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        order_sel[4] = 1
        return False
    
def ichimoku_analysis(symbol, timeframe=mt5.TIMEFRAME_M5, tenkan_period=9, kijun_period=26, senkou_period=52, cloud_shift=26):
    global order_buy, order_sel
    try:
        required_bars = max(senkou_period + cloud_shift, kijun_period + cloud_shift, tenkan_period + cloud_shift)
        rates = get_rates(symbol, timeframe, required_bars + 10)
        if rates is None:
            order_sel[5] = 1
            return False
        high = np.array([r['high'] for r in rates], dtype=np.float64)
        low = np.array([r['low'] for r in rates], dtype=np.float64)
        close = np.array([r['close'] for r in rates], dtype=np.float64)
        if len(close) < required_bars:
            order_sel[5] = 1
            return False
        tenkan = (talib.MAX(high, tenkan_period) + talib.MIN(low, tenkan_period)) / 2
        kijun = (talib.MAX(high, kijun_period) + talib.MIN(low, kijun_period)) / 2
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (talib.MAX(high, senkou_period) + talib.MIN(low, senkou_period)) / 2
        if np.isnan(tenkan[-1]) or np.isnan(kijun[-1]) or np.isnan(senkou_a[-1]) or np.isnan(senkou_b[-1]):
            order_sel[5] = 1
            return False
        current_idx = len(close) - 1
        cloud_idx = current_idx - cloud_shift if current_idx >= cloud_shift else 0
        cloud_top = max(senkou_a[cloud_idx], senkou_b[cloud_idx])
        cloud_bottom = min(senkou_a[cloud_idx], senkou_b[cloud_idx])
        tenkan_current, tenkan_prev = tenkan[-1], tenkan[-2] if len(tenkan) > 1 else tenkan[-1]
        kijun_current, kijun_prev = kijun[-1], kijun[-2] if len(kijun) > 1 else kijun[-1]
        current_close = close[-1]
        sma = talib.SMA(close, timeperiod=20)[-1]
        trend_up = current_close > sma if not np.isnan(sma) else True

        bullish_crossover = tenkan_prev <= kijun_prev and tenkan_current > kijun_current
        bearish_crossover = tenkan_prev >= kijun_prev and tenkan_current < kijun_current
        distance_to_cloud = abs(current_close - cloud_top) / cloud_top if current_close > cloud_top else abs(current_close - cloud_bottom) / cloud_bottom
        score = max(1, min(7, int(distance_to_cloud / 0.005) + 1))
        if bullish_crossover:
            score = min(7, score + 2)
        elif bearish_crossover:
            score = min(7, score + 2)

        order_buy[5] = score if current_close > cloud_top or bullish_crossover else 0
        order_sel[5] = score if current_close < cloud_bottom or bearish_crossover else 0
        signal = f"{'BUY' if order_buy[5] > 0 else 'SELL'} (Tenkan: {tenkan_current:.5f}, Kijun: {kijun_current:.5f}, Score: {score}/7)"
        print(f"Ichimoku Analysis | {symbol}: {signal}")
        return True
    except Exception as e:
        print(f"❌ Ichimoku Analysis Error for {symbol}: {str(e)}")
        order_sel[5] = 1
        return False

def bollinger_bands(symbol, timeframe=mt5.TIMEFRAME_M5, bb_period=20, bb_dev=2.0, sma_period=20):
    global order_buy, order_sel
    try:
        rates = get_rates(symbol, timeframe, bb_period + 10)
        if rates is None:
            order_sel[6] = 1
            return False
        closes = np.array([r['close'] for r in rates], dtype=np.float64)
        if len(closes) < bb_period:
            order_sel[6] = 1
            return False
        upper, middle, lower = talib.BBANDS(closes, timeperiod=bb_period, nbdevup=bb_dev, nbdevdn=bb_dev)
        if np.isnan(upper[-1]) or np.isnan(middle[-1]) or np.isnan(lower[-1]):
            order_sel[6] = 1
            return False
        current_close = closes[-1]
        percent_b = (current_close - lower[-1]) / (upper[-1] - lower[-1]) if (upper[-1] - lower[-1]) != 0 else 0.5
        band_width = (upper[-1] - lower[-1]) / middle[-1]
        band_width_prev = (upper[-2] - lower[-2]) / middle[-2] if len(upper) > 1 else band_width
        squeeze_threshold = 0.02
        is_squeeze = band_width < squeeze_threshold
        is_expansion = band_width > band_width_prev and band_width > squeeze_threshold
        sma = talib.SMA(closes, timeperiod=sma_period)[-1]
        trend_up = current_close > sma if not np.isnan(sma) else True

        score = max(1, min(6, int(abs(percent_b - 0.5) / 0.1) + 1))
        if is_expansion:
            score = min(6, score + 2)
        if is_squeeze:
            score = max(1, score - 2)

        order_buy[6] = score if percent_b < 0.2 and trend_up and not is_squeeze else 0
        order_sel[6] = score if percent_b > 0.8 and not trend_up and not is_squeeze else 0
        signal = f"{'BUY' if order_buy[6] > 0 else 'SELL'} (Percent B: {percent_b:.2f}, Band Width: {band_width:.4f}, Score: {score}/6)"
        print(f"Bollinger Bands Analysis | {symbol}: {signal}")
        return True
    except Exception as e:
        print(f"❌ Bollinger Bands Analysis Error for {symbol}: {str(e)}")
        order_sel[6] = 1
        return False

def volume_analysis(symbol, timeframe=mt5.TIMEFRAME_M5, lookback=100, ma_period=14, recent_candles=5):
    global order_buy, order_sel
    try:
        rates = get_rates(symbol, timeframe, lookback)
        if rates is None:
            order_sel[7] = 1
            return False
        df = pd.DataFrame(rates)
        if 'tick_volume' not in df.columns or len(df) < ma_period + recent_candles:
            order_sel[7] = 1
            return False
        volume = np.array(df['tick_volume'], dtype=np.float64)
        closes = np.array(df['close'], dtype=np.float64)
        avg_volume = talib.MA(volume, timeperiod=ma_period)[-1]
        if np.isnan(avg_volume) or avg_volume == 0:
            order_sel[7] = 1
            return False
        volume_ratio = volume[-1] / avg_volume
        volume_std = np.std(volume[-ma_period:])
        volume_volatility = volume_std / avg_volume if avg_volume != 0 else 1.0
        high_volatility = volume_volatility > 1.5
        recent = df.iloc[-recent_candles:]
        bull_candles = sum(recent['close'] > recent['open'])
        bear_candles = sum(recent['close'] < recent['open'])
        price_trend = recent['close'].iloc[-1] > recent['close'].iloc[0]
        sma = talib.SMA(closes, timeperiod=ma_period)[-1]
        trend_up = closes[-1] > sma if not np.isnan(sma) else True

        score = max(1, min(5, int(volume_ratio) + 1))
        if bull_candles > bear_candles and price_trend and not high_volatility:
            score = min(5, score + 1)
        elif bear_candles > bull_candles and not price_trend and not high_volatility:
            score = min(5, score + 1)
        if high_volatility:
            score = max(1, score - 1)

        order_buy[7] = score if volume_ratio > 1.5 and bull_candles > bear_candles and price_trend and trend_up else 0
        order_sel[7] = score if volume_ratio > 1.5 and bear_candles > bull_candles and not price_trend and not trend_up else 0
        signal = f"{'BUY' if order_buy[7] > 0 else 'SELL'} (Volume Ratio: {volume_ratio:.2f}, Bull Candles: {bull_candles}, Score: {score}/5)"
        print(f"Volume Analysis | {symbol}: {signal}")
        return True
    except Exception as e:
        print(f"❌ Volume Analysis Error for {symbol}: {str(e)}")
        order_sel[7] = 1
        return False

def adx_analysis(symbol: str, timeframe: int = mt5.TIMEFRAME_M5, 
                adx_period: int = 14) -> Tuple[str, float, int, int]:
    count = adx_period + 100
    data = get_ohlc_data(symbol, timeframe, count)
    if data is None or len(data) < adx_period:
        print(f"❌ Insufficient data for ADX: {len(data) if data else 0}/{count}")
        return 'Neutral', 0.0, 1, 1

    ohlc = {
        'high': np.array([bar['high'] for bar in data]),
        'low': np.array([bar['low'] for bar in data]),
        'close': np.array([bar['close'] for bar in data]),
        'open': np.array([bar['open'] for bar in data])
    }

    adx = talib.ADX(ohlc['high'], ohlc['low'], ohlc['close'], timeperiod=adx_period)
    if len(adx) < 1 or np.isnan(adx[-1]):
        print("❌ Invalid ADX value")
        return 'Neutral', 0.0, 1, 1

    plus_di = talib.PLUS_DI(ohlc['high'], ohlc['low'], ohlc['close'], timeperiod=adx_period)[-1]
    minus_di = talib.MINUS_DI(ohlc['high'], ohlc['low'], ohlc['close'], timeperiod=adx_period)[-1]
    engulfing = talib.CDLENGULFING(ohlc['open'], ohlc['high'], ohlc['low'], ohlc['close'])[-1]

    signal = 'Neutral'
    confidence = 0.0
    buy_score = 1
    sell_score = 1
    if adx[-1] > 25:
        confidence = min(adx[-1] / 100, 0.95)
        if plus_di > minus_di + 2:
            signal = 'Buy'
            buy_score = 8
            confidence += 0.05 * (engulfing == 100)
        elif minus_di > plus_di + 2:
            signal = 'Sell'
            sell_score = 8
            confidence += 0.05 * (engulfing == -100)
        confidence = min(confidence, 0.99)

    return signal, confidence, buy_score, sell_score

def range_trading(symbol: str, timeframe: int = mt5.TIMEFRAME_M5, 
                 period: int = 20) -> Tuple[str, float, int, int]:
    data = get_ohlc_data(symbol, timeframe, count=period)
    if data is None or len(data) < period:
        print(f"❌ Insufficient data for range trading: {len(data) if data else 0}/{period}")
        return 'Neutral', 0.0, 1, 1

    high = np.array([bar['high'] for bar in data])
    low = np.array([bar['low'] for bar in data])
    close = np.array([bar['close'] for bar in data])

    support = min(low[-period:])
    resistance = max(high[-period:])
    latest_close = close[-1]

    upper, _, lower = talib.BBANDS(close, timeperiod=period, nbdevup=2, nbdevdn=2, matype=0)
    if np.isnan(upper[-1]) or np.isnan(lower[-1]):
        print("❌ Invalid Bollinger Bands values")
        return 'Neutral', 0.0, 1, 1

    signal = 'Neutral'
    confidence = 0.6
    buy_score = 1
    sell_score = 1
    if latest_close <= support * 1.005 or latest_close <= lower[-1] * 1.005:
        signal = 'Buy'
        confidence = 0.85
        buy_score = 4
    elif latest_close >= resistance * 0.995 or latest_close >= upper[-1] * 0.995:
        signal = 'Sell'
        confidence = 0.85
        sell_score = 4
    return signal, confidence, buy_score, sell_score

def target_time(symbol: str) -> Tuple[str, float]:
    global order_buy, order_sel
    if not initialize_mt5():
        return 'Error', 0.0
    current_hour = datetime.now().strftime('%H')
    if current_hour in TIME_ZONES['useful']:
        signal, confidence, buy_score, sell_score = adx_analysis(symbol)
    elif current_hour in TIME_ZONES['neutral']:
        signal, confidence, buy_score, sell_score = range_trading(symbol)
    else:
        signal, confidence, buy_score, sell_score = 'Neutral', 0.0, 0, 0

    order_buy[8] = buy_score
    order_sel[8] = sell_score

    print(f"Symbol: {symbol}, Time: {current_hour}, Signal: {signal}, \n"
          f"Confidence: {confidence:.2%}, Buy Array: {order_buy}, \n"
          f"Sell Array: {order_sel}")
    
def trading_decision(symbol, confidence_threshold=0.9, weights=None):
    global order_buy, order_sel
    order_buy = [0] * 9
    order_sel = [0] * 9
    try:
        if not mt5.initialize():
            print("❌ MetaTrader5 initialization failed")
            return None
        if weights is None:
            weights = {
                'rsi': 0.9,
                'macd': 1.3,
                'fibonacci': 0.9,
                'moving_averages': 1.0,
                'atr': 1.0,
                'ichimoku': 1.3,
                'boll': 1.3,
                'volume': 0.7,
                'tar': 1.3,
            }

        analyses = [
            rsi_analysis(symbol),
            macd_analysis(symbol),
            fibonacci_analysis(symbol),
            moving_averages(symbol),
            ATR_Average_True_Range(symbol),
            ichimoku_analysis(symbol),
            bollinger_bands(symbol),
            volume_analysis(symbol),
            target_time(symbol)
        ]
        failed_analyses = sum(1 for a in analyses if not a)
        if failed_analyses == len(analyses):
            print("❌ All Analyses Failed")
            order_sel = [1] * 9
            buy_sum = 0
            sell_sum = sum([score * weights[list(weights.keys())[i]] for i, score in enumerate(order_sel)])
            decision = False
            confidence = 1.0
        else:
            rates = get_rates(symbol, mt5.TIMEFRAME_M5, 20)
            closes = np.array([r['close'] for r in rates], dtype=np.float64)
            sma = talib.SMA(closes, timeperiod=20)[-1]
            trend_up = closes[-1] > sma if not np.isnan(sma) else True
            for i in range(9):
                if order_buy[i] == 0 and order_sel[i] == 0:
                    order_buy[i] = 1 if trend_up else 0
                    order_sel[i] = 1 if not trend_up else 0

            buy_sum = sum([score * weights[list(weights.keys())[i]] for i, score in enumerate(order_buy)])
            sell_sum = sum([score * weights[list(weights.keys())[i]] for i, score in enumerate(order_sel)])
            total_sum = buy_sum + sell_sum
            if total_sum == 0:
                decision = True if trend_up else False
                confidence = 0.0
            elif buy_sum >= sell_sum:
                decision = True
                confidence = buy_sum / total_sum
            else:
                decision = False
                confidence = sell_sum / total_sum

        print("\n" + "="*70)
        print(f"Final Decision | {symbol} | {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"Buy Scores: {order_buy} (Weighted Sum: {buy_sum:.2f})")
        print(f"Sell Scores: {order_sel} (Weighted Sum: {sell_sum:.2f})")
        print(f"Confidence: {confidence*100:.1f}%")
        print(f"Final Signal: {decision}")
        print("="*70 + "\n")

        if confidence < confidence_threshold:
            print(f"⚠️ Confidence below {confidence_threshold*100:.1f}%, defaulting to trend-based decision")
            decision = True if trend_up else False
            confidence = confidence_threshold

        return {
            'symbol': symbol,
            'decision': decision,
            'buy_sum': buy_sum,
            'sell_sum': sell_sum,
            'confidence': confidence
        }
    except Exception as e:
        print(f"❌ Trading Decision Error: {str(e)}")
        return None
    finally:
        mt5.shutdown()

def is_market_open(symbol):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Could not get symbol info for {symbol}")
            return False
        
        if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
            print(f"❌ Trading is not allowed for {symbol} (Trade mode: {symbol_info.trade_mode})")
            return False
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print("❌ Could not get tick data")
            return False
            
        server_time = datetime.fromtimestamp(tick.time)
    
        current_hour = server_time.hour
        current_minute = server_time.minute
        weekday = server_time.weekday()  
        
        if (weekday == 6 and current_hour >= 22 and current_minute >= 5) or \
           (0 <= weekday <= 4) or \
           (weekday == 5 and current_hour < 21):
            return True
        
        print(f"⏳ Market closed for {symbol} at {server_time}")
        return False
        
    except Exception as e:
        print(f"❌ Error checking market hours: {str(e)}")
        return False