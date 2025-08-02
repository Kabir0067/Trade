import MetaTrader5 as mt5
from datetime import *
import pandas as pd
import numpy as np
from info import *
import logging
import talib


def get_symbol_data(symbol, timeframe=mt5.TIMEFRAME_M1, bars=50):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) < 30:
            print(f"❌ Маълумоти нокифоя барои {symbol} ({len(rates)} бар)")
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['symbol'] = symbol
        return df
    except Exception as e:
        print(f"❌ Хато дар гирифтани маълумот барои {symbol}: {e}")
        return None

def calculate_volatility(df):
    try:
        df['high_low'] = df['high'] - df['low']
        df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
        df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        return df.iloc[-1]['atr'] if not np.isnan(df.iloc[-1]['atr']) else 0
    except Exception as e:
        print(f"❌ Хато дар ҳисоби тағйирёбӣ барои {df['symbol'].iloc[0]}: {e}")
        return 0

def calculate_trend_strength(df):
    try:
        if len(df) < 28:
            print(f"⚠️ Маълумоти нокифоя барои ADX барои {df['symbol'].iloc[0]}")
            return 0
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        return adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 0
    except Exception as e:
        print(f"❌ Хато дар ҳисоби қувваи тамоюл барои {df['symbol'].iloc[0]}: {e}")
        return 0

def calculate_rsi(df):
    try:
        if len(df) < 14:
            print(f"⚠️ Маълумоти нокифоя барои RSI барои {df['symbol'].iloc[0]}")
            return 50
        rsi = talib.RSI(df['close'].values, timeperiod=14)
        return rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else 50
    except Exception as e:
        print(f"❌ Хато дар ҳисоби RSI барои {df['symbol'].iloc[0]}: {e}")
        return 50

def calculate_macd(df):
    try:
        if len(df) < 26:
            print(f"⚠️ Маълумоти нокифоя барои MACD барои {df['symbol'].iloc[0]}")
            return 0
        macd, signal, _ = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        return macd[-1] - signal[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0
    except Exception as e:
        print(f"❌ Хато дар ҳисоби MACD барои {df['symbol'].iloc[0]}: {e}")
        return 0

def get_spread(symbol):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return float('inf')
        return symbol_info.spread
    except Exception as e:
        print(f"❌ Хато дар гирифтани спред барои {symbol}: {e}")
        return float('inf')

def get_volume(symbol):
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0
        return symbol_info.volume
    except Exception as e:
        print(f"❌ Хато дар гирифтани ҳаҷми савдо барои {symbol}: {e}")
        return 0

def analyze_symbols():
    if not initialize_mt5():
        print("❌ Пайвастшавӣ ноком шуд. Интихоби пешфарз: BTCUSDm.")
        return 'BTCUSDm'

    analysis_results = []
    
    for symbol in topics:
        df = get_symbol_data(symbol, timeframe=mt5.TIMEFRAME_M1, bars=50)
        if df is None:
            print(f"⚠️ {symbol} аз сабаби набудани маълумот нодида шуд")
            continue

        volatility = calculate_volatility(df)
        trend_strength = calculate_trend_strength(df)
        rsi = calculate_rsi(df)
        macd_diff = calculate_macd(df)
        spread = get_spread(symbol)
        volume = get_volume(symbol)
        
        max_spread = max_spreads.get(symbol, 200)
        min_volume = min_volumes.get(symbol, 1000)
        priority = priority_factors.get(symbol, 1.0)
        
        if spread > max_spread or volume < min_volume or volatility == 0:
            print(f"⚠️ {symbol} нодида шуд — спред: {spread}/{max_spread}, ҳаҷм: {volume}/{min_volume}")
            continue

        score = (volatility * trend_strength * volume) / (spread + 1) * priority
        
        if 40 < rsi < 60: 
            score *= 1.3
        elif (rsi > 70 and macd_diff < 0) or (rsi < 30 and macd_diff > 0): 
            score *= 1.5
            
        if abs(macd_diff) > 0:
            score *= 1.4 if macd_diff > 0 else 1.2

        analysis_results.append({
            'symbol': symbol,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'rsi': rsi,
            'macd_diff': macd_diff,
            'spread': spread,
            'volume': volume,
            'score': score
        })

    if not analysis_results:
        print("❌ Ҳеҷ ашёи мувофиқ ёфт нашуд. Интихоби пешфарз: BTCUSDm.")
        return 'BTCUSDm'
    
    analysis_results.sort(key=lambda x: x['score'], reverse=True)
    best = analysis_results[0]
    
    print(f"\n🔍 Натиҷаҳои таҳлил:")
    print(f"🏆 Беҳтарин ашё: {best['symbol']} | Нишондиҳанда: {best['score']:.2f}")
    print(f"📊 Тағйирёбӣ: {best['volatility']:.4f} | Қувваи равия: {best['trend_strength']:.2f}")
    print(f"📈 RSI: {best['rsi']:.2f} | MACD: {best['macd_diff']:.4f}")
    print(f"💵 Спред: {best['spread']} | Ҳаҷми савдо: {best['volume']}\n")
    
    return best['symbol']
