import MetaTrader5 as mt5
from bstsymbl import *
from analysis import *
import logging
import time



def check_autotrading_enabled():
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            print("❌ Аз терминал маълумотро пайдо карда натавонистем.")
            return False
        if not terminal_info.trade_allowed:
            print("❌ Автотрейдинг хомуш аст.")
            return False
        return True
    except Exception as e:
        print(f"❌ Ҳангоми тафтиши Автотрейдинг хатоги шуд: {e}")
        return False


#---------------------------------Open order-----------------------------
def open_order_buy(money, symbol):
    if not  initialize_mt5():
        return
    
    if not is_market_open(symbol):
        print(f"❌ Рынок закрыт для {symbol}")
        mt5.shutdown()
        return False
    
    if not  check_autotrading_enabled():
        mt5.shutdown()
        return

    if not mt5.symbol_select(symbol, True):
        print(f"❌ Не удалось выбрать символ {symbol}")
        mt5.shutdown()
        return

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Информация о символе {symbol} недоступна.")
        mt5.shutdown()
        return

    if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        print(f"❌ Торговля по символу {symbol} отключена.")
        mt5.shutdown()
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("❌ Не удалось получить котировку.")
        mt5.shutdown()
        return

    price = tick.ask

    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step

    lot = money / 100000
    lot = round(lot / lot_step) * lot_step
    lot = max(min_lot, min(lot, max_lot))

    print(f"📊 Объем к открытию: {lot} | Ask цена: {price}")

    tick_value = symbol_info.trade_tick_value
    point = symbol_info.point

    tp_points = 2.0 / (tick_value * lot)
    sl_points = 500000.0 / (tick_value * lot)

    tp = price + tp_points * point
    sl = price - sl_points * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 20,
        "sl": sl,
        "tp": tp,
        "magic": 123456,
        "comment": "Buy order from Python script",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        print("❌ order_send вернул None. Проверьте параметры запроса и состояние терминала.")
        mt5.shutdown()
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Ошибка открытия ордера: {result.retcode}")
        print(f"➡️ Подробности: {result}")
    else:
        print(f"✅ Ордер успешно открыт! Ticket: {result.order}")

    mt5.shutdown()

def open_order_sell(money, symbol):
    if not  initialize_mt5():
        return
    
    if not is_market_open(symbol):
        print(f"❌ Рынок закрыт для {symbol}")
        mt5.shutdown()
        return False
    
    if not  check_autotrading_enabled():
        mt5.shutdown()
        return
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Не удалось выбрать символ {symbol}")
        mt5.shutdown()
        return
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"❌ Информация о символе {symbol} недоступна.")
        mt5.shutdown()
        return
    if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
        print(f"❌ Торговля по символу {symbol} отключена.")
        mt5.shutdown()
        return
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("❌ Не удалось получить котировку.")
        mt5.shutdown()
        return

    price = tick.bid
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step
    lot = money / 100000
    lot = round(lot / lot_step) * lot_step
    lot = max(min_lot, min(lot, max_lot))

    print(f"📊 Объем к открытию: {lot} | Bid цена: {price}")
    print(f"Минимальный лот: {min_lot}, Максимальный лот: {max_lot}, Шаг лота: {lot_step}")
    tick_value = symbol_info.trade_tick_value
    point = symbol_info.point

    tp_points = 25.0 / (tick_value * lot)
    sl_points = 50000.0 / (tick_value * lot)

    tp = price - tp_points * point
    sl = price + sl_points * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "sl": sl,
        "tp": tp,
        "comment": "Sell order from Python script",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None:
        print("❌ order_send вернул None. Проверьте параметры запроса и состояние терминала.")
        mt5.shutdown()
        return

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Ошибка открытия ордера: {result.retcode}")
        print(f"➡️ Подробности: {result}")
    else:
        print(f"✅ Ордер успешно открыт! Ticket: {result.order}")

    mt5.shutdown()
#------------------------------------------------------------------------

def wait_for_trade_completion():
    try:
        if not initialize_mt5():
            return        
        start_time = time.time()
        max_wait_time = 3600  
        while time.time() - start_time < max_wait_time:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                logging.info("✅ Ҳамаи савдоҳо пӯшида шуданд")
                break
            logging.info(f"⏳ Интизори пӯшидани савдоҳо... Боз: {len(positions)}")
            time.sleep(5)
        mt5.shutdown()
    except Exception as e:
        logging.error(f"❌ Хатогӣ дар интизори савдо: {e}")
        mt5.shutdown()


def main():
    while True:
        best_symbol =  analyze_symbols()
        conclusion = trading_decision(best_symbol)

        if conclusion['decision'] == True:
            open_order_buy(1000000,best_symbol)
        elif conclusion['decision'] == False:
            open_order_sell(1000000, best_symbol)
        wait_for_trade_completion()
        time.sleep(5)
        

if __name__ == "__main__":
    print("Пайвастшавӣ ба Exness...".center(50, '='))
    main()
