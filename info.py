import MetaTrader5 as mt5


# --------------------------Exness System-----------------------------
order_buy = [0, 0, 0, 0, 0, 0, 0, 0, 0]
order_sel = [0, 0, 0, 0, 0, 0, 0, 0, 0]

login =   'your login'
server =  "Your Server"
password ='your pass'

topics = ['XAUUSDm', 'EURUSDm', 'GBPUSDm', 'BTCUSDm', 'USDJPYm', 'USOILm', 'AAPLm']

max_spreads = {
    'XAUUSDm': 200, 
    'EURUSDm': 5,
    'GBPUSDm': 5,
    'BTCUSDm': 1500,
    'USDJPYm': 8,
    'USOILm': 15,
    'AAPLm': 5
}
min_volumes = {
    'XAUUSDm': 1000,
    'EURUSDm': 5000,
    'GBPUSDm': 5000,
    'BTCUSDm': 50,
    'USDJPYm': 5000,
    'USOILm': 2000,
    'AAPLm': 1000
}
priority_factors = {
    'XAUUSDm': 1.5,  
    'EURUSDm': 1.0,
    'GBPUSDm': 1.0,
    'BTCUSDm': 1.8,
    'USDJPYm': 1.0,
    'USOILm': 1.3,
    'AAPLm': 1.2
}
TIME_ZONES = {
    'useful': {'09', '10', '11', '13', '14', '15', '17', '18', '19', '20', '21'},
    'neutral': {'05', '06', '07', '08', '12', '16', '22', '23'},
    'dangerous': {'00', '01', '02', '03', '04'}
}
# -----------------------------------------------------------------------


def initialize_mt5():
    try:
        if not mt5.initialize(login=login, server=server, password=password):
            print(f"❌ Хатоги пайвастшавии: {mt5.last_error()}")
            return False
        print("✅ MT5 Бо муваффақият пайваст шуд")
        return True
    except Exception as e:
        print(f"❌ Баромад аз система ҳангоми пайвастшавӣ: {e}")
        return False
