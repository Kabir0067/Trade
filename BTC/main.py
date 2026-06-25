"""
═══════════════════════════════════════════════════════════════════════════
  BITCOIN SIGNAL BOT  —  Telegram front-end + 24/7 Crypto analysis loop
═══════════════════════════════════════════════════════════════════════════
"""

import sys
import math
import time
import threading

try:                                            # emoji-safe console on Windows cp1251
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass 

import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, BotCommand
import MetaTrader5 as mt5

import config
import mt5_clent as mc
import trade_manager
import signal_tracker
from analysis import MultiTimeframeSignalEngine, clamp

# The bot is created in _build_bot() (lazy) so importing this module never needs
# a token — no secret default is baked in anywhere.
bot = None

_last_signal_time = 0.0
_stale_since = 0.0                 # wall-clock when the feed first went stale (0 = fresh)
_state_lock = threading.Lock()


# ───────────────────────────────────────────────────────────────────────────
#  MONEY MANAGEMENT
# ───────────────────────────────────────────────────────────────────────────
def get_dynamic_base_lot():
    bal = mc.get_balance()
    if bal <= 0:
        return config.MIN_LOT
    lot = math.floor(bal / config.USD_PER_001_LOT) * 0.01
    return clamp(round(lot, 2), config.MIN_LOT, config.MAX_LOT)


def get_risk_based_lot(sl_distance):
    """ATR/risk-based size: risk RISK_PCT of balance over the SL (ATR-derived) distance.

    Uses the broker's real per-point value (tick_value/tick_size) so the dollar
    risk is exact, not an assumed CONTRACT_SIZE."""
    bal = mc.get_balance()
    if bal <= 0 or sl_distance <= 0:
        return config.MIN_LOT
    risk_usd = bal * config.RISK_PCT / 100.0
    per_unit = mc.value_per_price_unit()
    if per_unit <= 0:
        per_unit = config.CONTRACT_SIZE
    lot = risk_usd / (sl_distance * per_unit)
    return clamp(round(lot, 2), config.MIN_LOT, config.MAX_LOT)


def _button_lots(base, score=0):
    """Risk-scaled lot choices. Higher score unlocks gentle add-on sizes."""
    if score >= 90:
        multipliers = (0.5, 1.0, 1.5)
    elif score >= 80:
        multipliers = (0.5, 1.0)
    else:
        multipliers = (1.0,)
    return sorted({clamp(round(base * m, 2), config.MIN_LOT, config.MAX_LOT)
                   for m in multipliers})



def bot_commands():
    return [
        BotCommand("start", "🚀 Запустить бота и показать информацию"),
        BotCommand("buy",   "🟢 Быстрая ПОКУПКА (без SL/TP, 1-6×)"),
        BotCommand("sell",  "🔴 Быстрая ПРОДАЖА (без SL/TP, 1-6×)"),
        BotCommand("tp",    "🎯 Установить Take Profit в долларах ($)"),
        BotCommand("sl",    "🛑 Установить Stop Loss в долларах ($)"),
        BotCommand("time",  "🕐 Время/сессия (брокер vs OS)"),
        BotCommand("stats", "📊 Статистика сигналов (Crypto forward-тест)"),
    ]

# ───────────────────────────────────────────────────────────────────────────
#  SIGNAL MESSAGE
# ───────────────────────────────────────────────────────────────────────────
def send_signal_to_admin(s):
    signal = s['signal']
    score = s.get('score', 0)
    stars = s.get('stars', '⭐⭐⭐')
    price = s['close_price']
    sl, tp = s['sl'], s['tp']
    risk, reward = s.get('risk_pts', 0), s.get('reward_pts', 0)
    brk = s.get('breakdown', {})
    tf_text = "\n".join(f"  • {t}" for t in s.get('tf_details', [])[:6])
    tf_agree = s.get('tf_agree_buy', 0) if signal == 'BUY' else s.get('tf_agree_sell', 0)

    if signal == 'BUY':
        action_char, emoji = 'B', "🚀🟢 ПОКУПКА (BUY)"
    else:
        action_char, emoji = 'S', "🩸🔴 ПРОДАЖА (SELL)"

    msg = (
        f"{'━' * 30}\n"
        f"⚡ *КРИПТО СИГНАЛ (BITCOIN)* ⚡\n"
        f"{'━' * 30}\n\n"
        f"{emoji}\n"
        f"{stars}  *Уверенность: {score}%*\n\n"
        f"💲 *Цена:* `{price:.2f}`\n"
        f"🛑 *SL:* `{sl:.2f}` ({risk:.1f} pts)\n"
        f"🎯 *TP:* `{tp:.2f}` ({reward:.1f} pts)\n"
        f"📐 *R:R =* `1:{reward / max(risk, 0.1):.1f}`\n\n"
        f"📊 *Анализ:*\n"
        f"  🔹 SMC: `{brk.get('smc', 0):+.1f}`\n"
        f"  🔹 Momentum: `{brk.get('momentum', 0):+.1f}`\n"
        f"  🔹 Reversion: `{brk.get('reversion', 0):+.1f}`\n"
        f"  🔹 Confirm: `{brk.get('confirm', 0):+.1f}`\n"
        f"  🔹 Volume/Flow: `{brk.get('volume', 0):+.1f}`\n"
        f"  🎯 *Главный фактор: {s.get('dominant', '?').upper()}*\n\n"
        f"🌐 *Session:* {s.get('session', '?')} (×{s.get('session_power', 1.0)})\n"
        f"🕯 *Fresh:* {s.get('candle_fresh', '—')}\n"
        f"🔑 *Anchor {s.get('anchor', '?')}* → {s.get('anchor_dir', '?')}\n"
        f"⏱ *{tf_agree}/6 TF agree:*\n{tf_text}\n\n"
        f"{'━' * 30}\n"
        f"🎯 *с SL/TP*   ·   ⚡ *без SL/TP*   (1×–N× по уверенности)\n"
        f"Выберите объём (макс {config.MAX_LOT} лот):"
    )

    base_lot = get_dynamic_base_lot()
    signal_ts = int(time.time())
    # Button COUNT scales with confidence (user tiers): 65-69%→1, 70-79%→2,
    # 80-89%→3, 90%+→5. Higher conviction unlocks bigger add-on sizes. Every 🎯
    # button carries the SAME SL/TP — only the lot multiplier differs.
    if score >= 90:
        num_btn = 5
    elif score >= 80:
        num_btn = 3
    elif score >= 70:
        num_btn = 2
    else:
        num_btn = 1
    lots = [clamp(round(base_lot * n, 2), config.MIN_LOT, config.MAX_LOT)
            for n in range(1, num_btn + 1)]
    markup = InlineKeyboardMarkup()
    # Row 1 — WITH SL/TP (auto). risk/reward DISTANCES re-anchor to the live fill;
    # price+ts let execution reject a stale / over-drifted tap.
    row_sltp = []
    for n, lot in enumerate(lots, start=1):
        cb = f"T_{action_char}_{lot}_{risk:.2f}_{reward:.2f}_{price:.2f}_{signal_ts}"
        row_sltp.append(InlineKeyboardButton(f"🎯{n}× {lot}", callback_data=cb))
    markup.row(*row_sltp)
    # Row 2 — NO SL/TP, market order in the SAME direction & SAME count.
    row_raw = [InlineKeyboardButton(f"⚡{n}× {lots[n - 1]}", callback_data=f"O_{action_char}_{n}")
               for n in range(1, num_btn + 1)]
    markup.row(*row_raw)

    # Retry on transient Telegram/network blips so a generated signal is NEVER lost
    # to a dropped connection. Returns True only on REAL delivery — the caller starts
    # the dedupe/cooldown clocks ONLY then, so a failed send is re-attempted next cycle.
    for attempt in range(1, 4):
        try:
            bot.send_message(config.TG_ADMIN_CHAT_ID, msg, reply_markup=markup,
                             parse_mode="Markdown")
            return True
        except Exception as e:
            print(f"❌ Telegram send failed (attempt {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(2)
    print(f"❌ {signal} signal NOT delivered after 3 tries — will retry next cycle")
    return False


# ───────────────────────────────────────────────────────────────────────────
#  24/7 ANALYSIS LOOP
# ───────────────────────────────────────────────────────────────────────────
def background_analysis_loop():
    global _last_signal_time, _stale_since
    print("🤖 Bitcoin Signal Bot — 24/7 Crypto institutional analysis...")

    engine = MultiTimeframeSignalEngine(
        mc.GetRealBtcData(timeframe=mt5.TIMEFRAME_M5, num_bars=config.NUM_BARS))

    while True:
        try:
            if not mc.connect_to_mt5():
                print("⏳ MT5 connection failed, retry in 30s...")
                time.sleep(30)
                continue
            mc.resolve_symbol()

            rep = mc.isolation_report()        # one-time: prove which terminal/account we bound to
            if rep:
                try:
                    bot.send_message(
                        config.TG_ADMIN_CHAT_ID,
                        f"🔗 *{config.SYMBOL}* запущен\n"
                        f"Терминал: `{rep.get('terminal')}`\n"
                        f"Счёт: `{rep.get('login')}@{rep.get('server')}`\n"
                        f"Изоляция: {'OK ✅' if rep.get('isolation_ok') else 'ПРОВЕРЬ ⚠️'}",
                        parse_mode="Markdown")
                except Exception:
                    pass

            try:
                if not mc.is_market_open():
                    if mc.market_should_be_open() and config.MT5_PATH:
                        now_w = time.time()
                        if _stale_since == 0.0:
                            _stale_since = now_w
                        elif (now_w - _stale_since) >= config.MT5_STALE_RESTART_SEC \
                                and mc.seconds_since_terminal_restart() >= config.MT5_RESTART_COOLDOWN_SEC:
                            print("🔧 Feed stale — auto-restarting MT5 terminal...")
                            print("✅ MT5 terminal restarted" if mc.restart_terminal()
                                  else "❌ MT5 terminal restart failed (check MT5_PATH)")
                            _stale_since = 0.0
                    print("🛑 Crypto feed stale. Waiting...")
                    time.sleep(config.MARKET_CHECK_INTERVAL)
                    continue
            except ConnectionError as ce:
                print(f"📡 Connection issue: {ce}. Retry in 15s...")
                time.sleep(15)
                continue
            _stale_since = 0.0          # fresh feed reached -> reset the stale clock

            # manage open positions (break-even @1R, trailing, partial TP)
            try:
                for act in trade_manager.manage_positions(mc.SYMBOL):
                    print(f"⚙️  {act}")
            except Exception as e:
                print(f"⚠️ trade mgmt error: {e}")

            s = engine.generate_signal()

            signal_tracker.update(s)

            result = s.get('signal')

            if result in ('BUY', 'SELL'):
                now = time.time()
                with _state_lock:
                    spaced = (now - _last_signal_time) > config.SIGNAL_COOLDOWN
                if spaced:
                    print(f"🚨 SIGNAL: {result} {s.get('score')}% {s.get('stars', '')}")
                    # Commit the dedupe/cooldown clocks + throttle ONLY on REAL
                    # delivery — a signal dropped by a Telegram blip is left
                    # un-committed so the next cycle re-fires and re-sends it.
                    if send_signal_to_admin(s):
                        with _state_lock:
                            _last_signal_time = time.time()
                        engine.commit_pending()
                    else:
                        print(f"⚠️ {result} not delivered — kept live for next-cycle retry")
                else:
                    print(f"⏳ Cooldown active, skipping {result}")
            else:
                print(f"🕒 NEUTRAL: {s.get('reason', '')} | "
                      f"B:{s.get('buy_conf', 0)}% S:{s.get('sell_conf', 0)}%")

            time.sleep(config.ANALYSIS_INTERVAL)
        except Exception as e:
            print(f"❌ Loop error: {e}")
            time.sleep(10)


# ───────────────────────────────────────────────────────────────────────────
#  TELEGRAM HANDLERS
# ───────────────────────────────────────────────────────────────────────────
def send_welcome(message):
    bot.reply_to(
        message,
        "🏆 *Bitcoin Signal Bot — 24/7 Crypto Edition*\n\n"
        "Автоанализ BTCUSDm 24/7 для крипто-сделок\n"
        "• 6 таймфреймов (M1 → D1), M15/M5 — якорь\n"
        "• SMC + Order Flow (CVD) + VWAP + Volume Profile\n"
        "• Защита от крипто-спреда и Slippage\n"
        "• Исполнение по кнопке\n\n"
        "📊 /stats — forward-тест в реале (реальный рынок)\n\n"
        f"Лимит объёма: *{config.MAX_LOT} лот* | ${config.USD_PER_001_LOT} = 0.01 лот",
        parse_mode="Markdown")


def send_stats(message):
    try:
        report = signal_tracker.format_report()
    except Exception as e:
        report = f"⚠️ Scorecard error: {e}"
    try:
        bot.reply_to(message, report, parse_mode="Markdown")
    except Exception:
        bot.reply_to(message, report)


def send_tp_menu(message):
    markup = InlineKeyboardMarkup()
    row = []
    # Crypto specific dollar targets
    amounts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for amt in amounts:
        row.append(InlineKeyboardButton(f"${amt}", callback_data=f"SET_TP_{amt}"))
        if len(row) == 4:
            markup.row(*row)
            row = []
    if row:
        markup.row(*row)
    bot.reply_to(
        message,
        "🎯 *Take Profit — выберите прибыль в долларах*\n\n"
        "Бот выставит TP так, чтобы прибыль равнялась выбранной сумме.\n"
        "💡 Если прибыль уже достигнута — позиция закроется *сразу*.",
        reply_markup=markup, parse_mode="Markdown")


def send_sl_menu(message):
    markup = InlineKeyboardMarkup()
    row = []
    amounts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for amt in amounts:
        row.append(InlineKeyboardButton(f"${amt}", callback_data=f"SET_SL_{amt}"))
        if len(row) == 4:
            markup.row(*row)
            row = []
    if row:
        markup.row(*row)
    bot.reply_to(
        message,
        "🛑 *Stop Loss — выберите убыток в долларах*\n\n"
        "Бот выставит SL на выбранный убыток.\n"
        "💡 Если убыток уже превышен — позиция закроется *сразу*.",
        reply_markup=markup, parse_mode="Markdown")


def _clear_buttons(call):
    try:
        bot.edit_message_reply_markup(call.message.chat.id, call.message.message_id,
                                      reply_markup=None)
    except Exception:
        pass


def handle_trade_execution(call):
    # ONE order per signal message — claim it BEFORE doing anything (shared with the
    # ⚡ buttons via _open_guard, so a signal opens exactly one order whichever button
    # you tap). Released on failure (finally) and buttons cleared only on SUCCESS, so
    # a rejected open can still be retried.
    mid = getattr(call.message, "message_id", None)
    with _open_guard_lock:
        if mid in _open_guard:
            try:
                bot.answer_callback_query(call.id, "⏳ Уже выполнено (1 ордер на сигнал)")
            except Exception:
                pass
            return
        _open_guard.add(mid)
        if len(_open_guard) > 1000:
            _open_guard.clear()
    opened_ok = False
    try:
        parts = call.data.split('_')
        if len(parts) < 5:
            bot.answer_callback_query(call.id, "❌ Неверный запрос")
            return
        action_char = parts[1]
        lot_size = clamp(float(parts[2]), config.MIN_LOT, config.MAX_LOT)
        risk, reward = float(parts[3]), float(parts[4])
        signal_price = float(parts[5]) if len(parts) >= 6 else None
        signal_ts = int(float(parts[6])) if len(parts) >= 7 else None

        if signal_ts and time.time() - signal_ts > config.MAX_SIGNAL_EXECUTION_AGE_SEC:
            _clear_buttons(call)
            bot.answer_callback_query(call.id, "Signal expired", show_alert=True)
            return

        if not mc.connect_to_mt5():
            bot.answer_callback_query(call.id, "❌ MT5 connection failed!", show_alert=True)
            return
        mc.resolve_symbol()

        if mc.get_balance() <= 0:
            bot.answer_callback_query(call.id, "❌ Insufficient balance!", show_alert=True)
            return

        if action_char == 'B':
            order_type, action_text = mt5.ORDER_TYPE_BUY, "ПОКУПКА"
        else:
            order_type, action_text = mt5.ORDER_TYPE_SELL, "ПРОДАЖА"

        bot.answer_callback_query(call.id, f"⚡ {action_text} {lot_size} лот...")
        res = mc.open_order(order_type, lot_size, risk=risk, reward=reward,
                            signal_price=signal_price)

        if res.get("ok"):
            opened_ok = True
            _clear_buttons(call)                      # success -> remove buttons
            bot.send_message(
                call.message.chat.id,
                f"✅ *СДЕЛКА ОТКРЫТА!*\n\n"
                f"📌 Тип: *{action_text}*\n"
                f"📦 Объём: *{res['volume']} лот*\n"
                f"💲 Цена: `{res['price']:.2f}`\n"
                f"🛑 SL: `{res['sl']:.2f}`\n"
                f"🎯 TP: `{res['tp']:.2f}`\n"
                f"🎫 Ticket: `{res['ticket']}`",
                parse_mode="Markdown")
        else:
            bot.send_message(call.message.chat.id,
                             f"❌ Ошибка ордера: {res.get('error', '?')}")
    except Exception as e:
        print(f"❌ Callback error: {e}")
        try:
            bot.answer_callback_query(call.id, "❌ Internal error!", show_alert=True)
        except Exception:
            pass
    finally:
        if not opened_ok:
            with _open_guard_lock:
                _open_guard.discard(mid)              # failed -> allow a retry


def handle_tp_sl_setting(call):
    try:
        parts = call.data.split('_')
        action = parts[1] # 'TP' or 'SL'
        amount = float(parts[2])
        
        bot.answer_callback_query(call.id, f"⏳ Устанавливаем {action} на ${amount}...")
        _clear_buttons(call)               

        if not mc.connect_to_mt5():
            bot.send_message(call.message.chat.id, "❌ Ошибка: нет подключения к MT5")
            return
            
        mc.resolve_symbol()
        
        if action == "TP":
            res = mc.set_tp_in_usd(amount)
        else:
            res = mc.set_sl_in_usd(amount)
            
        if res.get("ok"):
            results = res.get("results", [])
            success_count = 0
            closed_count = 0
            for r in results:
                if r.get("retcode") == "CLOSED":
                    closed_count += 1
                elif r.get("retcode") == mt5.TRADE_RETCODE_DONE:
                    success_count += 1
                    
            msg = f"*{action} = ${amount}*\n\n"
            if success_count: 
                msg += f"🔹 Модифицировано позиций: `{success_count}`\n"
            if closed_count: 
                msg += f"⚡ Автоматически закрыто: `{closed_count}`\n"
            if not success_count and not closed_count: 
                msg += f"⚠️ Не изменено. Ответ брокера: `{results}`\n(Если retcode≠10009 — это причина отказа.)"
            
            bot.send_message(call.message.chat.id, msg, parse_mode="Markdown")
        else:
            bot.send_message(call.message.chat.id, f"❌ Ошибка: {res.get('error')}")
            
    except Exception as e:
        print(f"❌ Callback error (TP/SL): {e}")
        try:
            bot.answer_callback_query(call.id, "❌ Ошибка!", show_alert=True)
        except Exception:
            pass


def send_time_check(message):
    """PROOF of where the session clock comes from: the live broker tick, not the OS."""
    from datetime import datetime, timezone
    from analysis import get_session_info
    try:
        if not mc.connect_to_mt5():
            bot.reply_to(message, "❌ MT5 не подключён")
            return
        mc.resolve_symbol()
        os_utc = datetime.now(timezone.utc)
        offset = mc.get_server_utc_offset_hours()
        tick = mt5.symbol_info_tick(config.SYMBOL)
        broker_utc = mc.GetRealBtcData().broker_utc_now()
        timing = get_session_info(now=broker_utc)
        if tick and tick.time:
            age = os_utc.timestamp() - (tick.time - offset * 3600)
            src, epoch = f"✅ БРОКЕР (тик свежий, {age:.0f} с назад)", tick.time
        else:
            src, epoch = "⚠️ OS (тика нет — fallback)", "—"
        bot.reply_to(
            message,
            f"🕐 *Откуда берётся время сессии*\n\n"
            f"Источник: *{src}*\n"
            f"Брокер тик (epoch): `{epoch}`\n"
            f"Смещение брокера: `{offset:+d} ч`\n"
            f"Брокер UTC: `{broker_utc:%Y-%m-%d %H:%M:%S}`\n"
            f"OS UTC:     `{os_utc:%Y-%m-%d %H:%M:%S}`\n\n"
            f"🌐 Сессия сейчас: *{timing['session']}* (×{timing['session_power']})\n\n"
            f"_«Источник: БРОКЕР» ⇒ сессии идут от брокера, не от OS-часов._",
            parse_mode="Markdown")
    except Exception as e:
        bot.reply_to(message, f"⚠️ Ошибка: {e}")


# ───────────────────────────────────────────────────────────────────────────
#  FAST MANUAL ORDER  (instant market open — NO preview, NO SL/TP)
# ───────────────────────────────────────────────────────────────────────────
# One tap = ONE order: guard against double-taps / Telegram re-delivery opening
# several orders from the SAME button message.
_open_guard = set()
_open_guard_lock = threading.Lock()


def _open_menu(message, direction):
    """Show lot-multiplier buttons 1..6. Tapping one opens a market order INSTANTLY
    (no preview, no SL/TP). Lot = N × dynamic base lot (balance / $-per-0.01-lot)."""
    if not mc.connect_to_mt5():
        bot.reply_to(message, "❌ MT5 не подключён, попробуйте ещё раз.")
        return
    mc.resolve_symbol()
    base = get_dynamic_base_lot()
    char, title = ('B', "🟢 ПОКУПКА (BUY)") if direction == 'B' else ('S', "🔴 ПРОДАЖА (SELL)")
    markup = InlineKeyboardMarkup()
    row = []
    for n in range(1, 7):
        lot = clamp(round(base * n, 2), config.MIN_LOT, config.MAX_LOT)
        row.append(InlineKeyboardButton(f"{n}× = {lot}", callback_data=f"O_{char}_{n}"))
        if len(row) == 3:
            markup.row(*row)
            row = []
    if row:
        markup.row(*row)
    bot.reply_to(
        message,
        f"⚡ *БЫСТРЫЙ ОРДЕР — {title}*\n"
        f"_{config.SYMBOL} · без превью · без SL/TP · по рынку_\n\n"
        f"База: *{base}* лот (${config.USD_PER_001_LOT} = 0.01 лот).\n"
        f"Нажмите множитель — ордер откроется *сразу*:",
        reply_markup=markup, parse_mode="Markdown")


def send_buy_menu(message):
    _open_menu(message, 'B')


def send_sell_menu(message):
    _open_menu(message, 'S')


def handle_instant_open(call):
    """Tap a fast-order button -> open a market order INSTANTLY, no SL/TP."""
    # ONE order per button message — claim it BEFORE doing anything. Released on
    # failure (finally) so a failed tap doesn't dead-lock the buttons; buttons are
    # only cleared on SUCCESS, so a rejected open can still be retried.
    mid = getattr(call.message, "message_id", None)
    with _open_guard_lock:
        if mid in _open_guard:
            try:
                bot.answer_callback_query(call.id, "⏳ Уже выполнено (1 ордер на сообщение)")
            except Exception:
                pass
            return
        _open_guard.add(mid)
        if len(_open_guard) > 1000:
            _open_guard.clear()
    opened_ok = False
    try:
        parts = call.data.split('_')                 # O_B_3
        direction, mult = parts[1], int(parts[2])
        if not mc.connect_to_mt5():
            bot.answer_callback_query(call.id, "❌ MT5 не подключён!", show_alert=True)
            return
        mc.resolve_symbol()
        if mc.get_balance() <= 0:
            bot.answer_callback_query(call.id, "❌ Нет баланса!", show_alert=True)
            return
        base = get_dynamic_base_lot()
        lot = clamp(round(base * mult, 2), config.MIN_LOT, config.MAX_LOT)
        if direction == 'B':
            order_type, txt = mt5.ORDER_TYPE_BUY, "ПОКУПКА"
        else:
            order_type, txt = mt5.ORDER_TYPE_SELL, "ПРОДАЖА"
        bot.answer_callback_query(call.id, f"⚡ {txt} {lot} лот...")
        res = mc.open_order(order_type, lot, comment="FastManual")   # NO sl/tp/risk
        if res.get("ok"):
            opened_ok = True
            _clear_buttons(call)                      # success -> remove buttons
            bot.send_message(
                call.message.chat.id,
                f"✅ *{txt} ОТКРЫТА* (без SL/TP)\n\n"
                f"📦 Объём: *{res['volume']}* лот\n"
                f"💲 Цена: `{res['price']:.2f}`\n"
                f"🎫 Ticket: `{res['ticket']}`",
                parse_mode="Markdown")
        else:
            bot.send_message(call.message.chat.id, f"❌ Ошибка ордера: {res.get('error', '?')}")
    except Exception as e:
        print(f"❌ instant-open error: {e}")
        try:
            bot.answer_callback_query(call.id, "❌ Ошибка!", show_alert=True)
        except Exception:
            pass
    finally:
        if not opened_ok:
            with _open_guard_lock:
                _open_guard.discard(mid)              # failed -> allow a retry


# ───────────────────────────────────────────────────────────────────────────
#  STARTUP
# ───────────────────────────────────────────────────────────────────────────
def _build_bot():
    global bot
    bot = telebot.TeleBot(config.TG_BOT_TOKEN)
    # Silence telebot's TRANSIENT network noise: Telegram periodically closes the
    # long-poll connection (RemoteDisconnected / Connection aborted) — this is
    # normal and the poller auto-recovers, so the full traceback it dumps is just
    # alarming spam (and disk on a small VPS). We drop ONLY records whose text
    # carries a network-error signature; REAL problems (bad token, 401, 409
    # "two pollers same token", handler bugs) carry no such signature and stay
    # fully visible.
    _TELEBOT_QUIET = (
        "Break infinity polling",
        "Connection aborted", "RemoteDisconnected", "ConnectionError",
        "Connection reset", "ConnectionResetError", "Connection broken",
        "Read timed out", "ReadTimeout", "ReadTimeoutError",
        "Max retries exceeded", "NewConnectionError", "ProtocolError",
        "Failed to establish a new connection",
        "Temporary failure in name resolution",
    )

    def _quiet_telebot(record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(sig in msg for sig in _TELEBOT_QUIET)

    telebot.logger.addFilter(_quiet_telebot)
    bot.register_message_handler(send_welcome, commands=['start', 'help'])
    bot.register_message_handler(send_stats, commands=['stats', 'score'])
    bot.register_message_handler(send_tp_menu, commands=['tp'])
    bot.register_message_handler(send_sl_menu, commands=['sl'])
    bot.register_message_handler(send_buy_menu, commands=['buy'])
    bot.register_message_handler(send_sell_menu, commands=['sell'])
    bot.register_message_handler(send_time_check, commands=['time'])

    bot.register_callback_query_handler(handle_tp_sl_setting,
                                        func=lambda c: c.data.startswith('SET_TP_') or c.data.startswith('SET_SL_'))
    bot.register_callback_query_handler(handle_instant_open,
                                        func=lambda c: c.data.startswith('O_'))
    bot.register_callback_query_handler(handle_trade_execution,
                                        func=lambda c: c.data.startswith('T_'))
    
    try:
        bot.set_my_commands(bot_commands())
    except Exception as e:
        print(f"⚠️ Failed to set bot commands: {e}")
        
    return bot


def main():
    errors, warnings = config.validate()
    for w in warnings:
        print(f"⚠️  {w}")
    if errors:
        for e in errors:
            print(f"❌ {e}")
        print("\nFill in the missing values in your .env file and retry.")
        return 1

    _build_bot()
    threading.Thread(target=background_analysis_loop, daemon=True).start()
    print("🏆 Bitcoin Signal Bot LIVE — Crypto 24/7 Analysis Active")
    while True:
        try:
            bot.infinity_polling(timeout=20, long_polling_timeout=10)
        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C — остановка бота (clean shutdown)...")
            try:
                bot.stop_polling()
            except Exception:
                pass
            return 0
        except Exception as e:
            print(f"❌ Telegram polling crashed: {e}. Retry in 15s...")
            try:
                time.sleep(15)
            except KeyboardInterrupt:
                print("\n🛑 Ctrl+C — остановка бота (clean shutdown)...")
                return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:                  # Ctrl+C during startup/import — exit clean
        print("\n🛑 Остановлено.")
        raise SystemExit(0)
