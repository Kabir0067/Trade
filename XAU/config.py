"""
═══════════════════════════════════════════════════════════════════════════
  CENTRAL CONFIGURATION  —  Gold Signal Bot (XAUUSD, 10-30 min scalping)
═══════════════════════════════════════════════════════════════════════════

Secrets live ONLY in the `.env` file (next to this file) — there are NO
secret values hard-coded here.

Real OS environment variables take precedence over `.env` (so you can override
on a server without editing the file). No third-party library is required.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path):
    """Minimal .env loader: KEY=VALUE per line, '#' comments. Does NOT override
    variables already present in the real environment."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, val)
    except OSError:
        pass


_load_dotenv(os.path.join(_HERE, ".env"))


def _env(name, default, cast=str):
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return cast(val)
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────────────────
#  SECRETS  (from .env — no defaults baked into source)
# ─────────────────────────────────────────────────────────────────────────
TG_BOT_TOKEN     = _env("TG_BOT_TOKEN",     "")
TG_ADMIN_CHAT_ID = _env("TG_ADMIN_CHAT_ID", "")

# accept the canonical MT5_* names, falling back to legacy EXNESS_* names
MT5_LOGIN    = _env("MT5_LOGIN",    _env("EXNESS_LOGIN", 0, int), int)
MT5_PASSWORD = _env("MT5_PASSWORD", _env("EXNESS_PASSWORD", ""))   # empty => attach to logged-in terminal
MT5_SERVER   = _env("MT5_SERVER",   _env("EXNESS_SERVER", ""))
# Full path to terminal64.exe (optional). If set, the bot can AUTO-LAUNCH a
# closed/crashed terminal AND hard-restart a HUNG one — set it in .env on the
# server, e.g. MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_PATH     = _env("MT5_PATH", "")
# Auto-recover a HUNG terminal: if the feed stays stale this many seconds while the
# market SHOULD be open, kill terminal64.exe and relaunch it (needs MT5_PATH).
MT5_STALE_RESTART_SEC    = _env("MT5_STALE_RESTART_SEC", 300, int)
MT5_RESTART_COOLDOWN_SEC = _env("MT5_RESTART_COOLDOWN_SEC", 600, int)   # min gap between restarts

# ─────────────────────────────────────────────────────────────────────────
#  SYMBOL  (not a secret)
# ─────────────────────────────────────────────────────────────────────────
SYMBOL            = "XAUUSDm"

# ─────────────────────────────────────────────────────────────────────────
#  MONEY MANAGEMENT  ($250 of balance -> 0.01 lot, hard cap 10 lots)
# ─────────────────────────────────────────────────────────────────────────
USD_PER_001_LOT = _env("USD_PER_001_LOT", 250.0, float)
MAX_LOT         = _env("MAX_LOT",         10.0,  float)
MIN_LOT         = 0.01

# Reject execution when the live spread blows out (news / market open).
# In price terms: gold normal spread ~$0.28; this blocks abnormal widening.
MAX_SPREAD_USD  = _env("MAX_SPREAD_USD", 1.00, float)

# ─────────────────────────────────────────────────────────────────────────
#  EXECUTION GUARDRAILS  (ported from the BTC bot, tuned for GOLD's $ scale)
# ─────────────────────────────────────────────────────────────────────────
# Don't execute a stale tap: if the signal that produced this button is older
# than this, the setup has moved on — refuse the fill instead of chasing it.
MAX_SIGNAL_EXECUTION_AGE_SEC = _env("MAX_SIGNAL_EXECUTION_AGE_SEC", 180, int)
# Anti-chasing on the FILL: reject only ADVERSE drift (price moved against you —
# a worse entry than the signal) beyond max(spread, risk × this). Favourable drift
# (a better price) is allowed. In R units. NOTE: gold's scalp risk is tiny (~$3-4)
# so this must be looser than BTC's 0.12 — 0.5 allows ~$2 of adverse drift on a
# typical gold stop, covering normal manual-tap latency without chasing.
MAX_ENTRY_DRIFT_R            = _env("MAX_ENTRY_DRIFT_R", 0.5, float)
# Slippage budget (USD) converted to broker deviation points per symbol, so we
# cap real slippage instead of sending a blind hardcoded deviation. Gold moves in
# dollars — $1.0 ceiling mirrors MAX_SPREAD_USD.
MAX_SLIPPAGE_USD             = _env("MAX_SLIPPAGE_USD", 1.0, float)
MAX_SLIPPAGE_RISK_PCT        = _env("MAX_SLIPPAGE_RISK_PCT", 0.06, float)
MIN_DEVIATION_POINTS         = _env("MIN_DEVIATION_POINTS", 20, int)
# Wide cap used on EXITS (close / partial) to GUARANTEE the fill. 1000 pts is
# ~$10 on 2-digit gold / ~$1 on 3-digit — generous enough to always close.
MAX_DEVIATION_POINTS         = _env("MAX_DEVIATION_POINTS", 1000, int)
# Pre-flight order_check (margin / stops / volume) before going live. Pure safety
# — it NEVER reduces signals, only blocks an order the broker would reject anyway.
ENABLE_ORDER_CHECK           = _env("ENABLE_ORDER_CHECK", "1").lower() not in ("0", "false", "no", "off")

# ─────────────────────────────────────────────────────────────────────────
#  ANALYSIS LOOP TIMING
# ─────────────────────────────────────────────────────────────────────────
ANALYSIS_INTERVAL     = _env("ANALYSIS_INTERVAL",     10,  int)   # seconds between cycles
SIGNAL_COOLDOWN       = _env("SIGNAL_COOLDOWN",       120, int)   # min seconds between ANY two alerts
MARKET_CHECK_INTERVAL = _env("MARKET_CHECK_INTERVAL", 300, int)   # wait when market closed
MAX_DATA_AGE_SEC      = _env("MAX_DATA_AGE_SEC",      180, int)   # M5 candle staleness guard (3 min)

# ─────────────────────────────────────────────────────────────────────────
#  SIGNAL ENGINE  —  tuned for 10-30 minute trades
# ─────────────────────────────────────────────────────────────────────────
TF_WEIGHTS = {           # must sum to 100; execution TFs dominate
    "D1":  4,
    "H4":  8,
    "H1":  18,
    "M15": 30,           # base / anchor
    "M5":  25,           # primary trigger TF
    "M1":  15,
}
ANCHOR_TFS         = ["M15", "M5"]   # short-term anchors (NOT H4/H1)
ANCHOR_MIN_SCORE   = _env("ANCHOR_MIN_SCORE",   22, int)
MIN_TF_AGREE       = _env("MIN_TF_AGREE",        4, int)
MIN_CONFIDENCE     = _env("MIN_CONFIDENCE",     65, int)
TF_AGREE_MIN_SCORE = _env("TF_AGREE_MIN_SCORE", 22, int)
CONF_SCALE         = _env("CONF_SCALE",       58.0, float)

# ─────────────────────────────────────────────────────────────────────────
#  SL / TP  —  scalp-sized (reachable inside 10-40 min)
# ─────────────────────────────────────────────────────────────────────────
SL_ATR_MULT_MAX = _env("SL_ATR_MULT_MAX", 1.3, float)
SL_ATR_MULT_MIN = _env("SL_ATR_MULT_MIN", 0.7, float)
TP_ATR_MULT_MAX = _env("TP_ATR_MULT_MAX", 2.6, float)
MIN_RR          = _env("MIN_RR",          1.3, float)
TARGET_RR       = _env("TARGET_RR",       1.6, float)

NUM_BARS     = _env("NUM_BARS", 320, int)
ATR_FALLBACK = 2.0

# Anti-chasing overextension veto: stand aside if price is already stretched more
# than this many σ from its 20-bar mean (z-score) on the execution TF — avoids
# "buying the top / selling the bottom". Set very high (e.g. 99) to disable.
OVEREXT_Z = _env("OVEREXT_Z", 2.5, float)

# ─────────────────────────────────────────────────────────────────────────
#  SIGNAL SCORECARD  (forward paper-test on the REAL market — no money risk)
# ─────────────────────────────────────────────────────────────────────────
# Every signal the engine fires is recorded and then graded against the real M1
# path (did price hit TP or SL first?). This is how we MEASURE quality without a
# historical backtest. Purely additive; never executes a trade.
SCORECARD_ENABLED   = _env("SCORECARD_ENABLED", "1").lower() not in ("0", "false", "no", "off")
SCORECARD_TIMEOUT_MIN = _env("SCORECARD_TIMEOUT_MIN", 60, int)   # close paper trade after N min
SCORECARD_COST_R      = _env("SCORECARD_COST_R", 0.05, float)    # spread/commission haircut (in R)
# Independent of the Telegram cooldown: don't log the *same* signal zone twice.
SCORECARD_DEDUPE_MIN  = _env("SCORECARD_DEDUPE_MIN", 15, int)

# ─────────────────────────────────────────────────────────────────────────
#  SAFETY GATES  (principled overlays — NOT day-fitted)
# ─────────────────────────────────────────────────────────────────────────
# Anti-over-trading: one signal per setup. Suppress a repeat alert in the SAME
# direction within this window (matches the 10-30 min trade horizon — you don't
# keep re-entering the same move). Set 0 to disable.
SAME_DIR_COOLDOWN_MIN = _env("SAME_DIR_COOLDOWN_MIN", 15, int)

# ─────────────────────────────────────────────────────────────────────────
#  RISK SIZING & TRADE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────
MAGIC = _env("MAGIC", 234000, int)                   # our orders' magic — DISTINCT per bot
# ^ XAU=234000, BTC=234001 (set in each .env). Two bots on ONE account MUST use
# different magics so a kill/close-all of one never touches the other's trades.
# Legacy magics this bot ALSO manages for its OWN symbol (XAU keeps 234000, so it
# has no orphans — empty by default). New orders always use MAGIC.
LEGACY_MAGICS = [int(x) for x in _env("LEGACY_MAGICS", "").replace(" ", "").split(",")
                 if x.lstrip("-").isdigit()]
CONTRACT_SIZE = _env("CONTRACT_SIZE", 100.0, float)  # gold: 100 oz / lot

# ATR/risk-based position sizing: lot risks RISK_PCT of balance over the SL distance.
RISK_PCT = _env("RISK_PCT", 1.0, float)

# Trade management (applied to live positions every loop cycle)
ENABLE_TRADE_MGMT    = _env("ENABLE_TRADE_MGMT", "1").lower() not in ("0", "false", "no", "off")
BREAKEVEN_AT_R       = _env("BREAKEVEN_AT_R", 1.0, float)    # move SL to entry at +1R
BREAKEVEN_BUFFER_USD = _env("BREAKEVEN_BUFFER_USD", 0.15, float)
PARTIAL_TP_R         = _env("PARTIAL_TP_R", 1.0, float)      # take partial at +1R
PARTIAL_TP_PCT       = _env("PARTIAL_TP_PCT", 50.0, float)   # close this % at partial
TRAIL_START_R        = _env("TRAIL_START_R", 1.5, float)     # start trailing at +1.5R
TRAIL_R              = _env("TRAIL_R", 1.0, float)           # keep SL this many R behind price
TRAIL_SMOOTH_BARS    = _env("TRAIL_SMOOTH_BARS", 5, int)     # EMA smoothing for trail (anti-whipsaw)

# Time-stop: an intraday momentum signal that hasn't reached TP or SL after N
# minutes has lost its thesis — close it fast IF it is at least marginally green
# (lock the small profit before it reverts). 0 disables. Matches the 10-40 min
# trade horizon. (Losers are left to the SL; this only harvests stale winners.)
TIME_STOP_MIN            = _env("TIME_STOP_MIN", 40, int)
TIME_STOP_MIN_PROFIT_USD = _env("TIME_STOP_MIN_PROFIT_USD", 0.10, float)
# NOTE: the bot manages ONLY its own orders (magic 234000) — manually-opened
# trades are deliberately left untouched (owner decision).


# ─────────────────────────────────────────────────────────────────────────
#  ORDER BLOCKS  (SMC institutional demand/supply zones — additive confluence)
# ─────────────────────────────────────────────────────────────────────────
# A bonus added ON TOP of the existing SMC score when price reacts from a fresh
# order block in the signal's direction. Existing setups are unchanged; only a
# genuine institutional-zone reaction is rewarded — sharper, not louder.
# Set OB_ENABLED=0 in .env to disable instantly if it does not help.
OB_ENABLED      = _env("OB_ENABLED", "1").lower() not in ("0", "false", "no", "off")
OB_DISPLACE_ATR = _env("OB_DISPLACE_ATR", 1.0, float)  # displacement body strength (× ATR)
OB_MAX_AGE      = _env("OB_MAX_AGE", 30, int)          # bars an unmitigated zone stays valid
OB_SCORE        = _env("OB_SCORE", 8, int)             # confluence bonus added to SMC (capped)


# ─────────────────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────────────────
def validate():
    """Return (errors, warnings). Errors block startup; warnings degrade gracefully."""
    errors, warnings = [], []
    if not TG_BOT_TOKEN or ":" not in TG_BOT_TOKEN:
        errors.append("TG_BOT_TOKEN is missing/invalid (set it in .env)")
    if not TG_ADMIN_CHAT_ID:
        errors.append("TG_ADMIN_CHAT_ID is missing (set it in .env)")
    if not MT5_LOGIN or not MT5_SERVER:
        warnings.append("MT5_LOGIN/MT5_SERVER not set — will rely on the already "
                        "logged-in terminal session")
    if not MT5_PASSWORD:
        warnings.append("MT5_PASSWORD not set — attaching to the logged-in terminal "
                        "(fine if MT5 is already logged in)")
    return errors, warnings
