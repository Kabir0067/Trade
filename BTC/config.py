"""
═══════════════════════════════════════════════════════════════════════════
  CENTRAL CONFIGURATION  —  Bitcoin Signal Bot (BTCUSD, 24/7 Crypto Scalping)
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
# Full path to terminal64.exe
MT5_PATH     = _env("MT5_PATH", "")
# Auto-recover a HUNG terminal
MT5_STALE_RESTART_SEC    = _env("MT5_STALE_RESTART_SEC", 300, int)
MT5_RESTART_COOLDOWN_SEC = _env("MT5_RESTART_COOLDOWN_SEC", 600, int)

# ─────────────────────────────────────────────────────────────────────────
#  SYMBOL  (Crypto 24/7)
# ─────────────────────────────────────────────────────────────────────────
SYMBOL            = "BTCUSDm"

# ─────────────────────────────────────────────────────────────────────────
#  MONEY MANAGEMENT  (Risk adapted for Crypto Volatility)
# ─────────────────────────────────────────────────────────────────────────
USD_PER_001_LOT = _env("USD_PER_001_LOT", 200.0, float)
MAX_LOT         = _env("MAX_LOT",         1.0,   float)
MIN_LOT         = 0.01

# Reject execution when the live spread blows out. 
# Bitcoin spread is naturally wider ($15-$40). Block if it goes beyond $50.
MAX_SPREAD_USD  = _env("MAX_SPREAD_USD", 50.00, float)
# Also reject when spread is too expensive versus the current ATR. This keeps
# the bot away from thin/liquidation microstructure where nominal spread looks
# "normal" but eats too much of the expected move.
MAX_SPREAD_ATR_PCT = _env("MAX_SPREAD_ATR_PCT", 0.18, float)

# Execution guardrails. MT5 deviation is in broker points, so mt5_clent converts
# these USD/ATR budgets into points per symbol before sending an order.
MAX_SIGNAL_EXECUTION_AGE_SEC = _env("MAX_SIGNAL_EXECUTION_AGE_SEC", 180, int)
# Anti-chasing on the FILL: reject only ADVERSE drift (a worse price than the
# signal) beyond max(spread, risk × this). BTC is fast — a $50-100 move between
# signal and a manual tap is normal — so allow up to 0.8 of the (large) stop
# before calling it chasing. Favourable drift (a better price) is always allowed.
MAX_ENTRY_DRIFT_R            = _env("MAX_ENTRY_DRIFT_R", 0.8, float)
MAX_SLIPPAGE_USD             = _env("MAX_SLIPPAGE_USD", 15.0, float)
MAX_SLIPPAGE_RISK_PCT        = _env("MAX_SLIPPAGE_RISK_PCT", 0.06, float)
MIN_DEVIATION_POINTS         = _env("MIN_DEVIATION_POINTS", 20, int)
MAX_DEVIATION_POINTS         = _env("MAX_DEVIATION_POINTS", 1500, int)
ENABLE_ORDER_CHECK           = _env("ENABLE_ORDER_CHECK", "1").lower() not in ("0", "false", "no", "off")

# ─────────────────────────────────────────────────────────────────────────
#  ANALYSIS LOOP TIMING
# ─────────────────────────────────────────────────────────────────────────
ANALYSIS_INTERVAL     = _env("ANALYSIS_INTERVAL",     10,  int)   # seconds between cycles
SIGNAL_COOLDOWN       = _env("SIGNAL_COOLDOWN",       300, int)   # Bitcoin needs longer cooldown to avoid fakeouts
MARKET_CHECK_INTERVAL = _env("MARKET_CHECK_INTERVAL", 300, int)   
MAX_DATA_AGE_SEC      = _env("MAX_DATA_AGE_SEC",      180, int)   # staleness guard

# ─────────────────────────────────────────────────────────────────────────
#  SIGNAL ENGINE  —  Tuned for Crypto Momentum & Mathematical Acceleration
# ─────────────────────────────────────────────────────────────────────────
TF_WEIGHTS = {           # execution TFs dominate
    "D1":  4,
    "H4":  8,
    "H1":  18,
    "M15": 30,           # base / anchor
    "M5":  25,           # primary trigger TF
    "M1":  15,
}
ANCHOR_TFS         = ["M15", "M5"] 
ANCHOR_MIN_SCORE   = _env("ANCHOR_MIN_SCORE",   22, int)
MIN_TF_AGREE       = _env("MIN_TF_AGREE",       4, int)
MIN_CONFIDENCE     = _env("MIN_CONFIDENCE",     65, int)
MIN_CONF_EDGE      = _env("MIN_CONF_EDGE",       8, int)
TF_AGREE_MIN_SCORE = _env("TF_AGREE_MIN_SCORE", 22, int)
CONF_SCALE         = _env("CONF_SCALE",       58.0, float)

# ─────────────────────────────────────────────────────────────────────────
#  SL / TP  —  Expanded for Crypto Wicks and Long Trends
# ─────────────────────────────────────────────────────────────────────────
SL_ATR_MULT_MAX = _env("SL_ATR_MULT_MAX", 2.2, float)
SL_ATR_MULT_MIN = _env("SL_ATR_MULT_MIN", 1.3, float)
TP_ATR_MULT_MAX = _env("TP_ATR_MULT_MAX", 4.0, float)
MIN_RR          = _env("MIN_RR",          1.5, float)
TARGET_RR       = _env("TARGET_RR",       2.0, float)

NUM_BARS     = _env("NUM_BARS", 320, int)
ATR_FALLBACK = 50.0  # Safe fallback for crypto if ATR fails

# Anti-chasing overextension veto: crypto trends mathematically accelerate faster 
# and further. Z-score tolerance raised to 3.5 standard deviations.
OVEREXT_Z = _env("OVEREXT_Z", 3.5, float)
MIN_TRIGGER_BODY_ATR = _env("MIN_TRIGGER_BODY_ATR", 0.08, float)
STRONG_SIGNAL_IGNORE_BODY_SCORE = _env("STRONG_SIGNAL_IGNORE_BODY_SCORE", 82, int)
# Anti-stop-hunt: refuse to ENTER when the M5 trigger candle's RANGE (high-low)
# exceeds this × ATR — a violent manipulation/news spike that tends to wick one
# way, take stops, then reverse. BTC can run $2000 in a minute; this stands aside
# from that chaos instead of chasing it. Set 99 to disable.
MAX_TRIGGER_RANGE_ATR = _env("MAX_TRIGGER_RANGE_ATR", 4.5, float)

# ─────────────────────────────────────────────────────────────────────────
#  SIGNAL SCORECARD  (Forward testing logic)
# ─────────────────────────────────────────────────────────────────────────
SCORECARD_ENABLED     = _env("SCORECARD_ENABLED", "1").lower() not in ("0", "false", "no", "off")
SCORECARD_TIMEOUT_MIN = _env("SCORECARD_TIMEOUT_MIN", 120, int)  # BTC trades take longer to develop
SCORECARD_COST_R      = _env("SCORECARD_COST_R", 0.10, float)    # Crypto spread is heavier
SCORECARD_DEDUPE_MIN  = _env("SCORECARD_DEDUPE_MIN", 30, int)

# ─────────────────────────────────────────────────────────────────────────
#  SAFETY GATES
# ─────────────────────────────────────────────────────────────────────────
SAME_DIR_COOLDOWN_MIN = _env("SAME_DIR_COOLDOWN_MIN", 30, int)

# ─────────────────────────────────────────────────────────────────────────
#  RISK SIZING & TRADE MANAGEMENT  [CRITICALLY ADJUSTED FOR CRYPTO]
# ─────────────────────────────────────────────────────────────────────────
MAGIC = _env("MAGIC", 234001, int)                   # our orders' magic — DISTINCT per bot
# ^ XAU=234000, BTC=234001. Two bots on ONE account MUST use different magics so
# a kill/close-all of one never touches the other's trades.
# Legacy magics this bot ALSO manages for its OWN symbol, so positions opened
# before the magic split aren't orphaned. New orders always use MAGIC; these only
# adopt pre-existing open trades. BTC default 234000 = the old shared magic.
LEGACY_MAGICS = [int(x) for x in _env("LEGACY_MAGICS", "234000").replace(" ", "").split(",")
                 if x.lstrip("-").isdigit()]
CONTRACT_SIZE = _env("CONTRACT_SIZE", 1.0, float)    # Bitcoin: 1 lot = 1 BTC (NOT 100!)

RISK_PCT = _env("RISK_PCT", 1.0, float)

# Trade management (applied to live positions every loop cycle)
ENABLE_TRADE_MGMT    = _env("ENABLE_TRADE_MGMT", "1").lower() not in ("0", "false", "no", "off")
BREAKEVEN_AT_R       = _env("BREAKEVEN_AT_R", 1.0, float)    # move SL to entry at +1R
# $15 buffer for BTC (avoids getting stopped out by a tiny tick)
BREAKEVEN_BUFFER_USD = _env("BREAKEVEN_BUFFER_USD", 15.0, float)
PARTIAL_TP_R         = _env("PARTIAL_TP_R", 1.5, float)      # Moved to 1.5R to let BTC breathe
PARTIAL_TP_PCT       = _env("PARTIAL_TP_PCT", 50.0, float)   # close this % at partial
TRAIL_START_R        = _env("TRAIL_START_R", 2.0, float)     # start trailing at +2.0R
TRAIL_R              = _env("TRAIL_R", 1.5, float)           # keep SL wider for crypto (1.5R)
TRAIL_SMOOTH_BARS    = _env("TRAIL_SMOOTH_BARS", 5, int)     

# Time-stop: Crypto needs more time to complete a move. Extended to 90 mins.
TIME_STOP_MIN            = _env("TIME_STOP_MIN", 90, int)
# Minimum profit to lock in stale trade: $10.0 for BTC (instead of 10 cents)
TIME_STOP_MIN_PROFIT_USD = _env("TIME_STOP_MIN_PROFIT_USD", 10.0, float)


# ─────────────────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────────────────
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
