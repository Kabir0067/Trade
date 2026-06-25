"""
═══════════════════════════════════════════════════════════════════════════
  MT5 TERMINAL ISOLATION SHIM   (loaded automatically via PYTHONPATH)
═══════════════════════════════════════════════════════════════════════════

WHY THIS EXISTS
---------------
Both bots call ``mt5.initialize()`` with NO path first (an "attach to whatever
terminal is open" call). When two bots run on the same machine that bare call
makes BOTH of them attach to the SAME single terminal → a hard conflict.

This shim does NOT touch a single line of either bot's code or trading logic.
Python imports ``sitecustomize`` automatically at interpreter start-up (before
``main.py`` ever runs). We simply wrap ``MetaTrader5.initialize`` so that, in a
child process launched by the master ``runserver.py``, EVERY initialize() call
— even the bare one — is pinned to THIS bot's own private terminal, in
``portable`` mode (its data folder lives next to its own terminal64.exe).

Activation is per-process and opt-in: it only does anything when the master sets
the ``BOT_MT5_TERMINAL`` environment variable for that child. With the variable
absent (e.g. you run a bot by hand the old way) this file is a complete no-op,
so it is safe to leave on PYTHONPATH permanently.

Result: BTC → its own terminal, XAU → its own terminal. 100% isolated, 0 edits
to the bots.
"""

import os


def _install_mt5_isolation():
    terminal = (os.environ.get("BOT_MT5_TERMINAL") or "").strip().strip('"')
    if not terminal:
        return  # not managed by the master launcher → behave exactly as before

    try:
        import MetaTrader5 as _mt5
    except Exception:
        return  # MetaTrader5 not importable in this process → nothing to pin

    if getattr(_mt5, "_bot_isolation_installed", False):
        return  # already wrapped (defensive against double import)

    _orig_initialize = _mt5.initialize

    def _initialize(*args, **kwargs):
        # Force this process onto ITS OWN terminal, in portable mode, no matter
        # how the caller invoked initialize() (bare, path=..., or full creds).
        # A positional path (the documented first unnamed arg) is discarded so
        # our pinned path always wins.
        kwargs.pop("path", None)
        kwargs["path"] = terminal
        kwargs["portable"] = True
        return _orig_initialize(**kwargs)

    try:
        _mt5.initialize = _initialize
        _mt5._bot_isolation_installed = True  # type: ignore[attr-defined]
    except Exception:
        # If the attribute is somehow read-only, fall back silently to the
        # original behaviour rather than crash the bot.
        pass


_install_mt5_isolation()
