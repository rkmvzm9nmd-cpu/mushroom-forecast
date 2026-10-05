"""Tiny run log. Everything here ends up in the public status.json, so never log
saved-spot names or coordinates."""
import time
import traceback

ENTRIES = []


def info(msg):
    _add("info", msg)


def warn(msg):
    _add("warn", msg)


def error(msg, exc=None):
    if exc is not None:
        msg = f"{msg}: {exc.__class__.__name__}: {exc}\n" + "".join(
            traceback.format_exception(exc)[-6:])
    _add("error", msg)


def _add(level, msg):
    stamp = time.strftime("%H:%M:%S")
    ENTRIES.append({"t": stamp, "level": level, "msg": msg})
    print(f"[{stamp}] {level.upper():5s} {msg}", flush=True)
