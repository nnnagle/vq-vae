import sys, time

def timestamp() -> str:
    """Human-readable current time."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def log(msg: str, *args):
    """Prints timestamped log messages to stdout."""
    if args:
        msg = msg % args
    sys.stdout.write(f"[{timestamp()}] {msg}\n")
    sys.stdout.flush()

def fail(msg: str, code: int = 1):
    """Logs an error and exits."""
    log("ERROR: %s", msg)
    sys.exit(code)

def ensure(cond: bool, msg: str):
    """Raise fatal error if condition is false."""
    if not cond:
        fail(msg)
