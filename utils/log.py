import sys, time, functools

# -----------------------------------------------------------------------------
# Logging utilities
# -----------------------------------------------------------------------------
_LEVELS = {"INFO": 1, "WARN": 2, "ERROR": 3}
_VERBOSITY = _LEVELS["INFO"]

def set_verbosity(level: str = "INFO"):
    """
    Set global verbosity level. Options: INFO, WARN, ERROR.
    Messages below this level are suppressed.
    """
    global _VERBOSITY
    level = level.upper()
    if level not in _LEVELS:
        raise ValueError(f"Invalid verbosity level: {level}")
    _VERBOSITY = _LEVELS[level]

def timestamp() -> str:
    """Human-readable current time."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def _emit(level: str, msg: str, *args):
    """Internal emitter."""
    if _LEVELS[level] < _VERBOSITY:
        return
    if args:
        msg = msg % args
    sys.stdout.write(f"[{timestamp()}] {level}: {msg}\n")
    sys.stdout.flush()

def log(msg: str, *args):
    """Standard informational log message."""
    _emit("INFO", msg, *args)

def warn(msg: str, *args):
    """Warning-level log message."""
    _emit("WARN", msg, *args)

def fail(msg: str, code: int = 1):
    """Logs an error and exits."""
    _emit("ERROR", msg)
    sys.exit(code)

def ensure(cond: bool, msg: str):
    """Raise fatal error if condition is false."""
    if not cond:
        fail(msg)

# -----------------------------------------------------------------------------
# Timing decorator
# -----------------------------------------------------------------------------
def timeit(label: str = None):
    """
    Decorator to log runtime of a function.
    Example:
        @timeit("export_nlcd")
        def run_export(...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*a, **kw):
            start = time.time()
            try:
                return func(*a, **kw)
            finally:
                elapsed = time.time() - start
                name = label or func.__name__
                _emit("INFO", f"{name} completed in {elapsed:.2f}s")
        return wrapper
    return decorator
