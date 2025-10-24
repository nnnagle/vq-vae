import sys

def fail(msg: str, code: int = 1):
    log("ERROR: %s", msg)
    sys.exit(code)
