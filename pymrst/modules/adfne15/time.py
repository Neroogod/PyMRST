"""Native port of ADFNE15 ``Time.m``."""
from datetime import datetime
def time_string(formatted=False, *, now=None):
    value=datetime.now() if now is None else now
    return value.strftime("%Y-%m-%d,%H-%M-%S-%f")[:-3] if formatted else value.strftime("%Y%m%d%H%M%S")
