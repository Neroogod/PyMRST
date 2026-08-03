"""Importable subprocess targets used by the execution-boundary tests."""

import time


def echo(parameters):
    return {"value": parameters["value"]}


def delay(parameters):
    time.sleep(float(parameters.get("seconds", 1)))
    return {"finished": True}


def fail(parameters):
    raise RuntimeError(parameters.get("message", "failure"))
