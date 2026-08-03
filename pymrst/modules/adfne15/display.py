"""Native port of ADFNE15 ``Display.m``."""
def display(*values, stream=None):
    text=" ".join(str(value) for value in values)
    if stream is not None: print(text,file=stream)
    return text
