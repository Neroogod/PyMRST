"""Native port of ADFNE15 ``Summary.m``."""
def summary(rows, *, stream=None):
    text="\n".join(f"{str(row[0]):>40}: {row[1]}" for row in rows)
    if stream is not None: print(text,file=stream)
    return text
