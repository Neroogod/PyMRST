"""Parity owner for ADFNE15 ``Tensor.m``."""


class ADFNETensorUnavailableError(NotImplementedError):
    """Raised because the distributed ADFNE 1.5 Tensor owner is unavailable."""


def tensor(*_args, **_kwargs):
    raise ADFNETensorUnavailableError(
        "not included at this time; see the ADFNE 1.5 book preface")
