"""MRST ``faceAverage`` public entry point."""

from pymrst.modules.ad_core.backends.diagonal.operators import face_average

faceAverage = face_average

__all__ = ["face_average", "faceAverage"]
