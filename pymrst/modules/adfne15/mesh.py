"""Parity owner for ADFNE15 ``Mesh.m``, unavailable in the upstream release."""
class ADFNEMeshUnavailableError(NotImplementedError): pass
def mesh(lines, distance, preprocess=False, model="r"):
    raise ADFNEMeshUnavailableError("not included at this time; see the ADFNE 1.5 book preface")
