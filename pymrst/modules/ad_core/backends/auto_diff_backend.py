"""Base automatic-differentiation backend descriptor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoDiffBackend:
    """MRST-style AD backend configuration.

    The descriptor records backend capabilities. Numerical assembly is performed
    by PyMRST's native sparse ADI implementation unless `available` is false, in
    which case benchmark examples report the backend as unsupported.
    """

    name: str = "AutoDiffBackend"
    use_mex: bool = False
    row_major: bool = False
    deferred_assembly: bool = False
    available: bool = True
    reason: str = ""

    def label(self) -> str:
        """Human-readable benchmark label."""
        return self.name

    # -- MRST AutoDiffBackend.m dispatch, ported (Phase 2) ------------------
    #
    # Until now this class recorded *which* backend was asked for and supplied
    # none: `initVariablesAD` and `convertToAD` did not exist, so consumers
    # chose an AD implementation with `isinstance` checks at the call site and
    # the backend object was a label. These four methods are the interface that
    # makes it a backend - the thing that decides what an AD variable *is*.
    #
    # Additive: nothing called them before, and the base class dispatches to
    # the sparse ADI that PyMRST already used everywhere.

    def init_variables_ad(self, *values, **kwargs):
        """Seed primary variables as AD, in this backend's representation.

        The single method that makes a backend a backend rather than a name.
        """
        from pymrst.utils.adi import init_variables_adi
        return init_variables_adi(*values)

    initVariablesAD = init_variables_ad

    def convert_to_ad(self, v, sample):
        """Lift a plain array to AD with the same Jacobian layout as `sample`.

        Needed wherever a constant meets an AD expression: without it the
        constant carries no derivative blocks and the result silently loses
        the derivatives of whatever it was combined with.
        """
        from pymrst.modules.ad_core.utils.double_to_adi import double_to_adi
        return double_to_adi(v, sample)

    convertToAD = convert_to_ad

    def get_backend_description(self) -> str:
        return "Standard ADI (sparse jacobian)"

    getBackendDescription = get_backend_description

    def update_discrete_operators(self, model):
        """Hook for a backend that needs the discrete operators rebuilt in its
        own representation. The sparse path needs nothing."""
        return model

    updateDiscreteOperators = update_discrete_operators

    def with_options(self, **kwargs) -> "AutoDiffBackend":
        """Return a copy with updated options."""
        data = {
            "name": self.name,
            "use_mex": self.use_mex,
            "row_major": self.row_major,
            "deferred_assembly": self.deferred_assembly,
            "available": self.available,
            "reason": self.reason,
        }
        data.update(kwargs)
        return type(self)(**data)


__all__ = ["AutoDiffBackend"]
