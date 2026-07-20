"""MRST-shaped `mpsaw/statefunctions` namespace."""

from pymrst.modules.mpsaw import mpsaw_isotropic_stress, mpsaw_strain

strain = mpsaw_strain
stress = mpsaw_isotropic_stress

__all__ = ["mpsaw_isotropic_stress", "mpsaw_strain", "strain", "stress"]
