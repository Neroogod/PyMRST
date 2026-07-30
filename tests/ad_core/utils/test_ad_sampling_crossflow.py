from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal import GenericAD
from pymrst.modules.ad_core.utils import (
    cross_flow_mixture_density,
    double_to_adi,
    get_sample_ad,
    struct_prop_evaluated,
)
from pymrst.utils.adi import ADI


def test_double_to_adi_preserves_backend_and_zeroes_blocks():
    sparse_sample = ADI(np.array([1.0, 2.0]), [sp.eye(2, format="csr")])
    sparse = double_to_adi([4.0, 5.0, 6.0], sparse_sample)
    assert isinstance(sparse, ADI)
    assert np.array_equal(sparse.val, [4.0, 5.0, 6.0])
    assert sparse.jac[0].shape == (3, 2)
    assert sparse.jac[0].nnz == 0

    diagonal_sample = GenericAD(np.array([1.0, 2.0]), [sp.eye(2, format="csr")])
    diagonal = double_to_adi([7.0, 8.0, 9.0], diagonal_sample)
    assert isinstance(diagonal, GenericAD)
    assert diagonal.jac[0].shape == (3, 2)
    assert diagonal.jac[0].to_sparse().nnz == 0
    assert np.array_equal(double_to_adi([2.0], 1.0), [2.0])


def test_get_sample_ad_and_struct_prop_evaluated():
    ad = ADI(np.array([1.0]), [sp.csr_array([[1.0]])])
    sample, is_ad = get_sample_ad(np.array([2.0]), ad, np.array([3.0]))
    assert sample is ad and is_ad
    sample, is_ad = get_sample_ad(np.array([2.0]), np.array([3.0]))
    assert np.array_equal(sample, [2.0]) and not is_ad
    assert struct_prop_evaluated({"a": [1], "b": []}, "a")
    assert not struct_prop_evaluated({"a": [1], "b": []}, "b")


def test_cross_flow_mixture_density_and_bad_wells():
    mapping = SimpleNamespace(perf2well=np.array([0, 0, 1]), W=[object(), object()])
    mixture = cross_flow_mixture_density(
        mass_flux=np.array([[-2.0, -4.0], [1.0, 2.0], [0.0, 0.0]]),
        total_volume_flux=np.array([-1.0, 2.0, 3.0]),
        surface_mass_flux=np.array([[10.0, 20.0], [0.0, 0.0]]),
        mapping=mapping,
    )
    assert np.allclose(mixture[0], [12.0 / 2.0, 24.0 / 2.0])
    assert np.array_equal(mixture[1], [1.0, 1.0])
