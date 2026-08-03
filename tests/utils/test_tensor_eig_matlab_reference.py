"""MATLAB/MRST references for eigen and named tensor utilities."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.utils import multi_eig, multi_symm_eig
from pymrst.utils.tensor_assembly import (
    IndexArray,
    SparseTensor,
    TensorProd,
    proj_index_array,
    sort_index_array,
)
from pymrst.utils.tensor_assembly.utils import (
    comp_index_array,
    concat_index_array,
    replacefield,
)


REF = loadmat(Path(__file__).parents[2] /
              "scripts/matlab/ref_utils_tensor_eig.mat", squeeze_me=True)


def vec(name, dtype=None):
    return np.asarray(REF[name], dtype=dtype).ravel()


def tables():
    t1 = IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})
    t2 = IndexArray({"faces": [3, 7, 7], "nodes": [10, 20, 30]})
    return t1, t2


def test_multi_eig_values_match_blockwise_matlab_eig():
    got = multi_eig(vec("general_flat"), vec("general_sizes", int))
    expected = vec("general_values")
    np.testing.assert_allclose(np.sort_complex(got),
                               np.sort_complex(expected), rtol=1e-12)


def test_multi_symm_eig_values_and_vectors_match_matlab_problem():
    flat, sizes = vec("symm_flat"), vec("symm_sizes", int)
    values, vectors = multi_symm_eig(flat, sizes, vectors=True)
    np.testing.assert_allclose(values, vec("symm_values"), rtol=1e-12)
    offset_a = offset_v = offset_d = 0
    for size in sizes:
        n = int(size)
        a = flat[offset_a:offset_a+n*n].reshape((n, n), order="F")
        v = vectors[offset_v:offset_v+n*n].reshape((n, n), order="F")
        d = values[offset_d:offset_d+n]
        np.testing.assert_allclose(a @ v, v * d, atol=1e-12)
        np.testing.assert_allclose(v.T @ v, np.eye(n), atol=1e-12)
        offset_a += n*n
        offset_v += n*n
        offset_d += n


def test_projection_and_sort_match_mrst_index_arrays():
    t1, _ = tables()
    projected, mapping = proj_index_array(t1, ["cells"])
    np.testing.assert_array_equal((projected.inds + 1).squeeze(),
                                  REF["projected_inds"])
    np.testing.assert_array_equal(mapping["inds"] + 1,
                                  vec("projection_inds"))
    sorted_table = sort_index_array(t1, ["faces", "cells"])
    np.testing.assert_array_equal(sorted_table.inds + 1, REF["sorted_inds"])


def test_replacefield_and_concat_match_mrst():
    t1, _ = tables()
    renamed = replacefield(t1, ("cells", "c"))
    np.testing.assert_array_equal(renamed.inds + 1, REF["renamed_inds"])
    assert renamed.fdnames == [str(v) for v in vec("renamed_names")]
    swapped = replacefield(t1, ("cells", "faces", "interchange"))
    np.testing.assert_array_equal(swapped.inds + 1, REF["interchanged_inds"])
    assert swapped.fdnames == [str(v) for v in vec("interchanged_names")]
    disjoint = IndexArray({"cells": [2, 3], "faces": [8, 9]})
    out = concat_index_array(t1, disjoint, ["cells", "faces"])
    np.testing.assert_array_equal(out.inds + 1, REF["concatenated_inds"])


def test_comp_index_array_repairs_the_recorded_mrst_defect():
    assert not bool(REF["comp_ran"])
    assert "logical" in str(REF["comp_message"]).lower()
    t1, _ = tables()
    assert comp_index_array(t1, t1) is True
    different = IndexArray({"cells": [9, 9, 9], "faces": [9, 9, 9]})
    assert comp_index_array(t1, different) is False


def test_tensor_prod_dispatch_and_evaluation_match_mrst():
    t1, t2 = tables()
    product = TensorProd(tbl1=t1, tbl2=t2, reducefds=["faces"]).setup()
    np.testing.assert_array_equal(product.tbl3.inds + 1, REF["product_tbl3"])
    np.testing.assert_array_equal(product.dispind1 + 1, vec("product_disp1"))
    np.testing.assert_array_equal(product.dispind2 + 1, vec("product_disp2"))
    np.testing.assert_array_equal(product.dispind3 + 1, vec("product_disp3"))
    got = product.eval(vec("left"), vec("right"))
    np.testing.assert_allclose(got, vec("product_out"))


def test_sparse_tensor_matrix_and_application_match_mrst():
    t1, t2 = tables()
    product = TensorProd(tbl1=t1, tbl2=t2, reducefds=["faces"]).setup()
    tensor = SparseTensor.from_tensor_prod(vec("left"), product, argindex=1)
    np.testing.assert_allclose(tensor.get_matrix().toarray(),
                               REF["sparse_matrix"])
    np.testing.assert_allclose(tensor @ vec("right"),
                               vec("sparse_applied"))
