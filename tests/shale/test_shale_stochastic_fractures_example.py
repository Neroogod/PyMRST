import numpy as np

from pymrst.modules.shale import setup_stochastic_fractures


def test_bundled_stochastic_example_preserves_mrst_realization_and_sets():
    case = setup_stochastic_fractures()
    assert case.source == "bundled-mrst"
    assert case.grid.cartDims == (98, 15, 3)
    assert case.grid.cells.num == 4410
    assert case.hydraulic_count == case.natural_count == 350
    assert len(case.fracture_planes) == 700
    assert case.planar_count == 700
    assert case.nonplanar_count == 0
    assert case.fracture_planes[0].metadata == {"set": "hydraulic", "source_index": 0}
    assert case.fracture_planes[350].metadata == {"set": "natural", "source_index": 0}
    assert case.areas[0] > 0


def test_stochastic_example_can_preprocess_a_bounded_reference_subset():
    case = setup_stochastic_fractures(build_edfm=True, edfm_limit=1)
    assert case.global_grid is not None
    assert case.global_grid.n_fracture > 0
    assert len(case.processed_planes) == 1
    assert case.processed_planes[0].metadata["processed"] is True


def test_external_stochastic_generator_is_seeded_and_typed():
    calls = []

    def generator(rng, physical, tolerance):
        calls.append((physical.copy(), tolerance))
        x = rng.uniform(1, physical[0]-1)
        polygon = np.array([
            [x, 1, 1], [x, 3, 1], [x, 3, 3], [x, 1, 3], [x, 1, 1],
        ])
        return [polygon], [polygon + [1, 0, 0]]

    first = setup_stochastic_fractures(generator=generator, seed=12345)
    second = setup_stochastic_fractures(generator=generator, seed=12345)
    assert first.source == "generator"
    assert first.hydraulic_count == first.natural_count == 1
    assert np.allclose(first.fracture_planes[0].points, second.fracture_planes[0].points)
    assert len(calls) == 2
