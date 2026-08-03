"""Hydrostatic initial pressure (MRST ``utils/incompHydrostaticPressure.m``).

Integrates `dp/dz = rho g` down the column, switching density at each fluid
contact, and samples the result at the cell centroids. Returns both the
per-cell pressure and the depth-to-pressure function, because equilibration
code needs to evaluate at depths that are not cell centres.

`contacts` are the depths where the fluid changes, ascending;
`densities` has one more entry than `contacts` - the fluid above the first
contact, then one per contact. Getting that off by one puts every fluid in the
wrong compartment, so the lengths are checked rather than assumed.

The integration reproduces MRST's running sample loop exactly, including its
indexing quirk: after the first point at ``z_min``, the next stored depth is
``z_min + 2*h`` while pressure advances by only one ``h``. The last stored
depth is consequently ``z_max + h``. This is observable near the top and in
reference results, so correcting it here would break parity. MRST also linearly
extrapolates outside this sampled interval.
"""

import numpy as np

from pymrst.params.gravity_const import gravity_vector


def incomp_hydrostatic_pressure(g, contacts, densities, top_pressure=0.0,
                                n_samples: int = 1000):
    """Return ``(pressure_per_cell, depth_to_pressure)``."""
    contacts = np.atleast_1d(np.asarray(contacts, dtype=float)).ravel()
    densities = np.atleast_1d(np.asarray(densities, dtype=float)).ravel()
    if densities.size != contacts.size + 1:
        raise ValueError(
            f"densities must have one more entry than contacts: got "
            f"{densities.size} densities for {contacts.size} contact(s)")
    if np.any(np.diff(contacts) < 0):
        raise ValueError("contacts must be given in increasing depth")

    centroids = np.asarray(g.cells.centroids, dtype=float)
    if centroids.shape[1] < 3:
        raise ValueError("a hydrostatic column needs a depth coordinate; "
                         "this grid is two-dimensional")
    depth = centroids[:, 2]

    z_min, z_max = float(depth.min()), float(depth.max())
    magnitude = float(np.linalg.norm(gravity_vector(3)))
    step = (z_max - z_min) / (n_samples - 1)

    samples = np.empty(n_samples)
    pressures = np.empty(n_samples)
    samples[0], pressures[0] = z_min, float(top_pressure)
    for i in range(1, n_samples):
        samples[i] = z_min + step * (i + 1)
        below = np.flatnonzero(contacts < samples[i])
        index = below[-1] + 1 if below.size else 0
        pressures[i] = pressures[i - 1] + densities[index] * step * magnitude

    def depth_to_pressure(z):
        query = np.asarray(z, dtype=float)
        flat = query.ravel()
        result = np.interp(flat, samples, pressures)
        low = flat < samples[0]
        high = flat > samples[-1]
        if np.any(low):
            slope = (pressures[1] - pressures[0]) / (samples[1] - samples[0])
            result[low] = pressures[0] + slope * (flat[low] - samples[0])
        if np.any(high):
            slope = (pressures[-1] - pressures[-2]) / \
                (samples[-1] - samples[-2])
            result[high] = pressures[-1] + slope * (flat[high] - samples[-1])
        return result.reshape(query.shape)

    return depth_to_pressure(depth), depth_to_pressure


incompHydrostaticPressure = incomp_hydrostatic_pressure


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    GRID = compute_geometry(cart_grid([2, 2, 20], [10.0, 10.0, 100.0]))
    G = float(np.linalg.norm(gravity_vector(3)))

    def test_a_single_fluid_is_linear_in_depth() -> None:
        p, _ = incomp_hydrostatic_pressure(GRID, [], [1000.0])
        depth = np.asarray(GRID.cells.centroids)[:, 2]
        slope = np.polyfit(depth, p, 1)[0]
        assert np.isclose(slope, 1000.0 * G, rtol=1e-3), slope
        print(f"ok test_a_single_fluid_is_linear_in_depth "
              f"(gradient {slope:.1f} Pa/m)")

    def test_pressure_increases_with_depth() -> None:
        p, _ = incomp_hydrostatic_pressure(GRID, [], [1000.0])
        depth = np.asarray(GRID.cells.centroids)[:, 2]
        order = np.argsort(depth)
        assert np.all(np.diff(p[order]) >= -1e-9)
        print("ok test_pressure_increases_with_depth")

    def test_the_top_pressure_is_honoured() -> None:
        p, fn = incomp_hydrostatic_pressure(GRID, [], [1000.0],
                                            top_pressure=1.0e5)
        zmin = np.asarray(GRID.cells.centroids)[:, 2].min()
        assert np.isclose(fn(zmin), 1.0e5), fn(zmin)
        print("ok test_the_top_pressure_is_honoured")

    def test_the_gradient_changes_at_a_contact() -> None:
        """The reason contacts exist: a lighter fluid above a heavier one."""
        p, fn = incomp_hydrostatic_pressure(GRID, [50.0], [800.0, 1000.0])
        above = (fn(40.0) - fn(20.0)) / 20.0
        below = (fn(90.0) - fn(70.0)) / 20.0
        assert np.isclose(above, 800.0 * G, rtol=1e-2), above
        assert np.isclose(below, 1000.0 * G, rtol=1e-2), below
        print("ok test_the_gradient_changes_at_a_contact")

    def test_the_returned_function_matches_the_cell_values() -> None:
        p, fn = incomp_hydrostatic_pressure(GRID, [50.0], [800.0, 1000.0])
        depth = np.asarray(GRID.cells.centroids)[:, 2]
        assert np.allclose(fn(depth), p)
        print("ok test_the_returned_function_matches_the_cell_values")

    def test_two_contacts_give_three_gradients() -> None:
        _, fn = incomp_hydrostatic_pressure(
            GRID, [30.0, 70.0], [500.0, 800.0, 1000.0])
        for lo, hi, rho in ((5.0, 25.0, 500.0), (40.0, 60.0, 800.0),
                            (75.0, 95.0, 1000.0)):
            gradient = (fn(hi) - fn(lo)) / (hi - lo)
            assert np.isclose(gradient, rho * G, rtol=2e-2), (rho, gradient)
        print("ok test_two_contacts_give_three_gradients")

    def test_a_wrong_density_count_is_rejected() -> None:
        try:
            incomp_hydrostatic_pressure(GRID, [50.0], [1000.0])
        except ValueError as error:
            assert "one more entry" in str(error)
            print("ok test_a_wrong_density_count_is_rejected")
            return
        raise AssertionError("n contacts need n+1 densities")

    def test_unsorted_contacts_are_rejected() -> None:
        try:
            incomp_hydrostatic_pressure(GRID, [70.0, 30.0], [1.0, 2.0, 3.0])
        except ValueError as error:
            assert "increasing depth" in str(error)
            print("ok test_unsorted_contacts_are_rejected")
            return
        raise AssertionError("descending contacts must be rejected")

    def test_a_two_dimensional_grid_is_rejected() -> None:
        flat = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        try:
            incomp_hydrostatic_pressure(flat, [], [1000.0])
        except ValueError as error:
            assert "two-dimensional" in str(error)
            print("ok test_a_two_dimensional_grid_is_rejected")
            return
        raise AssertionError("a 2-D grid has no depth column")

    test_a_single_fluid_is_linear_in_depth()
    test_pressure_increases_with_depth()
    test_the_top_pressure_is_honoured()
    test_the_gradient_changes_at_a_contact()
    test_the_returned_function_matches_the_cell_values()
    test_two_contacts_give_three_gradients()
    test_a_wrong_density_count_is_rejected()
    test_unsorted_contacts_are_rejected()
    test_a_two_dimensional_grid_is_rejected()
    print("\nAll tests passed!")
