"""Drive the interactive inspector with a real simulation (plotToolbar).

Runs a two-phase oil-water waterflood on a layered reservoir, keeps every
timestep, and hands the whole schedule to the toolbar. You can then scrub
time, switch between pressure and saturation, threshold to find the swept
region, and cut a plane through the model to see the water underrunning the
tight layer.

    python examples/toolbar_demo.py                 # opens the window
    python examples/toolbar_demo.py --shot out.png  # headless, one frame

In the window:

    n / p       next / previous dataset
    left/right  step back / forward in time
    space       play the schedule
    t           cycle transform (linear, log10, abs, log10(abs))
    c           toggle the clip plane (drag it with the mouse)
    h           toggle the histogram
    r           reset the threshold

With pyvistaqt installed you also get Dataset / Transform / View menus.
"""

import argparse

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import TwoPhaseOilWaterModelAD
from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.plotting import PlotToolbar
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE, YEAR


def build():
    """A layered reservoir with a tight streak the water must get around."""
    nx, ny, nz = 20, 10, 6
    g = compute_geometry(cart_grid([nx, ny, nz], [400.0, 200.0, 30.0]))

    z = g.cells.centroids[:, 2]
    layer = np.floor(z / (30.0 / nz)).astype(int)
    perm = np.array([400.0, 600.0, 5.0, 300.0, 150.0, 80.0])[layer]
    rock = make_rock(g, (perm * MILLI * DARCY)[:, None],
                     np.full((g.cells.num, 1), 0.2))

    fluid = init_simple_adi_fluid(
        mu=[1 * CENTI * POISE, 5 * CENTI * POISE],
        rho=[1000.0, 800.0], n=[2.0, 2.0],
        c=[1e-10, 1e-9], p_ref=200 * BARSA,
    )

    inj = np.arange(nz) * nx * ny                       # heel column
    prod = inj + (nx - 1)                               # toe column
    W = add_well(None, g, rock, inj, well_type="rate", val=300.0 / DAY,
                 compi=np.array([1.0, 0.0]), name="INJ")
    W = add_well(W, g, rock, prod, well_type="bhp", val=200 * BARSA,
                 compi=np.array([1.0, 0.0]), name="PROD")

    model = TwoPhaseOilWaterModelAD(g, rock, fluid, W)
    return g, rock, model


def run(model, n_steps=24, dt=30.0 * DAY):
    """Waterflood, keeping every state for the toolbar to scrub."""
    st = model.initial_state(200 * BARSA, 0.15)
    states, times = [st], [0.0]
    for k in range(n_steps):
        st, ok, its = model.step(states[-1], dt)
        if not ok:
            raise RuntimeError(f"step {k} did not converge")
        states.append(st)
        times.append(times[-1] + dt)
    return states, np.array(times)


def main(shot=None):
    g, rock, model = build()
    print(f"Reservoir: {g.cells.num} cells, "
          f"perm {rock.perm.min() / (MILLI * DARCY):.0f}-"
          f"{rock.perm.max() / (MILLI * DARCY):.0f} mD "
          f"(a 5 mD streak in layer 3)")

    states, times = run(model)
    sw = np.stack([s.sW for s in states])
    print(f"Simulated {len(states) - 1} steps to "
          f"{times[-1] / YEAR:.1f} years; water saturation "
          f"{sw[0].mean():.3f} -> {sw[-1].mean():.3f}")

    # states carry pressure and sW; add the static rock so it can be
    # compared against the flood front in the same window
    fields = {
        "pressure": np.stack([s.pressure for s in states]),
        "sW": sw,
        "perm": rock.perm[:, 0],
        "poro": rock.poro,
    }

    tb = PlotToolbar(g, fields, times=times, time_unit=YEAR,
                     time_label="years", cmap="viridis",
                     off_screen=shot is not None,
                     title="PyMRST - waterflood")

    if shot is not None:
        tb.set_field("sW")
        tb.set_step(tb.n_steps - 1)
        tb.screenshot(shot)
        print(f"wrote {shot}")
        tb.close()
        return

    tb.set_field("sW")
    print("\nWindow open. n/p dataset, left/right time, space play, "
          "c clip, h histogram.")
    tb.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default=None,
                    help="render one frame headlessly and exit")
    main(ap.parse_args().shot)
