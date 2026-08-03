"""Shale compositional fluid registry (`getShaleCompFluidCase`)."""

from dataclasses import dataclass
import numpy as np

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.fluids.fluid_cases import FluidCaseInfo, get_compositional_fluid_case
from pymrst.modules.compositional.fluids.table_compositional_mixture import TableCompositionalMixture
from pymrst.utils.units import PSIA, RANKINE
from pymrst.modules.shale.compositional_fns.sorbed_compositional_fluid import SorbedCompositionalFluid


BARSA = 1.0e5


@dataclass
class ShaleFluidCaseInfo(FluidCaseInfo):
    diffusion_coefficients: np.ndarray | None = None
    parachor: np.ndarray | None = None


def _info(injection, initial, pressure, temperature, diffusion=None, parachor=None):
    return ShaleFluidCaseInfo(np.asarray(injection, float), np.asarray(initial, float),
                              float(pressure), float(temperature),
                              None if diffusion is None else np.asarray(diffusion, float),
                              None if parachor is None else np.asarray(parachor, float))


def _mixture(names, tc, pc, vc, acc, mw, bic, label):
    fluid = Mixture(tuple(names), np.asarray(tc), np.asarray(pc), np.asarray(vc),
                    np.asarray(acc), np.asarray(mw), name=label)
    fluid.set_binary_interaction(np.asarray(bic))
    return fluid


def _table_case(names, injection, initial, pressure, temperature, *, outnames=None):
    """Construct one of MRST's table-backed shale catalog entries."""
    fluid = TableCompositionalMixture(names, outnames=outnames)
    return fluid, _info(injection, initial, pressure, temperature)


def get_shale_comp_fluid_case(name: str):
    """Return a native shale mixture and MRST initial/injection metadata."""
    key = name.lower().replace("-", "_")
    if key in {"lumped_1", "spe5"}:
        fluid, base = get_compositional_fluid_case(key)
        # These shale-local copies use 8.314 exactly, unlike the base
        # compositional registry's higher-precision gas constant.
        zcrit = ([.28981, .27055, .27588, .25668, .21967, .18250]
                 if key == "lumped_1" else [.290, .277, .264, .257, .245, .235])
        fluid.Vcrit = np.asarray(zcrit) * 8.314 * fluid.Tcrit / fluid.Pcrit
        return fluid, _info(base.injection, base.initial, base.pressure, base.T)
    if key == "simple":
        return _table_case(
            ("Methane", "CarbonDioxide", "n-Decane"),
            [.1, .9, 0], [.3, .1, .6], 75 * BARSA, 423.15,
        )
    if key == "fourmix":
        return _table_case(
            ("Methane", "Nitrogen", "n-Pentane", "n-Decane"),
            [0, 1, 0, 0], [.1, 0, .45, .45], 75 * BARSA, 323.15,
        )
    if key == "20components":
        names = (
            "Methane", "Nitrogen", "n-Pentane", "n-Decane", "CycloHexane",
            "CarbonMonoxide", "1-Butene", "Ammonia", "HydrogenSulfide",
            "IsoButane", "n-Dodecane", "n-Heptane", "n-Hexane", "n-Nonane",
            "n-Octane", "p-Xylene", "n-Propane", "n-Undecane", "Oxygen", "Water",
        )
        # Preserve MRST's deliberately non-normalized diagnostic injection vector.
        return _table_case(names, np.arange(1, 21) / 20, np.ones(20) / 20,
                           75 * BARSA, 323.15)
    if key in {"liquid_initial", "vapor_initial"}:
        injection = [1, 0] if key == "liquid_initial" else [0, 1]
        initial = injection[::-1]
        return _table_case(("Oxygen", "Water"), injection, initial,
                           75 * BARSA, 303.15, outnames=("O2", "H2O"))
    if key == "watertracer":
        return _table_case(("Water", "Water"), [1, 0], [0, 1],
                           75 * BARSA, 303.15, outnames=("Tracer1", "Tracer2"))
    if key == "verysimple":
        return _table_case(("CarbonDioxide", "n-Decane"), [1, 0], [.1, .9],
                           75 * BARSA, 423.15)
    if key == "onlydecane":
        return _table_case(("n-Decane",), [1], [1], 75 * BARSA, 423.15)
    if key == "watermethane":
        return _table_case(("Methane", "Water"), [1, 0], [.01, .99],
                           75 * BARSA, 423.15)
    if key == "onlymethane":
        return _table_case(("Methane",), [1], [1], 5000 * PSIA, 366.48)
    if key == "c1c2":
        return _table_case(("Methane", "Ethane"), [.999, .001], [.999, .001],
                           5000 * PSIA, 366.48)
    if key in {"barnett3comps", "c1c2c3"}:
        names = ("Methane", "Ethane", "n-Propane")
        fluid = (SorbedCompositionalFluid(names, [[10769611, 5591648, 5819175], [2.99, 4.86, 9.56]])
                 if key == "barnett3comps" else TableCompositionalMixture(names))
        fluid.set_binary_interaction([[0, .005, .01], [.005, 0, .005], [.01, .005, 0]])
        pressure = (5000 if key == "barnett3comps" else 5466.482220000024) * PSIA
        diffusion = [2.8e-7, 2.5e-7, 1.9e-7] if key == "barnett3comps" else None
        return fluid, _info([0, 0, 1], [.991, .0088, .0002], pressure, 355.84, diffusion)
    if key == "bakken":
        tc = np.array([335.336, 549.969, 665.97, 759.208, 875.479, 1053.25, 1332.095, 1844.491])*RANKINE
        pc = np.array([655.02, 721.99, 615.76, 546.46, 461.29, 363.34, 249.61, 190.12])*PSIA
        vc = np.array([1.58, 2.34, 3.25, 4.11, 5.39, 8.81, 15.19, 36])*.06242795996802/1000
        mw = np.array([16.535, 30.433, 44.097, 58.124, 78.295, 120.562, 220.716, 443.518])/1000
        bic = np.zeros((8, 8)); bic[0] = [0,.005,.0035,.0035,.0037,.0033,.0033,.0033]; bic[1,2:] = [.0031,.0031,.0031,.0026,.0026,.0026]
        bic = np.triu(bic) + np.triu(bic, 1).T
        fluid = _mixture(["C1","C2","C3","C4","C5-C6","C7-C12","C13-C21","C22-C80"], tc, pc, vc,
                         [.0102,.1028,.152,.1894,.2684,.4291,.7203,1.0159], mw, bic, "Bakken shale")
        return fluid, _info([1,0,0,0,0,0,0,0], [.36736,.14885,.09334,.05751,.06406,.15854,.0733,.03704],
                            5700*PSIA, 352.594, parachor=[74.8,107.7,151.9,189.6,250.2,350.2,590,1216.8])
    if key == "eagleford":
        tc = np.array([547.56, 227.16, 343.08, 494.532, 789.624, 1332.522]) * RANKINE
        pc = np.array([1069.865, 492.3143, 667.1961, 536.4021, 368.5744, 257.9139]) * PSIA
        vc = np.array([9.4e-5, 8.95e-5, 9.9e-5, 2.293e-4, 3.943e-4, 8.87e-4])
        mw = np.array([44.01, 28.01, 16.04, 52.02, 103.01, 267.15]) / 1000
        bic = np.array([
            [0, .02, .103, .1299, .15, .15],
            [0, 0, .031, .082, .12, .12],
            [0, 0, 0, .0174, .0462, .111],
            [0, 0, 0, 0, .0073, .0444],
            [0, 0, 0, 0, 0, .0162],
            [0, 0, 0, 0, 0, 0],
        ])
        bic = np.triu(bic) + np.triu(bic, 1).T
        fluid = _mixture(
            ["CO2", "N2", "C1", "C2-C5", "C6-C10", "C11+"],
            tc, pc, vc, [.225, .04, .008, .1723, .2839, .6716], mw, bic,
            "Eagle Ford shale",
        )
        # MRST's source contains eight injection zeros for a six-component
        # mixture. Keep the physical six-component stream so downstream EOS
        # and well constructors receive a dimensionally valid vector.
        return fluid, _info(
            np.zeros(6), [.01183, .00161, .11541, .26438, .38089, .22588],
            8125 * PSIA, 405.372, parachor=[78, 41, 77, 171.07, 297.42, 661.45],
        )
    if key == "bakken_light":
        tc = np.array([343.08, 653.94, 920.808, 1042.812, 1419.732]) * RANKINE
        pc = np.array([667.38, 625.338, 496.272, 454.377, 317.226]) * PSIA
        vc = np.array([1.586, 3.156, 5.347, 6.507, 14.749]) * .06242795996802 / 1000
        mw = np.array([16.04, 42.82, 83.74, 105.91, 200.0]) / 1000
        bic = np.array([
            [0, .0078, .0242, .0324, .0779],
            [.0078, 0, .0046, .0087, .0384],
            [.0242, .0046, 0, .0006, .0169],
            [.0324, .0087, .0006, 0, .0111],
            [.0779, .0384, .0169, .0111, 0],
        ])
        fluid = _mixture(
            ["C1", "C2-C4", "C5-C7", "C8-C9", "C10+"],
            tc, pc, vc, [.008, .1432, .2474, .2861, .6869], mw, bic,
            "Bakken light shale",
        )
        return fluid, _info(
            [1, 0, 0, 0, 0], [.2506, .22, .2, .13, .1994], 5700 * PSIA, 352.594,
            parachor=[77, 145.2, 250, 306, 686.3],
        )
    raise ValueError(f"unknown shale compositional fluid case {name!r}")


__all__ = ["ShaleFluidCaseInfo", "get_shale_comp_fluid_case"]
