"""CoolProp fluid property table (MRST ``coolPropFluidsStructs.m``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoolPropFluid:
    name: str
    Tcrit: float
    Pcrit: float
    rhocrit: float
    acentric: float
    molarmass: float


COOLPROP_FLUIDS = (
    CoolPropFluid('R507A', 343.765, 3704900, 490.74, 0.286, 0.0988592),
    CoolPropFluid('RC318', 388.38, 2777500, 619.972700656, 0.355345, 0.2000312),
    CoolPropFluid('SES36', 450.7, 2849000, 517.58, 0.352, 0.18485),
    CoolPropFluid('SulfurDioxide', 430.64, 7886587.60035, 525.002841, 0.256129936199, 0.0640638),
    CoolPropFluid('SulfurHexafluoride', 318.7232, 3754983, 742.3, 0.21, 0.1460554192),
    CoolPropFluid('Toluene', 591.75, 4126000, 291.98665298, 0.2657, 0.09213842),
    CoolPropFluid('trans-2-Butene', 428.61, 4027300, 236.37592616, 0.210076834436, 0.05610632),
    CoolPropFluid('Water', 647.096, 22064000, 322, 0.3442920843, 0.018015268),
    CoolPropFluid('Xenon', 289.733, 5842000, 1102.8612, 0.00363, 0.131293),
    CoolPropFluid('1-Butene', 419.29, 4005100, 237.8907968, 0.191860647355, 0.05610632),
    CoolPropFluid('Acetone', 508.1, 4700000, 272.971958, 0.3071, 0.05807914),
    CoolPropFluid('Air', 132.5306, 3786000, 342.684564168, 0.0335, 0.02896546),
    CoolPropFluid('Ammonia', 405.4, 11333000, 225, 0.25601, 0.01703026),
    CoolPropFluid('Argon', 150.687, 4863000, 535.6, -0.00219, 0.039948),
    CoolPropFluid('Benzene', 562.02, 4894000, 304.7922436, 0.21083697327, 0.0781118),
    CoolPropFluid('CarbonDioxide', 304.1282, 7377300, 467.600001282, 0.22394, 0.0440098),
    CoolPropFluid('CarbonMonoxide', 132.86, 3494000, 303.909585, 0.0497, 0.0280101),
    CoolPropFluid('CarbonylSulfide', 378.77, 6370000, 445.156491, 0.0978, 0.0600751),
    CoolPropFluid('cis-2-Butene', 435.75, 4225500, 238.11522208, 0.20235958587, 0.05610632),
    CoolPropFluid('CycloHexane', 553.6, 4082400, 271.33016352, 0.20926, 0.08415948),
    CoolPropFluid('Cyclopentane', 511.72, 4571200, 267.907678, 0.201112573381, 0.0701329),
    CoolPropFluid('CycloPropane', 398.3, 5579700, 258.500001907, 0.130549566369, 0.042081),
    CoolPropFluid('D4', 586.5, 1347215.35609, 309.37023768, 0.598124826799, 0.29661576),
    CoolPropFluid('D5', 619.15, 1160000, 304.909284957, 0.658, 0.3707697),
    CoolPropFluid('D6', 645.78, 961000, 279.095729841, 0.736, 0.444924),
    CoolPropFluid('Deuterium', 38.34, 1679600, 69.405886, -0.136290274128, 0.0040282),
    CoolPropFluid('Dichloroethane', 561.6, 5227585.10308, 428.49247, 0.268578500874, 0.098959),
    CoolPropFluid('DiethylEther', 466.7, 3649016.89653, 263.99999972, 0.281929331175, 0.0741216),
    CoolPropFluid('DimethylCarbonate', 557, 4908800, 360.3116, 0.346, 0.0900779),
    CoolPropFluid('DimethylEther', 400.378, 5336800, 273.6465336, 0.196, 0.04606844),
    CoolPropFluid('Ethane', 305.322, 4872200, 206.180000007, 0.099, 0.03006904),
    CoolPropFluid('Ethanol', 514.71, 6268000, 273.1858492, 0.644, 0.04606844),
    CoolPropFluid('EthylBenzene', 617.12, 3622400, 290.99996364, 0.304, 0.106165),
    CoolPropFluid('Ethylene', 282.35, 5041800, 214.24, 0.0866, 0.02805376),
    CoolPropFluid('EthyleneOxide', 468.92, 7304686.15964, 315.8568552, 0.21019548905, 0.04405256),
    CoolPropFluid('Fluorine', 144.414, 5172400, 592.86422643, 0.0449, 0.03799681),
    CoolPropFluid('HeavyWater', 643.847, 21672051.4798, 355.999969829, 0.364300568809, 0.020027508),
    CoolPropFluid('Helium', 5.1953, 227600, 72.56717426, -0.385, 0.004002602),
    CoolPropFluid('HFE143m', 377.921, 3635000, 465.00000003, 0.28887136567, 0.10004),
    CoolPropFluid('Hydrogen', 33.145, 1296400, 31.26226704, -0.219, 0.00201588),
    CoolPropFluid('HydrogenChloride', 324.55, 8288160.90445, 430.23862, 0.128189217431, 0.0364609),
    CoolPropFluid('HydrogenSulfide', 373.1, 9000000, 347.2841672, 0.1005, 0.03408088),
    CoolPropFluid('IsoButane', 407.817, 3629000, 225.5, 0.183531783208, 0.0581222),
    CoolPropFluid('IsoButene', 418.09, 4009800, 233.9633544, 0.192593452162, 0.05610632),
    CoolPropFluid('Isohexane', 497.7, 3040000, 233.9661024, 0.2797, 0.08617536),
    CoolPropFluid('Isopentane', 460.35, 3378000, 235.99865938, 0.2274, 0.07214878),
    CoolPropFluid('Krypton', 209.48, 5525000, 909.2083, -0.00089, 0.083798),
    CoolPropFluid('m-Xylene', 616.89, 3534600, 282.929725, 0.326, 0.106165),
    CoolPropFluid('MD2M', 599.4, 1227000, 284.171639662, 0.668, 0.310685),
    CoolPropFluid('MD3M', 628.36, 945000, 263.921879135, 0.722, 0.384839),
    CoolPropFluid('MD4M', 653.2, 877000, 278.177742673, 0.824647147264, 0.45899328),
    CoolPropFluid('MDM', 564.09, 1410044.75582, 256.739908797, 0.528065849054, 0.23653146),
    CoolPropFluid('Methane', 190.564, 4599200, 162.660002678, 0.01142, 0.0160428),
    CoolPropFluid('Methanol', 512.5, 8215850, 273, 0.5720322, 0.03204216),
    CoolPropFluid('MethylLinoleate', 799, 1341000, 238.051213304, 0.805406387056, 0.29447206),
    CoolPropFluid('MethylLinolenate', 772, 1369000, 247.798121314, 1.14260525867, 0.29245618),
    CoolPropFluid('MethylOleate', 782, 1246000, 241.000222029, 0.905849359988, 0.29648794),
    CoolPropFluid('MethylPalmitate', 755, 1350000, 242.59424202, 0.910316178, 0.27045066),
    CoolPropFluid('MethylStearate', 775, 1239000, 237.101584226, 1.01756, 0.29850382),
    CoolPropFluid('MM', 518.75, 1939000, 258.151840734, 0.418, 0.16237752),
    CoolPropFluid('n-Butane', 425.125, 3796000, 228, 0.200810094644, 0.0581222),
    CoolPropFluid('n-Decane', 617.7, 2103000, 233.3419552, 0.4884, 0.14228168),
    CoolPropFluid('n-Dodecane', 658.1, 1817000, 226.5453372, 0.574182221241, 0.17033484),
    CoolPropFluid('n-Heptane', 540.13, 2736000, 232, 0.349, 0.100202),
    CoolPropFluid('n-Hexane', 507.82, 3034000, 233.18, 0.299, 0.08617536),
    CoolPropFluid('n-Nonane', 594.55, 2281000, 232.141731, 0.4433, 0.1282551),
    CoolPropFluid('n-Octane', 569.32, 2497000, 234.8994874, 0.395, 0.1142285),
    CoolPropFluid('n-Pentane', 469.7, 3370000, 232, 0.251, 0.07214878),
    CoolPropFluid('n-Propane', 369.89, 4251200, 220.4781, 0.1521, 0.04409562),
    CoolPropFluid('n-Undecane', 638.8, 1990400, 236.794019, 0.539037101372, 0.15630826),
    CoolPropFluid('Neon', 44.4918, 2680000, 481.914878, -0.0384492992737, 0.020179),
    CoolPropFluid('Neopentane', 433.74, 3196000, 235.9265106, 0.1961, 0.07214878),
    CoolPropFluid('Nitrogen', 126.192, 3395800, 313.3, 0.0372, 0.02801348),
    CoolPropFluid('NitrousOxide', 309.52, 7245000, 452.011456, 0.1613, 0.0440128),
    CoolPropFluid('Novec649', 441.81, 1869026.58306, 606.804096, 0.47102359358, 0.3160438),
    CoolPropFluid('o-Xylene', 630.259, 3737500, 284.9999425, 0.312, 0.106165),
    CoolPropFluid('OrthoDeuterium', 38.34, 1679600, 69.405886, -0.136290274128, 0.0040282),
    CoolPropFluid('OrthoHydrogen', 33.22, 1310650, 31.1352666, -0.219, 0.00201594),
    CoolPropFluid('Oxygen', 154.581, 5043000, 436.143644, 0.0222, 0.0319988),
    CoolPropFluid('p-Xylene', 616.168, 3531500, 286.0000168, 0.324, 0.106165),
    CoolPropFluid('ParaDeuterium', 38.34, 1679600, 69.405886, -0.136290274128, 0.0040282),
    CoolPropFluid('ParaHydrogen', 32.938, 1285800, 31.32274344, -0.219, 0.00201588),
    CoolPropFluid('Propylene', 364.211, 4555000, 229.62914118, 0.146, 0.04207974),
    CoolPropFluid('Propyne', 402.38, 5626000, 244.8999998, 0.204, 0.04006),
    CoolPropFluid('R11', 471.06, 4394000, 565, 0.188750648253, 0.137368),
    CoolPropFluid('R113', 487.21, 3392200, 560, 0.252535, 0.187375),
    CoolPropFluid('R114', 418.83, 3257000, 579.9691372, 0.2523, 0.170921),
    CoolPropFluid('R115', 353.1, 3129036.96828, 614.77633568, 0.248434993066, 0.154466416),
    CoolPropFluid('R116', 293.03, 3048000, 613.32452808, 0.2566, 0.13801182),
    CoolPropFluid('R12', 385.12, 4136100, 565, 0.179478317344, 0.120913),
    CoolPropFluid('R123', 456.831, 3672000, 550.003648227, 0.281922497036, 0.152931),
    CoolPropFluid('R1233zd(E)', 439.6, 3623637.77636, 480.219392, 0.30245229358, 0.1304944),
    CoolPropFluid('R1234yf', 367.85, 3382200, 475.553441976, 0.276, 0.1140415928),
    CoolPropFluid('R1234ze(E)', 382.52, 3636250, 489.238433112, 0.313, 0.1140415928),
    CoolPropFluid('R1234ze(Z)', 423.27, 3533000, 470, 0.3274, 0.1140415928),
    CoolPropFluid('R124', 395.425, 3624295, 560, 0.288095084221, 0.1364762),
    CoolPropFluid('R125', 339.173, 3617700, 573.5822706, 0.3052, 0.1200214),
    CoolPropFluid('R13', 301.88, 3879000, 582.88122, 0.174586327798, 0.104459),
    CoolPropFluid('R134a', 374.21, 4059280, 511.899951696, 0.32684, 0.102032),
    CoolPropFluid('R13I1', 396.44, 3952566.07172, 868.00061824, 0.17618111777, 0.1959104),
    CoolPropFluid('R14', 227.51, 3750000, 625.661610529, 0.1785, 0.0880046),
    CoolPropFluid('R141b', 477.5, 4212000, 458.55946002, 0.2195, 0.11694962),
    CoolPropFluid('R142b', 410.26, 4055000, 445.99694314, 0.2321, 0.10049503),
    CoolPropFluid('R143a', 345.857, 3761000, 431.00006645, 0.261489646224, 0.084041),
    CoolPropFluid('R152A', 386.411, 4520000, 368, 0.275217114532, 0.066051),
    CoolPropFluid('R161', 375.25, 5010000, 301.81366, 0.216242841066, 0.0480595),
    CoolPropFluid('R21', 451.48, 5181200, 526.013794619, 0.2061, 0.1029227),
    CoolPropFluid('R218', 345.02, 2640000, 627.9845622, 0.3172, 0.18801933),
    CoolPropFluid('R22', 369.295, 4990000, 523.84216696, 0.22082, 0.086468),
    CoolPropFluid('R227EA', 374.9, 2925242.23394, 594.2508657, 0.357641241953, 0.17002886),
    CoolPropFluid('R23', 299.293, 4832000, 526.504152, 0.262964892496, 0.07001385),
    CoolPropFluid('R236EA', 412.44, 3420000, 564.9746944, 0.368782380399, 0.1520384),
    CoolPropFluid('R236FA', 398.07, 3200000, 551.2912384, 0.376911797634, 0.1520384),
    CoolPropFluid('R245ca', 447.57, 3940740.48875, 525.4679248, 0.354565419124, 0.13404794),
    CoolPropFluid('R245fa', 427.01, 3651000, 519.4357675, 0.3776, 0.13404794),
    CoolPropFluid('R32', 351.255, 5782000, 424.00000123, 0.2769, 0.052024),
    CoolPropFluid('R365MFC', 460, 3266207.06788, 473.838464, 0.377445097168, 0.14807452),
    CoolPropFluid('R40', 416.3, 6671686.97714, 363.218982472, 0.150068496396, 0.05048752),
    CoolPropFluid('R404A', 345.27, 3734800, 482.162772, 0.293, 0.0976038),
    CoolPropFluid('R407C', 359.345, 4631700, 453.43094, 0.363, 0.0862036),
    CoolPropFluid('R41', 317.28, 5897000, 316.506156, 0.2004, 0.03403292),
    CoolPropFluid('R410A', 344.494, 4901200, 459.0300696, 0.296, 0.0725854),
)


def coolPropFluidsStructs() -> tuple[CoolPropFluid, ...]:
    """Return MRST's stored CoolProp fluid structs as immutable records."""
    return tuple(COOLPROP_FLUIDS)


def cool_prop_fluids_structs() -> tuple[CoolPropFluid, ...]:
    """Pythonic alias for :func:`coolPropFluidsStructs`."""
    return coolPropFluidsStructs()


__all__ = ["COOLPROP_FLUIDS", "CoolPropFluid", "coolPropFluidsStructs", "cool_prop_fluids_structs"]
