"""
PFR in isolation: the methanol reactor with its inlet and outlet streams only.
Generic helpers live in ``hysys.py``.

    from hysys import open_case
    from hysys_pfr import cache_objects, run_point

    case    = open_case(r"C:\\...\\PFR.hsc")
    objects = cache_objects(case)
    row     = run_point(objects, pressure=(90, "bar"), temperature=(259, "C"), volume=(19, "m3"),
                        flows=({"Hydrogen": 8106, "CO": 212, "CO2": 1803, "H2O": 29,
                                "Methanol": 16, "Nitrogen": 162}, "kgmole/h"))

Dimensioned inputs are ``(number, "unit")`` pairs; `flows` is a
``({component: number}, "unit")`` pair and unnamed components are zero.  The
reactor runs isothermally: outlet temperature is set equal to the inlet and
the duty is a result.  Result keys name their unit.
"""
from __future__ import annotations

import time
from datetime import datetime

from hysys import read, write, by_component, solve
from hysys_methanol import set_segments

__all__ = ["cache_objects", "run_point"]

FLOW_UNIT = "kgmole/h"          # unit of the in_* / out_* result keys


def cache_objects(case, *, segments: int) -> dict:
    """
    Look up the streams and reactor used by `run_point` once per session and
    set the PFR discretisation to `segments`, the same value the loop case
    gets in ``hysys_methanol.cache_objects`` so both reactors agree.
    """
    fs = case.Flowsheet
    rin = fs.MaterialStreams("RinV")
    reactor = fs.Operations("Reactor100")
    set_segments(reactor, segments)
    return {
        "solver": case.Solver,
        "rin": rin,                                  # reactor inlet: fully specified per point
        "rout": fs.MaterialStreams("RoutV"),         # reactor outlet
        "reactor": reactor,
        "components": list(rin.FluidPackage.Components.Names),   # order of component arrays
        "segments": int(segments),
    }


def run_point(objects: dict, *, pressure, temperature, volume, flows,
              timeout: float = 120.0) -> dict:
    """
    Set inlet conditions, composition and reactor volume, solve, return one flat row.

    Outlet flows come back per component as ``out_<name>_kgmole_h`` next to
    the inlet ``in_<name>_kgmole_h``, plus CO2 conversion, duty and pressure drop.
    """
    o = objects
    solver, rin, rout, reactor = o["solver"], o["rin"], o["rout"], o["reactor"]
    by_name, unit = flows
    unknown = set(by_name) - set(o["components"])
    if unknown:
        raise KeyError(f"not in fluid package: {sorted(unknown)}; valid: {o['components']}")
    t0 = time.time()

    solver.CanSolve = False
    write(rin.Pressure, pressure)
    write(rin.Temperature, temperature)
    write(rout.Temperature, temperature)            # isothermal reactor, duty is a result
    rin.ComponentMolarFlow.SetValues(tuple(float(by_name.get(c, 0.0)) for c in o["components"]), unit)
    write(reactor.TotalVolume, volume)
    solve(solver, timeout)
    solver.CanSolve = True                          # leave HYSYS live for inspection

    converged = bool(rout.MolarFlow.IsKnown and rout.Temperature.IsKnown)
    inlet = by_component(rin, "ComponentMolarFlow", FLOW_UNIT)
    outlet = by_component(rout, "ComponentMolarFlow", FLOW_UNIT) if converged else {}
    tag = FLOW_UNIT.replace("/", "_")
    return {
        "pressure_bar": read(rin.Pressure, "bar"),
        "temperature_C": read(rin.Temperature, "C"),
        "volume_m3": read(reactor.TotalVolume, "m3"),
        **{f"in_{c}_{tag}": v for c, v in inlet.items()},
        **{f"out_{c}_{tag}": v for c, v in outlet.items()},
        "co2_conversion": 1 - outlet["CO2"] / inlet["CO2"] if converged and inlet["CO2"] else None,
        "co_formation_kgmole_h": outlet["CO"] - inlet["CO"] if converged else None,
        "reactor_duty_kW": read(reactor.HeatFlow, "kW") if converged else None,
        "reactor_dP_bar": read(reactor.PressureDrop, "bar") if converged else None,
        "segments": o["segments"],
        "converged": converged,
        "solve_time_s": round(time.time() - t0, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
