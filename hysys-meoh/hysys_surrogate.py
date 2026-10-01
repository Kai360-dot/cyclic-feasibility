"""
Methanol loop with the PFR replaced by a surrogate.  The case is a copy of
``methanol_loop.hsc`` with Reactor100 ignored, so RoutV is free to be written,
and without RCY-1: with the reactor ignored the HYSYS flowsheet has no cycle
left, so the tear RinV -> surrogate -> RoutV is iterated from Python alone and
the rest of the flowsheet, including the separation, is solved by HYSYS in one
pass.

    from hysys import open_case
    from helpers import Surrogate
    from hysys_surrogate import cache_objects, run_point

    case    = open_case("C:/.../methanol_surrogate_no_rcy.hsc")
    objects = cache_objects(case, box)            # box: {name: (lo, hi)} in INPUTS order
    net     = Surrogate.load("pfr_surrogate_raw.txt")
    row     = run_point(objects, net, pressure=(90, "bar"), temperature=(250, "C"),
                        ratio=3.0, purge_rate=0.03)

The net is the "raw" surrogate from ``methanol_surrogate_pipeline.ipynb``:
inputs in `INPUTS` order (C, bar, kgmole/s), outputs the six outlet flows in
kgmole/h, duty in kW and pressure drop in bar.  The "stoi" surrogate is used
through ``stoi_as_raw(Surrogate.load("pfr_surrogate_stoi.txt"))``.  Both were
trained at a fixed reactor volume, so `run_point` takes no volume.  `box` is
the training box; inputs that leave it are reported per point.
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np

from hysys import read, write, by_component, to_internal, solve

__all__ = ["cache_objects", "reactor_step", "is_converged", "run_point", "stoi_as_raw",
           "INPUTS", "COMPONENTS", "STOICH"]

COMPONENTS = ["Hydrogen", "CO", "CO2", "H2O", "Methanol", "Nitrogen"]   # the ones that reach the reactor
INPUTS = ["temperature_C", "pressure_bar", *COMPONENTS]                 # net input order
STOICH = np.array([[-3, -1], [0, 1], [-1, -1], [1, 1], [1, 0], [0, 0]], float)   # rows COMPONENTS, cols xi1, xi2


def stoi_as_raw(net):
    """
    Wrap the "stoi" net (extents, duty, drop) so it returns what the "raw" net
    returns: the six outlet flows in kgmole/h, duty in kW, pressure drop in bar.
    """
    def wrapped(x):
        xi1, xi2, duty, dp = net(x)
        inlet = np.asarray(x[2:], float) * 3600.0                        # kgmole/s -> kgmole/h
        return np.concatenate([inlet + STOICH @ [xi1, xi2], [duty, dp]])
    return wrapped


def cache_objects(case, box: dict) -> dict:
    """
    Same lookups as ``hysys_methanol.cache_objects`` plus the training `box`
    (``{name: (lo, hi)}`` in `INPUTS` order); Reactor100 must be ignored in this case.
    """
    if list(box) != INPUTS:
        raise ValueError(f"box order {list(box)} differs from INPUTS {INPUTS}")
    fs = case.Flowsheet
    streams, ops = fs.MaterialStreams, fs.Operations
    rin = streams("RinV")
    if not ops("Reactor100").IsIgnored:
        raise RuntimeError("Reactor100 is active; this module expects the cut case with the reactor ignored")
    return {
        "solver": case.Solver,
        "units": case.Application.UnitConversionSetManager,
        "co2in": streams("co2_1atm"),
        "h2in": streams("h2"),
        "rin": rin,
        "rout": streams("RoutV"),
        "methanol": streams("Methanol"),
        "purge": streams("Purge"),
        "recycle_gas": streams("VAPToMixer"),
        "compressors": [ops("K-100"), ops("K-105")],
        "split": ops("TEE-100"),
        "column": ops("Twp101"),
        "flare": ops("CRV-100"),
        "pressure_cell": ops("MeOH Pressure").Cell("A1"),
        "ratio_cell": ops("co2:h2_ratio_equals_3").Cell("C2"),
        "components": list(rin.FluidPackage.Components.Names),          # order of component arrays
        "box": {k: tuple(v) for k, v in box.items()},
    }


def reactor_step(objects: dict, net, timeout: float = 120.0) -> dict:
    """
    One pass of the tear: read RinV, evaluate the net, write RoutV, solve.

    Returns the net input vector `x` (INPUTS order), the predicted outlet flows
    in kgmole/h, duty, pressure drop, net CO formation, and `outside_box`, the
    names of inputs outside the training box.
    """
    o = objects
    rin, rout = o["rin"], o["rout"]
    flows = by_component(rin, "ComponentMolarFlow", "kgmole/s")
    x = np.array([read(rin.Temperature, "C"), read(rin.Pressure, "bar"), *(flows[c] for c in COMPONENTS)])
    y = net(x)
    outlet = dict(zip(COMPONENTS, np.maximum(y[:6], 0.0)))              # kgmole/h; clip numerical negatives
    duty, dp = float(y[6]), float(y[7])

    o["solver"].CanSolve = False
    rout.ComponentMolarFlow.SetValues(tuple(float(outlet.get(c, 0.0)) for c in o["components"]), "kgmole/h")
    write(rout.Pressure, (x[1] - dp, "bar"))
    solve(o["solver"], timeout)
    outside = [k for k, v, (lo, hi) in zip(INPUTS, x, o["box"].values()) if not lo <= v <= hi]
    return {"x": x, "outlet_kgmole_h": outlet, "duty_kW": duty, "dP_bar": dp,
            "co_formation_kgmole_h": outlet["CO"] - flows["CO"] * 3600.0, "outside_box": outside}


def is_converged(new, old, tol: float = 1e-3) -> bool:
    """Max relative change over the tear vector below `tol`."""
    new, old = np.asarray(new, float), np.asarray(old, float)
    return float((np.abs(new - old) / (np.abs(new) + 1e-8)).max()) < tol


def run_point(objects: dict, net, *, pressure, temperature, ratio, purge_rate,
              tol: float = 1e-3, max_passes: int = 100, timeout: float = 120.0) -> dict:
    """
    Set the operating point, iterate the tear until `is_converged`, and return
    the same result row as ``hysys_methanol.run_point`` with the reactor
    quantities taken from the net, plus `tear_passes`, `tear_converged` and
    `outside_box` from the last pass.  No volume: the net was trained at one.
    """
    o = objects
    solver = o["solver"]
    t0 = time.time()

    solver.CanSolve = False
    o["column"].IsIgnored = True
    o["flare"].IsIgnored = True
    o["pressure_cell"].CellValue = to_internal(o["units"], "Pressure", pressure)
    write(o["rin"].Temperature, temperature)
    write(o["rout"].Temperature, temperature)       # isothermal reactor, as in the original case
    o["ratio_cell"].CellValue = float(ratio)
    o["split"].SplitsValue = (float(purge_rate), 1.0 - float(purge_rate))
    solve(solver, timeout)                          # propagate the new inputs to RinV

    old, tear_ok = None, False
    for passes in range(1, max_passes + 1):
        step = reactor_step(o, net, timeout)
        x = step["x"][2:]                           # the six RinV flows are the tear vector
        if old is not None and is_converged(x, old, tol):
            tear_ok = True
            break
        old = x
    o["column"].IsIgnored = False
    o["flare"].IsIgnored = False
    solve(solver, timeout)
    solver.CanSolve = True

    co2in, h2in, meoh, purge = o["co2in"], o["h2in"], o["methanol"], o["purge"]
    purge_kg_h = by_component(purge, "ComponentMassFlow", "kg/h")
    column_ok = bool(o["column"].ColumnFlowsheet.CfsConverged)
    return {
        "pressure_bar": read(o["rin"].Pressure, "bar"),
        "temperature_C": read(o["rin"].Temperature, "C"),
        "ratio": float(ratio),
        "purge_rate": float(purge_rate),
        "methanol_kg_h": read(meoh.MassFlow, "kg/h"),
        "hydrogen_purge_kg_h": purge_kg_h["Hydrogen"],
        "co2_purge_kg_h": purge_kg_h["CO2"],
        "co2_input_kg_h": read(co2in.MassFlow, "kg/h"),
        "h2_input_kg_h": read(h2in.MassFlow, "kg/h"),
        "reactor_duty_kW": step["duty_kW"],
        "reactor_dP_bar": step["dP_bar"],
        "co_formation_kgmole_h": step["co_formation_kgmole_h"],
        "compression_kW": sum(read(k.EnergyStream.HeatFlow, "kW") for k in o["compressors"]),
        "carbon_efficiency": (by_component(meoh, "ComponentMolarFlow", "kgmole/h")["Methanol"]
                              / by_component(co2in, "ComponentMolarFlow", "kgmole/h")["CO2"]),
        "recycle_ratio": (read(o["recycle_gas"].MolarFlow, "kgmole/h")
                          / (read(co2in.MolarFlow, "kgmole/h") + read(h2in.MolarFlow, "kgmole/h"))),
        "tear_passes": passes,
        "tear_converged": tear_ok,
        "column_converged": column_ok,
        "outside_box": step["outside_box"],
        "converged": tear_ok and column_ok,
        "solve_time_s": round(time.time() - t0, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
