"""
Methanol synthesis loop: which flowsheet objects to cache and how one operating
point is run.  Generic helpers live in ``hysys.py``.

    from hysys import open_case
    from hysys_methanol import cache_objects, run_point

    case    = open_case(r"C:\\...\\methanol.hsc")
    objects = cache_objects(case)
    row     = run_point(objects, pressure=(90, "bar"), temperature=(250, "C"),
                        volume=(25, "m3"), ratio=3.0, purge_rate=0.05)

Dimensioned inputs are ``(number, "unit")`` pairs; ratio and purge_rate are
dimensionless.  Every result key names its unit, or has none if dimensionless.
"""
from __future__ import annotations

import time
from datetime import datetime

from hysys import read, write, by_component, to_internal, solve

__all__ = ["cache_objects", "run_point"]

# Recycle composition tolerance multiplier (HYSYS default 10).
RECYCLE_SENSITIVITY = 0.1


def set_segments(reactor, segments: int) -> None:
    """Set the PFR's number of integration segments and verify it took."""
    segments = int(segments)
    if segments < 1:
        raise ValueError(f"segments must be positive, got {segments}")
    reactor.NumberOfSegments = segments
    if int(reactor.NumberOfSegments) != segments:
        raise RuntimeError(f"{reactor.Name}: NumberOfSegments is {reactor.NumberOfSegments}, wanted {segments}")


def cache_objects(case, *, segments: int) -> dict:
    """
    Look up streams, unit operations and cells used by `run_point` once per
    session, tighten the recycle tolerance to `RECYCLE_SENSITIVITY` and set the
    PFR discretisation to `segments` (pass the same value to ``hysys_pfr``).
    """
    fs = case.Flowsheet
    streams, ops = fs.MaterialStreams, fs.Operations
    set_segments(ops("Reactor100"), segments)
    recycle = ops("RCY-1")
    recycle.CompSensitivityValue = RECYCLE_SENSITIVITY
    recycle.ComponentSensitivityValue = tuple(RECYCLE_SENSITIVITY for _ in recycle.ComponentSensitivityValue)
    return {
        "solver": case.Solver,
        "units": case.Application.UnitConversionSetManager,
        "co2in": streams("co2_1atm"),
        "h2in": streams("h2"),
        "rin": streams("RinV"),                     # reactor inlet
        "rout": streams("RoutV"),                   # reactor outlet
        "methanol": streams("Methanol"),
        "purge": streams("Purge"),
        "recycle_gas": streams("VAPToMixer"),       # recycle after compression, into MIX-101
        "reactor": ops("Reactor100"),
        "compressors": [ops("K-100"), ops("K-105")],   # fresh syngas 30 bar -> P, recycle; the only P-dependent ones
        "split": ops("TEE-100"),                    # purge / recycle tee: (purge, recycle)
        "column": ops("Twp101"),
        "flare": ops("CRV-100"),
        "recycle": recycle,
        "segments": int(segments),
        # spreadsheet cells fan one number out to several specs; cells are unitless to COM.
        "pressure_cell": ops("MeOH Pressure").Cell("A1"),           # exported to kPa specs
        "ratio_cell": ops("co2:h2_ratio_equals_3").Cell("C2"),     # H2 : CO2 in the fresh feed
    }


def run_point(objects: dict, *, pressure, temperature, volume, ratio, purge_rate,
              timeout: float = 120.0) -> dict:
    """
    Set the operating point, solve, and return one flat result row.

    Freezes the solver, ignores the column and flare so the synthesis loop
    converges on its own, applies the inputs, solves, restores the column and
    flare, solves again.  Filter rows on ``converged`` before trusting them.
    """
    o = objects
    solver = o["solver"]
    t0 = time.time()

    solver.CanSolve = False
    o["column"].IsIgnored = True
    o["flare"].IsIgnored = True
    o["pressure_cell"].CellValue = to_internal(o["units"], "Pressure", pressure)
    write(o["rin"].Temperature, temperature)
    write(o["rout"].Temperature, temperature)       # isothermal reactor, duty is a result
    write(o["reactor"].TotalVolume, volume)
    o["ratio_cell"].CellValue = float(ratio)
    o["split"].SplitsValue = (float(purge_rate), 1.0 - float(purge_rate))
    solve(solver, timeout)
    o["column"].IsIgnored = False
    o["flare"].IsIgnored = False
    solve(solver, timeout)
    solver.CanSolve = True                          # leave HYSYS live for inspection

    co2in, h2in, meoh, purge, reactor = o["co2in"], o["h2in"], o["methanol"], o["purge"], o["reactor"]
    purge_kg_h = by_component(purge, "ComponentMassFlow", "kg/h")
    inlet = by_component(o["rin"], "ComponentMolarFlow", "kgmole/h")
    co_in = inlet["CO"]
    co_out = by_component(o["rout"], "ComponentMolarFlow", "kgmole/h")["CO"]
    recycle_ok = o["recycle"].RecycleConvergence == 1
    column_ok = bool(o["column"].ColumnFlowsheet.CfsConverged)
    return {
        "pressure_bar": read(o["rin"].Pressure, "bar"),
        "temperature_C": read(o["rin"].Temperature, "C"),
        "volume_m3": read(reactor.TotalVolume, "m3"),
        "ratio": float(ratio),
        "purge_rate": float(purge_rate),
        "methanol_kg_h": read(meoh.MassFlow, "kg/h"),
        "hydrogen_purge_kg_h": purge_kg_h["Hydrogen"],
        "co2_purge_kg_h": purge_kg_h["CO2"],
        "co2_input_kg_h": read(co2in.MassFlow, "kg/h"),
        "h2_input_kg_h": read(h2in.MassFlow, "kg/h"),
        **{f"in_{c}_kgmole_h": v for c, v in inlet.items()},       # reactor inlet, per component
        "reactor_duty_kW": read(reactor.HeatFlow, "kW"),
        "reactor_dP_bar": read(reactor.PressureDrop, "bar"),
        "co_formation_kgmole_h": co_out - co_in,    # net CO made per pass (reverse water-gas shift)
        "compression_kW": sum(read(k.EnergyStream.HeatFlow, "kW") for k in o["compressors"]),
        "carbon_efficiency": (by_component(meoh, "ComponentMolarFlow", "kgmole/h")["Methanol"]
                              / by_component(co2in, "ComponentMolarFlow", "kgmole/h")["CO2"]),
        "recycle_ratio": (read(o["recycle_gas"].MolarFlow, "kgmole/h")
                          / (read(co2in.MolarFlow, "kgmole/h") + read(h2in.MolarFlow, "kgmole/h"))),
        "segments": o["segments"],
        "recycle_converged": recycle_ok,
        "column_converged": column_ok,
        "recycle_iterations": int(o["recycle"].IterationsValue),
        "converged": recycle_ok and column_ok,
        "solve_time_s": round(time.time() - t0, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
