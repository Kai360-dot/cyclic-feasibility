"""
Generic helpers for driving Aspen HYSYS cases from Python via COM.

Every number that crosses the COM boundary carries an explicit unit:

    write(stream.Temperature, (250, "C"))          # set
    read(stream.Pressure, "bar")                   # get
    by_component(stream, "ComponentMassFlow", "kg/h")   # {"Hydrogen": ..., ...}

Flowsheet-specific modules (``hysys_methanol``, ``hysys_pfr``) build on these
and expose ``cache_objects(case)`` and ``run_point(objects, ...)``.
"""
from __future__ import annotations

import time

import pythoncom
import win32com.client

__all__ = ["open_case", "solve", "read", "write", "by_component", "to_internal",
           "unit_of", "members", "dump"]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def open_case(path: str, visible: bool = True):
    """
    Return the HYSYS case at `path`, attaching if it is already open.

    HYSYS registers each open case in the Windows Running Object Table under
    its file path.  If an entry matches, attach; otherwise launch HYSYS and
    open the file.  `path` must be absolute; the comparison is case-insensitive.
    """
    pythoncom.CoInitialize()
    rot = pythoncom.GetRunningObjectTable()
    ctx = pythoncom.CreateBindCtx(0)
    enum = rot.EnumRunning()
    while (found := enum.Next(1)):
        moniker = found[0]
        try:
            if moniker.GetDisplayName(ctx, None).lower() != path.lower():
                continue
            obj = rot.GetObject(moniker).QueryInterface(pythoncom.IID_IDispatch)
            return win32com.client.Dispatch(obj)
        except pythoncom.com_error:
            continue

    print("Case not open in HYSYS - launching (this can take a minute)...")
    app = win32com.client.Dispatch("HYSYS.Application.NewInstance")
    app.Visible = visible
    return app.SimulationCases.Open(path)


def solve(solver, timeout: float) -> None:
    """Release the solver and block until it has finished forgetting and solving."""
    start = time.time()
    solver.CanSolve = True
    while solver.IsForgetting or solver.IsSolving:
        if time.time() - start > timeout:
            solver.CanSolve = False
            raise TimeoutError(f"HYSYS did not converge within {timeout:.0f} s")
        time.sleep(0.1)
    solver.CanSolve = False


# ---------------------------------------------------------------------------
# Units: every value is a (number, "unit") pair on the way in, and every read
# names its unit.  Conversion is done by HYSYS; unknown unit names raise.
# ---------------------------------------------------------------------------
def _pair(value) -> tuple[float, str]:
    """Validate a ``(number, "unit")`` pair; numpy scalars are cast to float."""
    try:
        number, unit = value
        return float(number), str(unit)
    except (TypeError, ValueError):
        raise TypeError(f"expected (number, 'unit'), got {value!r}") from None


def write(variable, value) -> None:
    """Set a HYSYS variable: ``write(stream.Temperature, (250, "C"))``."""
    variable.SetValue(*_pair(value))


def read(variable, unit: str) -> float:
    """Read a HYSYS variable in `unit`: ``read(stream.Pressure, "bar")``."""
    return float(variable.GetValue(unit))


def by_component(stream, prop: str, unit: str) -> dict[str, float]:
    """
    Read a per-component stream array keyed by component name, e.g.
    ``by_component(stream, "ComponentMolarFlow", "kgmole/h")``.
    Use ``unit=""`` for dimensionless arrays such as ``ComponentMolarFraction``.
    """
    names = stream.FluidPackage.Components.Names
    return dict(zip(names, getattr(stream, prop).GetValues(unit)))


def to_internal(units, quantity: str, value) -> float:
    """
    Convert a ``(number, "unit")`` pair to HYSYS's calculation unit for
    `quantity` (a unit-set name such as "Pressure").  Needed only for targets
    that have no unit of their own, such as spreadsheet cells.
    """
    number, unit = _pair(value)
    return units.Item(quantity).Item(unit).ToCalculationUnit(number)


def unit_of(units, variable) -> tuple[str, str]:
    """Return ``(calculation_unit, display_unit)`` for a HYSYS variable object."""
    uset = units.Item(variable.UnitConversionType)
    return uset.CalculationUnit.name, uset.CurrentDisplayUnit.name


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
_COM_PLUMBING = {"AddRef", "Release", "QueryInterface", "GetTypeInfo",
                 "GetTypeInfoCount", "GetIDsOfNames", "Invoke"}


def members(obj, contains: str | None = None) -> dict[str, list[str]]:
    """
    List the members of a HYSYS COM object, grouped by kind.

    ``dir()`` shows nothing for late-bound COM proxies; this reads the type
    library instead.  Returns ``{"read-write": [...], "read-only": [...],
    "write-only": [...], "methods": [...]}``.  `contains` filters names
    case-insensitively.
    """
    ti = obj._oleobj_.GetTypeInfo()
    kinds: dict[str, int] = {}
    for i in range(ti.GetTypeAttr().cFuncs):
        fd = ti.GetFuncDesc(i)
        name = ti.GetNames(fd.memid)[0]
        if name.startswith("_") or name in _COM_PLUMBING:
            continue
        kinds[name] = kinds.get(name, 0) | fd.invkind

    out = {"read-write": [], "read-only": [], "write-only": [], "methods": []}
    for name, k in sorted(kinds.items()):
        if contains and contains.lower() not in name.lower():
            continue
        get = bool(k & pythoncom.INVOKE_PROPERTYGET)
        put = bool(k & (pythoncom.INVOKE_PROPERTYPUT | pythoncom.INVOKE_PROPERTYPUTREF))
        if k & pythoncom.INVOKE_FUNC and not (get or put):
            out["methods"].append(name)
        else:
            out["read-write" if get and put else "read-only" if get else "write-only"].append(name)
    return out


def _fmt(value, width: int = 60) -> str:
    """One-line rendering of a COM property value."""
    if isinstance(value, (int, float, str, bool)) or value is None:
        s = repr(value)
    elif isinstance(value, tuple):
        s = f"tuple[{len(value)}] {value[:6]!r}{' ...' if len(value) > 6 else ''}"
    else:
        try:
            s = f"<{value.Name}>"
        except Exception:
            s = "<object>"
    return s if len(s) <= width else s[: width - 4] + " ..."


def dump(obj, contains: str | None = None) -> None:
    """
    Print every readable property of a HYSYS COM object with its current value.

    ``...Value`` properties are annotated with their calculation unit, read
    from the sibling variable object (``MassFlowValue`` -> ``MassFlow``).
    Properties that raise for the object's current state print as ``<error>``.
    """
    try:
        title = f"{obj.Name} ({obj.TypeName})"
    except Exception:
        title = "object"
    print(f"=== {title} ===")
    m = members(obj, contains)
    readable = set(members(obj)["read-write"]) | set(members(obj)["read-only"])
    try:
        units = obj.Application.UnitConversionSetManager
    except Exception:
        units = None

    def unit_for(name: str) -> str:
        base = name[:-5]
        if units is None or not name.endswith("Value") or base not in readable:
            return ""
        try:
            return units.Item(getattr(obj, base).UnitConversionType).CalculationUnit.name
        except Exception:
            return ""

    for kind in ("read-write", "read-only"):
        if m[kind]:
            print(f"\n[{kind}]")
        for name in m[kind]:
            try:
                print(f"  {name:36s} {_fmt(getattr(obj, name))} {unit_for(name)}".rstrip())
            except Exception:
                print(f"  {name:36s} <error>")
    if m["methods"]:
        print("\n[methods]")
        for name in m["methods"]:
            print(f"  {name}()")
