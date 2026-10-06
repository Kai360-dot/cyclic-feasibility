import sys
import pymc
import magnus
print("python", sys.version)
print("pymc   ", pymc.__file__, getattr(pymc, "__version__", "?"))
print("magnus ", magnus.__file__)
print("magnus doc:", magnus.__doc__)
print("magnus names:", sorted(n for n in dir(magnus) if not n.startswith("_")))
ns = magnus.NSFeas()
print("NSFeas options: NUMLIVE", ns.options.NUMLIVE, "NUMPROP", ns.options.NUMPROP, "MAXTHREAD", ns.options.MAXTHREAD)
dag = pymc.FFGraph()
x = dag.add_var("x"); y = dag.add_var("y")
f = pymc.exp(x) * y + x / y if hasattr(pymc, "exp") else x * y + x / y
print("FFVar expr:", f)
print("SMOKE OK")
