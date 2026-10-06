# A Python exception raised inside a pymc.FFCustom callback while magnus.NSFeas
# evaluates the DAG must surface as an error, not crash the interpreter.
import pymc
from magnus import NSFeas

calls = {"n": 0}

def model(x):
    calls["n"] += 1
    if calls["n"] == 700:
        raise ValueError("deliberate failure in Python callback")
    return [x[0] ** 2 + x[1]]

DAG = pymc.FFGraph()
DAG.options.MAXTHREAD = 1
d = [DAG.add_var("d0"), DAG.add_var("d1")]
op = pymc.FFCustom()
op.set_D_eval(model)
y = op(ndep=1, var=d, uid=0)[0]

NS = NSFeas()
NS.options.MAXTHREAD = 1
NS.options.DISPLEVEL = 0
NS.options.NUMLIVE = 256
NS.set_dag(DAG)
NS.set_control(d, [-1, -1], [1, 1])
NS.set_constraint([y - 0.75, 0.20 - y])
NS.setup()
try:
    flag = NS.sample()
    print("sample() returned", flag, "after", calls["n"], "callback calls")
except Exception as e:
    print("sample() raised", type(e).__name__, ":", e, "after", calls["n"], "callback calls")
NS.stats.display()
live = NS.live_points[0]
print("live points:", live.shape)
print("V3 DONE (interpreter alive)")
