"""Demo of the NSFeas misalignment bug (single-threaded mode).

The Python callback fails deterministically for x0 > 0.8. After sampling, the
criterion value stored with every live, dead and discarded point is recomputed
from the point's own coordinates. With the bug, values of later proposals are
attached to earlier points whenever a proposal in the same batch failed."""
import numpy as np
import pymc
from magnus import NSFeas

def g_of(x):
    y = x[0] ** 2 + x[1]
    return max(y - 0.75, 0.20 - y)

def model(x):
    if x[0] > 0.8:
        raise ValueError("deliberate failure region")
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
NS.options.NUMLIVE = 300
NS.options.NUMPROP = 16
NS.set_dag(DAG)
NS.set_control(d, [-1, -1], [1, 1])
NS.set_constraint([y - 0.75, 0.20 - y])
NS.setup()
flag = NS.sample()
NS.stats.display()

bad_total = 0
for name, pts in (("live", NS.live_points), ("dead", NS.dead_points), ("discard", NS.discard_points)):
    X, val = np.asarray(pts[0]), np.asarray(pts[1]).ravel()
    in_fail_region = int(np.sum(X[:, 0] > 0.8))
    recomputed = np.array([g_of(x) for x in X])
    mismatch = np.abs(recomputed - val) > 1e-9
    bad_total += int(mismatch.sum())
    print(f"{name:8s}: {len(X):6d} points, {in_fail_region:4d} lie in the failure region, "
          f"{int(mismatch.sum()):4d} carry a criterion value that does not belong to them")
print("RESULT:", "BUG PRESENT" if bad_total else "CONSISTENT", f"(status {flag})")
