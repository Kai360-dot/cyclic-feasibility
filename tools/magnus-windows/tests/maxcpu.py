"""MAXCPU (wall-time limit) check: with a 0.1 s limit and a callback that costs
about 0.2 ms, the sampler must run for roughly 0.1 s and then report an
interruption (status 2); it must not stop at iteration 0."""
import time
import pymc
from magnus import NSFeas

def model(x):
    t = time.perf_counter()
    while time.perf_counter() - t < 2e-4:
        pass
    return [x[0] ** 2 + x[1]]

DAG = pymc.FFGraph(); DAG.options.MAXTHREAD = 1
d = [DAG.add_var("d0"), DAG.add_var("d1")]
op = pymc.FFCustom(); op.set_D_eval(model)
y = op(ndep=1, var=d, uid=0)[0]
NS = NSFeas()
NS.options.MAXTHREAD = 1; NS.options.DISPLEVEL = 0
NS.options.NUMLIVE = 400; NS.options.NUMPROP = 16
NS.options.MAXCPU = 0.1
NS.set_dag(DAG); NS.set_control(d, [-1, -1], [1, 1]); NS.set_constraint([y - 0.75, 0.20 - y])
NS.setup()
t0 = time.perf_counter(); flag = NS.sample(); dt = time.perf_counter() - t0
print(f"status {flag}, iterations {NS.stats.iter}, dead points {len(NS.dead_points[0])}, wall {dt:.2f} s")
