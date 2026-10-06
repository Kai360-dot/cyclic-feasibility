// Windows subset of the MAGNUS Python interface.
// Replaces src/interface/magnus.cpp, which requires the cronos and canon modules.
// Registers only the base classes and the nested-sampling feasibility solver.
#include <pybind11/pybind11.h>

namespace py = pybind11;

void mc_base(py::module_&);
void mc_nsfeas(py::module_&);

PYBIND11_MODULE(magnus, m)
{
  // FFGraph, FFVar etc. are registered by pymc; let an ImportError propagate.
  py::module_::import("pymc");

  m.doc() = "Python interface of library MAGNUS (Windows subset: BaseSampling, BaseFeas, BaseMBDOE, NSFeas)";

  mc_base(m);
  mc_nsfeas(m);
}
