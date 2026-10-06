# Native Windows (MSVC) build of the MC++ v4 Python module "pymc" and a reduced
# MAGNUS module "magnus" (BaseSampling, BaseFeas, BaseMBDOE, NSFeas).
# Normally driven by setup.py in this folder. Expected layout under OMEGA_WORK_DIR:
#   mcpp/      clone of omega-icl/mcpp at the pinned commit, with submodules
#   magnus/    clone of omega-icl/magnus at the pinned commit, patch applied
#   deps/openblas/      prebuilt OpenBLAS x64 (bin/libopenblas.dll, lib/libopenblas.lib)
#   deps/boost_1_90_0/  Boost headers (subfolder boost/)
#   deps/armadillo/     Armadillo headers (subfolder include/)
#   deps/canon/base_opt.hpp   single header from omega-icl/canon
cmake_minimum_required(VERSION 3.20)
project(omega_win LANGUAGES CXX)

set(HERE "${CMAKE_CURRENT_SOURCE_DIR}")
set(OMEGA_WORK_DIR "${HERE}/work" CACHE PATH "Folder holding the clones and dependencies")
set(WORK "${OMEGA_WORK_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
# Same as the upstream make build: -O2 with asserts enabled (no /DNDEBUG).
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2")

set(PYBIND11_FINDPYTHON ON)
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)
add_subdirectory("${WORK}/mcpp/extern/pybind11" pybind11)

add_library(omega_common INTERFACE)
target_include_directories(omega_common INTERFACE
  "${WORK}/mcpp/src/mc"
  "${WORK}/mcpp/src/3rdparty/fadbad++"
  "${WORK}/mcpp/src/3rdparty/boost"
  "${WORK}/deps/boost_1_90_0"
  "${WORK}/deps/armadillo/include")
# FLAG_MC of mcpp/src/makeoptions.mk (without HSL) plus Windows-specific defines.
target_compile_definitions(omega_common INTERFACE
  MC__USE_THREAD MC__USE_TADIFF MC__USE_ARMADILLO MC__USE_BOOST BOOST_UBLAS_NO_STD_CERR
  NOMINMAX ARMA_DONT_USE_WRAPPER)
# /fp:strict lets MSVC accept the constant 0./0. (NaN) in ffunc.hpp, so no MC++ source patch is needed.
target_compile_options(omega_common INTERFACE /bigobj /EHsc /utf-8 /W3 /fp:strict)

# pymc: the object list of mcpp/src/pymc/pymc.mk, sources unchanged.
set(PYMC_SRC mcfunc interval mccormick supmodel ffunc fflin ffvect ffcustom pymc)
list(TRANSFORM PYMC_SRC PREPEND "${WORK}/mcpp/src/pymc/")
list(TRANSFORM PYMC_SRC APPEND ".cpp")
pybind11_add_module(pymc MODULE NO_EXTRAS ${PYMC_SRC})
target_link_libraries(pymc PRIVATE omega_common)

# magnus: base.cpp and nsfeas.cpp unchanged, plus the reduced module init.
pybind11_add_module(magnus MODULE NO_EXTRAS
  "${WORK}/magnus/src/interface/base.cpp"
  "${WORK}/magnus/src/interface/nsfeas.cpp"
  "${HERE}/win/magnus_nsfeas_only.cpp")
target_include_directories(magnus PRIVATE "${WORK}/magnus/src" "${WORK}/deps/canon")
target_link_libraries(magnus PRIVATE omega_common "${WORK}/deps/openblas/lib/libopenblas.lib")

# "cmake --install <build> --prefix <dir>" puts the three files directly into <dir>.
install(TARGETS pymc magnus LIBRARY DESTINATION . RUNTIME DESTINATION .)
install(FILES "${WORK}/deps/openblas/bin/libopenblas.dll" DESTINATION .)
