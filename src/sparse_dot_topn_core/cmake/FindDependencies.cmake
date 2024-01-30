# -- Python
find_package(
  Python 3.8 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# -- Nanobind
find_package(nanobind CONFIG REQUIRED)

# -- Eigen
find_package(Eigen3 NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  include(GetEigen)
  set(Eigen3_DIR "${STDN_EIGEN_TARGET_DIR}/eigen/cmake")
  find_package(Eigen3 NO_MODULE REQUIRED)
endif()

# -- OpenMP
if(NOT SDTN_DISABLE_OPENMP)
  message(STATUS "sparse-dot-topn | OpenMP disabled: OFF")
  find_package(OpenMP)
  if ((NOT OpenMP_FOUND) AND APPLE)
    include(SetHomebrew)
    set(OpenMP_ROOT ${HOMEBREW_PREFIX}/opt/libomp)
    find_package(OpenMP)
  endif()
  if (NOT OpenMP_FOUND AND SDTN_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
  endif()
else()
  set(SDTN_ENABLE_OPENMP OFF)
  message(STATUS "sparse-dot-topn | OpenMP disabled: ON")
endif()
