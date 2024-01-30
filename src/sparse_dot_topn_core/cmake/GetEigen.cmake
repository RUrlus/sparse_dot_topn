include(FetchContent)

set(SDTN_EIGEN_DEFAULT_VERSION 3.4.0)
if (NOT SDTN_EIGEN_VERSION)
    message(STATUS "sdtn: Setting Eigen version to 'v${SDTN_EIGEN_DEFAULT_VERSION}' as none was specified.")
    set(SDTN_EIGEN_VERSION "${SDTN_EIGEN_DEFAULT_VERSION}" CACHE STRING "Choose the version of Eigen." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE SDTN_EIGEN_VERSION PROPERTY STRINGS "3.4.0")
ENDIF ()

set(STDN_EIGEN_TARGET_DIR ${PROJECT_SOURCE_DIR}/src/sparse_dot_topn_core/extern)

message(STATUS "sdtn: collecting eigen v${SDTN_EIGEN_VERSION} at ${STDN_EIGEN_TARGET_DIR}")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG        ${SDTN_EIGEN_VERSION}
    SOURCE_DIR     ${STDN_EIGEN_TARGET_DIR}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    EXCLUDE_FROM_ALL
)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)
FetchContent_MakeAvailable(Eigen)

list(POP_BACK CMAKE_MESSAGE_INDENT)
set(EIGEN3_ROOT_DIR "${STDN_EIGEN_TARGET_DIR}/eigen")
