target_include_directories(_sparse_dot_topn_core PUBLIC ${SDTN_INCLUDE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(_sparse_dot_topn_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(_sparse_dot_topn_core PRIVATE SDTN_OMP_ENABLED=TRUE)
endif()

target_compile_definitions(_sparse_dot_topn_core PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION})

# -- Optional
if(SDTN_ENABLE_DEVMODE)
    target_compile_options(_sparse_dot_topn_core PRIVATE ${SDTN_DEVMODE_OPTIONS})
endif()

# -- Options & Properties
set_property(TARGET _sparse_dot_topn_core PROPERTY CXX_STANDARD ${SDTN_CPP_STANDARD})
set_property(TARGET _sparse_dot_topn_core PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET _sparse_dot_topn_core PROPERTY POSITION_INDEPENDENT_CODE ON)

# -- Compiler Flags
set(SDTN_ARCHITECTURE_FLAGS "")
if (SDTN_ENABLE_ARCH_FLAGS)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native ARCH_NATIVE_SUPPORTED)
    if(ARCH_NATIVE_SUPPORTED)
        set(SDTN_ARCHITECTURE_FLAGS ${SDTN_ARCHITECTURE_FLAGS} -march=native)
    endif()
    check_cxx_compiler_flag(-mtune=native TUNE_NATIVE_SUPPORTED)
    if(TUNE_NATIVE_SUPPORTED)
        set(SDTN_ARCHITECTURE_FLAGS ${SDTN_ARCHITECTURE_FLAGS} -mtune=native)
    endif()
    # check_cxx_compiler_flag(-msse2 SSE2_SUPPORTED)
    # if(SSE2_SUPPORTED)
    #     set(SDTN_ARCHITECTURE_FLAGS ${SDTN_ARCHITECTURE_FLAGS} -msse2)
    # endif()
    # check_cxx_compiler_flag(-msse4 SSE4_SUPPORTED)
    # if(SSE4_SUPPORTED)
    #     set(SDTN_ARCHITECTURE_FLAGS ${SDTN_ARCHITECTURE_FLAGS} -msse4)
    # endif()
    # check_cxx_compiler_flag(-mavx AVX_SUPPORTED)
    # if(AVX_SUPPORTED)
    #     set(SDTN_ARCHITECTURE_FLAGS ${SDTN_ARCHITECTURE_FLAGS} -mavx)
    # endif()
endif()
target_compile_options(_sparse_dot_topn_core PRIVATE "$<$<CONFIG:RELEASE>:${SDTN_ARCHITECTURE_FLAGS}>")