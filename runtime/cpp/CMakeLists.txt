cmake_minimum_required(VERSION 3.17)
project(alveare_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build. Without this CMAKE_BUILD_TYPE is empty, i.e. the
# whole runtime compiled at -O0 — which mattered a lot: the host-side Q4 dot products
# (Gemma-4-E4B's per-layer PLE injection is ~200 ms/token, ~26% of decode) are scalar
# C++ and were left unvectorized/uninlined.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set — defaulting to Release (-O3)")
endif()

# Locate XRT. Prefer the packaged CMake config (e.g. /opt/xilinx/xrt on a
# Xilinx-style install); fall back to a system install that ships only the
# headers + libs (Ubuntu XRT packages put them under /usr, with no XRTConfig).
list(APPEND CMAKE_PREFIX_PATH "/opt/xilinx/xrt" "$ENV{XILINX_XRT}")
find_package(XRT QUIET)
if(NOT XRT_FOUND)
    find_path(XRT_INCLUDE_DIRS xrt/xrt_device.h
        PATHS /usr/include /opt/xilinx/xrt/include "$ENV{XILINX_XRT}/include")
    find_library(XRT_COREUTIL_LIB xrt_coreutil
        PATHS /usr/lib/x86_64-linux-gnu /opt/xilinx/xrt/lib "$ENV{XILINX_XRT}/lib")
    if(NOT XRT_INCLUDE_DIRS OR NOT XRT_COREUTIL_LIB)
        message(FATAL_ERROR "XRT not found: no XRTConfig.cmake and no system xrt headers/libs")
    endif()
    message(STATUS "XRT (system): includes=${XRT_INCLUDE_DIRS} lib=${XRT_COREUTIL_LIB}")
    set(XRT_COREUTIL_TARGET ${XRT_COREUTIL_LIB})
else()
    set(XRT_COREUTIL_TARGET xrt_coreutil)
endif()

# Project headers plus vendored single-header deps (nlohmann/json).
include_directories(include third_party)

# OpenMP parallelizes the host-side decode work (the PLE Q4 dot products, per-layer
# norms, etc.). Optional — the #pragma omp loops are no-ops without it.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(ALVEARE_OMP OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for host decode parallelism")
endif()

# Shared runtime sources (everything except the per-binary main()).
set(ALVEARE_CORE_SRC
    src/bf16.cpp
    src/npy.cpp
    src/config.cpp
    src/npu.cpp
    src/weights.cpp
    src/model.cpp
    src/tokenizer.cpp
    src/gemma_tokenizer.cpp
    src/prompt_lookup.cpp
    src/generator.cpp
    src/server.cpp
)

add_executable(alveare_runtime src/main.cpp ${ALVEARE_CORE_SRC})
target_include_directories(alveare_runtime PRIVATE ${XRT_INCLUDE_DIRS})
target_link_libraries(alveare_runtime PRIVATE ${XRT_COREUTIL_TARGET} pthread ${ALVEARE_OMP})

# NPU registry smoke test: runs a harvested GEMV shape and checks parity.
add_executable(parity_test test/parity_test.cpp ${ALVEARE_CORE_SRC})
target_include_directories(parity_test PRIVATE ${XRT_INCLUDE_DIRS})
target_link_libraries(parity_test PRIVATE ${XRT_COREUTIL_TARGET} pthread ${ALVEARE_OMP})

# Layer decode test
add_executable(layer_test test/layer_test.cpp ${ALVEARE_CORE_SRC})
target_include_directories(layer_test PRIVATE ${XRT_INCLUDE_DIRS})
target_link_libraries(layer_test PRIVATE ${XRT_COREUTIL_TARGET} pthread ${ALVEARE_OMP})

# Gemma-4-E4B oracle test
add_executable(e4b_test test/e4b_test.cpp ${ALVEARE_CORE_SRC})
target_include_directories(e4b_test PRIVATE ${XRT_INCLUDE_DIRS})
target_link_libraries(e4b_test PRIVATE ${XRT_COREUTIL_TARGET} pthread ${ALVEARE_OMP})

# Micro-benchmark: NPU kernel context-switch cost (see test/bench_switch.cpp)
add_executable(bench_switch test/bench_switch.cpp ${ALVEARE_CORE_SRC})
target_include_directories(bench_switch PRIVATE ${XRT_INCLUDE_DIRS})
target_link_libraries(bench_switch PRIVATE ${XRT_COREUTIL_TARGET} pthread ${ALVEARE_OMP})

# Statistical kernel benchmark (median + spread) — see test/bench_stat.cpp
add_executable(bench_stat test/bench_stat.cpp ${ALVEARE_CORE_SRC})
target_include_directories(bench_stat PRIVATE ${XRT_INCLUDE_DIRS})
target_link_libraries(bench_stat PRIVATE ${XRT_COREUTIL_TARGET} pthread ${ALVEARE_OMP})
