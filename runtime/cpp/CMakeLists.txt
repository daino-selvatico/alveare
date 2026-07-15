cmake_minimum_required(VERSION 3.17)
project(alveare_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Shared runtime sources (everything except the per-binary main()).
set(ALVEARE_CORE_SRC
    src/bf16.cpp
    src/npy.cpp
    src/config.cpp
    src/npu.cpp
)

add_executable(alveare_runtime src/main.cpp ${ALVEARE_CORE_SRC})
target_include_directories(alveare_runtime PRIVATE ${XRT_INCLUDE_DIRS})
target_link_libraries(alveare_runtime PRIVATE ${XRT_COREUTIL_TARGET} pthread)

# NPU registry smoke test: runs a harvested GEMV shape and checks parity.
add_executable(npu_smoke test/npu_smoke.cpp ${ALVEARE_CORE_SRC})
target_include_directories(npu_smoke PRIVATE ${XRT_INCLUDE_DIRS})
target_link_libraries(npu_smoke PRIVATE ${XRT_COREUTIL_TARGET} pthread)
