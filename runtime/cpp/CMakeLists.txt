cmake_minimum_required(VERSION 3.17)
project(alveare_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure XRT is found. Usually installed in /opt/xilinx/xrt
list(APPEND CMAKE_PREFIX_PATH "/opt/xilinx/xrt")
find_package(XRT REQUIRED)

include_directories(include)

# We will add other source files as we create them
add_executable(alveare_runtime
    src/main.cpp
    src/bf16.cpp
    src/npy.cpp
    src/config.cpp
)

# Link against XRT
target_include_directories(alveare_runtime PRIVATE ${XRT_INCLUDE_DIRS})
target_link_libraries(alveare_runtime PRIVATE xrt_coreutil pthread)
