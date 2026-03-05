# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required( VERSION 3.15 )
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc /opt/rocm/opencl)

configure_file("${PROJECT_SOURCE_DIR}/src/include/config.h.in" "${PROJECT_BINARY_DIR}/src/include/config.h")

include_directories(include "${PROJECT_BINARY_DIR}/src/include")
add_executable(fin main.cpp fin.cpp base64.cpp)
target_compile_definitions( fin PRIVATE -D__HIP_PLATFORM_HCC__=1 )
target_link_libraries(fin MIOpen hip::host)
target_link_libraries(fin ${CMAKE_THREAD_LIBS_INIT})
if(rocblas_FOUND)
    target_link_libraries( fin $<BUILD_INTERFACE:roc::rocblas> )
    # target_link_libraries( fin PRIVATE roc::rocblas )
endif()
if(HAS_LIB_STD_FILESYSTEM)
    target_link_libraries(fin stdc++fs)
endif()

# Cmake does not add flags correctly for gcc
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
    set_target_properties(fin PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)
endif()
install(TARGETS fin
    OPTIONAL 
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
