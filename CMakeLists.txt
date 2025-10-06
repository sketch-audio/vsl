cmake_minimum_required(VERSION 3.30)

option(VSL_BUILD_TESTS "Build tests" ON)

project(vsl)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# VSL
set(VSL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vsl)
add_library(vsl INTERFACE
    ${VSL_DIR}/impl/vsl_conv.h
    ${VSL_DIR}/impl/vsl_core.h
    ${VSL_DIR}/impl/vsl_impl.h
    ${VSL_DIR}/impl/vsl_logic.h
    ${VSL_DIR}/impl/vsl_math.h
    ${VSL_DIR}/impl/vsl_rand.h
    ${VSL_DIR}/vsl.h
)
target_include_directories(vsl INTERFACE ${VSL_DIR})

if(VSL_BUILD_TESTS)
    add_subdirectory(tests)
endif()