cmake_minimum_required(VERSION 3.30)

project(vsl
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vsl INTERFACE)
target_include_directories(vsl INTERFACE include)

if(PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
endif()