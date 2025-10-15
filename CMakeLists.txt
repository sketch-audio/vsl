cmake_minimum_required(VERSION 3.30)

project(vsl VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vsl INTERFACE)
target_include_directories(vsl INTERFACE include)

if(MSVC)
    target_compile_options(vsl INTERFACE /arch:SSE4.2)
else()
    target_compile_options(vsl INTERFACE -msse4.2 -Wno-unused-command-line-argument) # Needed for universal build.
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(FetchContent)
    FetchContent_Declare(
        audio_bench
        GIT_REPOSITORY https://github.com/sketch-audio/audio_bench.git
        GIT_TAG main
        # SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../audio_bench
    )
    FetchContent_MakeAvailable(audio_bench)
    add_subdirectory(bench)
    add_subdirectory(tests)
endif()