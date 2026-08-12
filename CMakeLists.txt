cmake_minimum_required(VERSION 3.30)

project(vsl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vsl INTERFACE)
target_include_directories(vsl INTERFACE include)

if(MSVC)
    target_compile_options(vsl INTERFACE /arch:SSE4.2)
elseif(APPLE)
    # We need to restrict the `-mfma` to the x86_64 slice.
    target_compile_options(vsl INTERFACE -msse4.2 -Xarch_x86_64 -mfma -Wno-unused-command-line-argument)
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(FetchContent)
    FetchContent_Declare(
        audio_bench
        GIT_REPOSITORY https://github.com/sketch-audio/audio_bench.git
        GIT_TAG 0adbf7a177fda9d44f606b3d74aafcfda5799940
        SYSTEM
    )
    FetchContent_MakeAvailable(audio_bench)
    enable_testing()
    add_subdirectory(bench)
    add_subdirectory(tests)
endif()