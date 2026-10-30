set(VSL_TEST_SOURCES
    tests.cpp
    source/test_float32x4.cpp
    source/test_float64x2.cpp
    source/test_mask32x4.cpp
    source/test_mask64x2.cpp
    source/test_math.cpp
    source/test_random.cpp
    source/test_complex.cpp
    source/test_conv.cpp
    source/test_logic.cpp
)

if(MSVC)
    set(VSL_WARNINGS /W4 /permissive-)
else()
    set(VSL_WARNINGS -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wswitch-enum -Wswitch-default)
endif()

add_executable(tests ${VSL_TEST_SOURCES})
target_link_libraries(tests PRIVATE vsl audio_bench)
target_compile_options(tests PRIVATE ${VSL_WARNINGS})
add_test(NAME tests COMMAND tests)

# The same tests under -ffast-math. Clang keeps VSL's own arithmetic in order there (see vsl_math.hpp).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_executable(tests_fast_math ${VSL_TEST_SOURCES})
    target_link_libraries(tests_fast_math PRIVATE vsl audio_bench)
    target_compile_options(tests_fast_math PRIVATE ${VSL_WARNINGS} -ffast-math -Wno-nan-infinity-disabled)
    add_test(NAME tests_fast_math COMMAND tests_fast_math)
endif()

# Accuracy against double truth, every 64th float. Run `accuracy 1` for every float before a release.
find_package(Threads REQUIRED)
add_executable(accuracy accuracy.cpp)
target_link_libraries(accuracy PRIVATE vsl Threads::Threads)
target_compile_options(accuracy PRIVATE ${VSL_WARNINGS})
add_test(NAME accuracy COMMAND accuracy)
