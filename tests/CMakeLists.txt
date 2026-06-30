add_executable(tests tests.cpp)
target_sources(tests PRIVATE
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
target_link_libraries(tests PRIVATE vsl audio_bench)

# set warnings
if(MSVC)
    target_compile_options(tests PRIVATE /W4 /permissive-)
else()
    target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wswitch-enum -Wswitch-default)
endif()

add_test(NAME tests COMMAND tests)