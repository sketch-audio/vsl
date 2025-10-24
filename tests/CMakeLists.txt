add_executable(tests tests.cpp)
target_link_libraries(tests PRIVATE vsl audio_bench)

# set warnings
if(MSVC)
    target_compile_options(tests PRIVATE /W4)
else()
    target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wswitch-enum -Wswitch-default)
endif()