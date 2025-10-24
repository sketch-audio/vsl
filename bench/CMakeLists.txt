add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE vsl audio_bench)

# set warnings
if(MSVC)
    target_compile_options(bench PRIVATE /W4)
else()
    target_compile_options(bench PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wswitch-enum -Wswitch-default)
endif()