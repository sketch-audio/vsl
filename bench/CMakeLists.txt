add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE vsl audio_bench)

if(MSVC)
    target_compile_options(bench PRIVATE /W4 /WX)
else()
    target_compile_options(bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()