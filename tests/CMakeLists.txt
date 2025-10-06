add_executable(tests tests.cpp)
target_link_libraries(tests PRIVATE vsl)

if(MSVC)
    target_compile_options(tests PRIVATE /W4 /WX)
else()
    target_compile_options(tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()