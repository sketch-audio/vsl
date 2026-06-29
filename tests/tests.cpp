#include <audio_bench/audio_bench.hpp>

// Test cases self-register from each translation unit in source/.
auto main() -> int
{
    const auto result = audio_bench::Tests::run_all();
    return static_cast<int>(result);
}