#include <audio_bench/audio_bench.hpp>

#include "test_float32x4.hpp"
#include "test_float64x2.hpp"
#include "test_mask32x4.hpp"
#include "test_mask64x2.hpp"
#include "test_math.hpp"
#include "test_random.hpp"

auto main() -> int
{
    test_float32x4();
    test_float64x2();
    test_mask32x4();
    test_mask64x2();
    test_math();
    test_random();
    audio_bench::Tests::run_all();
    return 0;
}