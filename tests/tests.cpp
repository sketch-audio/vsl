#include <audio_bench/audio_bench.h>

#include "test_float32x4.h"
#include "test_float64x2.h"
#include "test_mask32x4.h"
#include "test_mask64x2.h"
#include "test_math.h"
#include "test_random.h"

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