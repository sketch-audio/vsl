#include <bit>
#include <limits>

// #include "test_float32x4.h"
// #include "test_float64x2.h"
// #include "test_mask32x4.h"
// #include "test_mask64x2.h"
#include "test_math.h"
//#include "test_random.h"

int main()
{
    // We need two's complement.
    static_assert(std::bit_cast<int32_t>(std::numeric_limits<uint32_t>::max()) == -1);
    static_assert(std::bit_cast<int64_t>(std::numeric_limits<uint64_t>::max()) == -1);

    // test_float32x4();
    // test_float64x2();
    // test_mask32x4();
    // test_mask64x2();
    test_math();
    // test_random(); // Random really pushes the casts, integer multiplication.

    audio_bench::run_all_tests();

    return 0;
}