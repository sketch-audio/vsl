#include <iostream>

#include "source/bench_math.hpp"

int main()
{
    std::cout << "\nVSL math benchmarks vs std (equal call count, N=100000)\n";
    std::cout << "vector = vsl on f32x4 / f64x2 (broadcast); ratios are 'x faster than std'.\n\n";
    bench_rounding();
    bench_roots();
    bench_trig();
    bench_invtrig();
    bench_explog();
    bench_power();
    bench_hyperbolic();
    std::cout << "\nBenchmarks complete.\n";
    return 0;
}
