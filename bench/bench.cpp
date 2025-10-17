#include <iostream>

#include "source/bench_math.h"

int main() {
    std::cout << "\nRunning VSL benchmarks...\n\n";
    // bench_cos();
    // bench_sin();
    // bench_tan();
    // bench_cos_2pi();
    // bench_sin_2pi();
    // bench_exp2();
    // bench_log2();
    // bench_exp();
    bench_pow();
    // bench_sqrt();
    // bench_cosh();
    // bench_sinh();
    // bench_tanh();
    std::cout << "Benchmarks complete.\n";
    return 0;
}