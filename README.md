# VSL — Vector Support Library

A header-only C++20 SIMD-compatible math library.

## Requirements

- C++20
- ARM NEON or x86 SSE4.2

## Usage

```cpp
#include <vsl/vsl.hpp>

const auto x0 = vsl::float32x4{1.f}; // All lanes contain 1.f.
const auto x1 = vsl::float32x4{1.f, 2.f, 3.f, 4.f};
const auto y = vsl::sin(x0);
const auto z = vsl::tanh(x1);
const auto s = vsl::exp(2.); // Scalar works too.
```

## Vector types

Four concrete SIMD types are defined in `vsl_impl.hpp`, each wrapping the platform intrinsic type (`float32x4_t` / `__m128`, etc.):

| Type | Lanes | Element |
|---|---|---|
| `float32x4` | 4 | `float` |
| `float64x2` | 2 | `double` |
| `mask32x4` | 4 | `uint32_t` |
| `mask64x2` | 2 | `uint64_t` |

All four types support arithmetic operators (`+`, `-`, `*`, `/`), comparison operators (returning the corresponding mask type), bitwise operators on the mask types, and scalar lane access via `operator[]`. Static member functions `select`, `any`, and `all` are provided on each type.

## Type traits

`vsl_core.hpp` provides traits and aliases for working generically across scalar and vector types:

- `scalar_t<T>` — the scalar element type (`float32x4` → `float`)
- `vector_t<T>` — the vector counterpart of a scalar (`float` → `float32x4`)
- `float_t<T>` — the floating-point type of the same bit width
- `mask_t<T>` — the unsigned integer / mask type of the same bit width
- `signed_t<T>` — the signed integer type of the same bit width
- `unsigned_t<T>` — the unsigned integer type of the same bit width
- `num_members_v<T>` — number of lanes (`1` for scalars, `4` for `float32x4`, etc.)
- `ieee_exp_bias_v<T>`, `ieee_sig_bits_v<T>`, `ieee_exp_bits_v<T>` — IEEE 754 format constants

Concepts: `Scalar`, `Vector`, `Integral`, `Floating_point`.

## Conversions

`vsl_convert.hpp` provides branchless conversion functions that work on both scalar and vector types:

- `reinterpret_as_float(x)` / `reinterpret_as_int(x)` — bit-cast between float and integer
- `unsigned_to_float(x)` / `signed_to_float(x)` — integer-to-float cast (static_cast semantics)
- `float_to_signed(x)` — truncating float-to-integer cast
- `unsigned_to_signed(x)` / `signed_to_unsigned(x)` — bit-cast between signed and unsigned
- `mask_to_bool(x)` / `bool_to_mask(x)` — convert between boolean and bitmask representation

## Logic

`vsl_logic.hpp` provides branchless logic utilities that are scalar/vector transparent:

- `select(cond, val_t, val_f)` — vector-compatible ternary; maps to `vbslq` / `_mm_blendv` on vectors, ternary on scalars
- `any(cond)` — true if any lane is set
- `all(cond)` — true if all lanes are set
- `all_same(x)` — true if all lanes hold the same value
- `all_unique(x)` — true if all lanes hold distinct values
- `all_finite(x)` — true if no lane is NaN or infinity
- `abs_equal(a, b, tol)` — absolute-tolerance comparison
- `rel_equal(a, b, tol)` — relative-tolerance comparison
- `bit_equal(a, b, tol)` — comparison by number of matching mantissa bits

## Math

`vsl_math.hpp` provides transcendental and rounding functions, all templated on the float type.

The transcendental functions take an optional `Precision` first, so the float type is still deduced:

```cpp
const auto y = vsl::tanh(x);                         // Precision::Standard
const auto f = vsl::tanh<vsl::Precision::Fast>(x);
```

`Fast` only differs from `Standard` where it's measurably cheaper and stays inaudible; elsewhere the two are the same. Both give the same results at 0, the clamps and overflow. Measured on an M2 (cost saved is per sample, scalar and `float32x4`):

| `Fast` | Accuracy | Saves |
|---|---|---|
| `tanh` | -83 dB, saturates at 0.99993 | ~20% |
| `tan_pi`, `kernel::tan_pi` | 7.6e-5 relative up to 20 kHz at 48 kHz | 15–25% |
| `sin_2pi`, `kernel::sin_2pi` | 4.7 ULP | 10–20% |
| `exp2`, `exp` (and `db_to_gain`, `cosh`, `sinh`) | 2.7 / 3.3 ULP, exactly 1 at 0 | 5–25% |
| `log2`, `log`, `log10`, `gain_to_db` | 21 ULP (1.5e-6 relative, 2.5e-5 dB) | 12–20% |
| `pow` | 1e-5 relative | ~15% |

| Function | Notes |
|---|---|
| `abs` | |
| `sqrt`, `rsqrt` | `rsqrt` is `1 / sqrt` on every backend |
| `rsqrt_estimate` | The hardware estimate: ~8 bits on NEON, ~12 on SSE |
| `trunc`, `floor`, `ceil`, `round` | |
| `rint` | Round to nearest, ties to even: one instruction on both backends |
| `fmod(x, y)` | |
| `wrap(x)` | Normalize to [0, 1] |
| `wrap(x, k)` | Symmetric wrap to [-k, k] |
| `wrap(x, a, b)` | Arbitrary range wrap |
| `min`, `max`, `clamp` | NaN `x` returns the bound on every backend |
| `smooth_max(a, b, c)` | Smooth max of `a`, `b` with parameter `c` |
| `mul_add(a, b, c)` | `a * b + c`, fused where the target has it (NEON; x86 built with FMA or AVX2). No single-rounding guarantee |
| `sign` | |
| `sin(x)`, `cos(x)`, `tan(x)` | Every float. Relative accuracy holds near zeros and poles for \|x\| up to ~1e2, absolute (~3e-7) to ~1e4; 0 for \|x\| ≥ 2^24 |
| `sin_2pi(x)`, `cos_2pi(x)`, `tan_pi(x)` | Turns: every float, exact reduction. `tan_pi<Fast>`: 0.07 cents up to 20 kHz at 48 kHz |
| `sin_cos_2pi(x)` | `{sin, cos}` of 2πx with one reduction |
| `kernel::sin`, `kernel::cos` ([-π, π]), `kernel::tan` ([-π/2, π/2]) | No reduction: for arguments already in range. About half the cost of the radians functions |
| `kernel::sin_2pi`, `kernel::cos_2pi` ([0, 1]), `kernel::tan_pi` ([-1/2, 1/2]) | No reduction: a phase in [0, 1), or `fc / fs` |
| `reduce_turns(x)` | `x - round(x)`: exact, in [-1/2, 1/2] |
| `reduce_pi(x)` | `{r, n}` with `x = n π + r`, `r` in [-π/2, π/2] |
| `acos(x)`, `asin(x)`, `atan(x)` | |
| `exp(x)`, `exp2(x)` | `inf` on overflow, `0` below the smallest normal |
| `expm1(x)` | `e^x - 1`, accurate near 0 (one-pole coefficients at low cutoffs) |
| `db_to_gain(db)`, `gain_to_db(gain)` | Exactly 1 at 0 dB and back; -inf dB ↔ 0; ~1e-5 dB |
| `log(x)`, `log2(x)`, `log10(x)` | Requires x >= 0; `log2(0)` is `-inf` |
| `logB(b, x)` | Arbitrary-base logarithm |
| `pow(x, y)` | Requires x >= 0; `pow(0, 0)` is 1, `pow(0, y < 0)` is `inf` |
| `cosh(x)`, `sinh(x)`, `tanh(x)` | `tanh` is finite for every input; Standard: -120 dB, saturates at exactly ±1. Fast: -83 dB |

Subnormals behave as under flush-to-zero: results never come out subnormal, and `log2` treats subnormal inputs as 0. Under `-ffast-math` (strictly, `-ffinite-math-only`), where infinities are undefined behaviour, overflow saturates to the largest finite value and `log2(0)` is -150 instead. With clang, VSL's headers opt out of reassociation, so its results hold under `-ffast-math` (on x86, clang's reciprocal estimates for division and square root can still cost ~1 ULP); your own code keeps fast math. With GCC and MSVC, expect a few ULP where the compiler reassociates.

`double` and `float64x2` use the float kernels: about 1e-7 relative error, the same as `float`. Use `std::` where you need full double precision.

Approximation coefficients were derived by Claude (Anthropic) using Python scripts in `work/`, following the Chebyshev fitting technique described at https://mooooo.ooo/chebyshev-sine-approximation/ (`tanh` uses a weighted Remez fit and a Padé approximant, `work/remez_tanh.py`). Each script fits the approximant on a reduced domain, evaluates f32 ULP error, and emits the Horner-form C++ coefficients; scripts with a `Fast` set take its degree as an argument (`python3 work/cheb_log2.py 6`), and the header names the script and degree beside each set. Accuracy is verified by the test suite (`accuracy 1` checks every float).

## Random number generation

`vsl_random.hpp` provides `Generator<X, Engine>`, a seedable PRNG that works on all four float types (scalar and vector), producing uniform floats in a configurable range.

Two engines are available via the `Engine` enum:

- `Engine::Lin_cong` (default) — linear congruential generator
- `Engine::Xor_shift` — xorshift generator

```cpp
using Rng = vsl::Generator<vsl::float32x4>;
auto rng = Rng{-1.f, 1.f};
const auto noise = rng.next();
rng.reset();  // restore to seed state, optionally masked per-lane
```

## Complex numbers

`vsl_complex.hpp` provides `Complex<X>`, a simple complex number type templated on any arithmetic type including the SIMD vector types. Supports `+`, `-`, `*`, `/` between complex values and between complex and scalar.

```cpp
using Cplex = vsl::Complex<vsl::float32x4>;
const auto z = Cplex{re, im};
const auto w = z * z;
```

## Building tests and benchmarks

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/tests/tests
./build/bench/bench
```

`tests/accuracy` checks every math function against double truth on every 64th float, with one bound per function that every backend must meet; `./build/tests/accuracy 1` checks every float (a few minutes). Under clang, `tests_fast_math` reruns the tests with `-ffast-math`.

Benchmarks use [audio_bench](https://github.com/sketch-audio/audio_bench) and compare vsl against `std::` equivalents.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
