# VSL — Vector Support Library

A header-only C++20 SIMD math library. All functions operate uniformly across scalar `float`, scalar `double`, `float32x4` (4-lane f32), and `float64x2` (2-lane f64) through a single templated implementation. SIMD dispatch is resolved at compile time with no virtual dispatch or runtime branching.

## Requirements

- C++20
- ARM NEON or x86 SSE4.2

## Usage

```cpp
#include <vsl/vsl.hpp>

const auto x0 = vsl::float32x4{1.f}; // All lanes contain 1.f.
const auto x1 = vsl::float32x4{1.f, 2.f, 3.f, 4.f};
const auto y = vsl::sin<false>(x0); // Wrapping not required here.
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

`vsl_conv.hpp` provides branchless conversion functions that work on both scalar and vector types:

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

`vsl_math.hpp` provides transcendental and rounding functions, all templated on the float type:

| Function | Notes |
|---|---|
| `abs` | |
| `sqrt`, `rsqrt` | `rsqrt` uses native reciprocal square root instruction |
| `trunc`, `floor`, `ceil`, `round` | |
| `fmod(x, y)` | |
| `wrap(x)` | Normalize to [0, 1] |
| `wrap(x, k)` | Symmetric wrap to [-k, k] |
| `wrap(x, a, b)` | Arbitrary range wrap |
| `min`, `max`, `clamp` | |
| `sign` | |
| `cos<needs_wrap>(x)`, `sin<needs_wrap>(x)`, `tan<needs_wrap>(x)` | Template parameter suppresses range reduction when input is already reduced |
| `cos_2pi<needs_wrap>(x)`, `sin_2pi<needs_wrap>(x)` | Input in cycles [0, 1] |
| `acos(x)`, `asin(x)`, `atan(x)` | |
| `exp(x)`, `exp2(x)` | |
| `log(x)`, `log2(x)`, `log10(x)` | |
| `logB(b, x)` | Arbitrary-base logarithm |
| `pow(x, y)` | Requires x >= 0 |
| `cosh(x)`, `sinh(x)`, `tanh(x)` | |

Approximation coefficients were derived by Claude (Anthropic) using Python scripts in `work/`, following the Chebyshev fitting technique described at https://mooooo.ooo/chebyshev-sine-approximation/. Each script fits the approximant on a reduced domain, evaluates f32 ULP error, and emits the Horner-form C++ coefficients. Accuracy is verified by the test suite.

## Random number generation

`vsl_rand.hpp` provides `Random_gen<X, Engine>`, a seedable PRNG that works on all four float types (scalar and vector), producing uniform floats in a configurable range.

Two engines are available via the `Random_engine` enum:

- `Random_engine::lin_cong` (default) — linear congruential generator
- `Random_engine::xor_shift` — xorshift generator

```cpp
using Rng = vsl::Random_gen<vsl::float32x4>;
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

## Denormals

`vsl_denormals.hpp` provides `No_denormals`, an RAII guard that sets the flush-to-zero and denormals-are-zero flags on x86 (SSE MXCSR) for the duration of its scope. On NEON the struct is a no-op.

```cpp
{
    auto nd = vsl::No_denormals{};
    // Denormals flushed to zero in this scope.
}
```

## Building tests and benchmarks

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/tests/tests
./build/bench/bench
```

Benchmarks use [audio_bench](https://github.com/sketch-audio/audio_bench) and compare vsl against `std::` equivalents.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
