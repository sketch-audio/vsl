# Migration guide

## 2.x → 3.0

### Trig functions are full range; `needs_wrap` is gone

Every trig function now takes any float and reduces it accurately. The `bool` template
parameter is replaced by an optional `Precision`, so old calls stop compiling:

```cpp
vsl::sin<true>(x);        // 2.x
vsl::sin(x);              // 3.0: any x

vsl::sin<false>(x);       // 2.x: x already in [-pi, pi]
vsl::kernel::sin(x);      // 3.0: same domain, no reduction
```

| 2.x | 3.0 |
|---|---|
| `sin<true>`, `cos<true>`, `tan<true>` | `sin`, `cos`, `tan` |
| `sin<false>`, `cos<false>` ([-π, π]) | `kernel::sin`, `kernel::cos` |
| `tan<false>` ([-π/2, π/2]) | `kernel::tan` |
| `sin_2pi<true>`, `cos_2pi<true>`, `tan_pi<true>` | `sin_2pi`, `cos_2pi`, `tan_pi` |
| `sin_2pi<false>`, `cos_2pi<false>` ([0, 1]) | `kernel::sin_2pi`, `kernel::cos_2pi` (same domain) |
| `tan_pi<false>` ([-1/2, 1/2]) | `kernel::tan_pi` (same domain) |

New: `sin_cos_2pi(x)` (one reduction, `{sin, cos}`), `reduce_turns(x)`, `reduce_pi(x)`.

### `Precision`

Transcendental functions take an optional `vsl::Precision` (`Standard` by default, or `Fast`):

```cpp
vsl::tanh<vsl::Precision::Fast>(x);
```

`Fast` differs for `tanh`, `tan_pi`, `sin_2pi`, `exp2`/`exp`, `log2`/`log`/`log10` and `pow`
(and the functions built on them); see the README for its accuracy.

### `fma` → `mul_add`

`vsl::fma` is deprecated; use `vsl::mul_add`. It was never single-rounding on SSE4.2, and
the new name doesn't promise it. It's now fused on x86 when the build targets FMA (`-mfma`,
`-march=x86-64-v3`, `/arch:AVX2`).

### New

- `expm1(x)`, `db_to_gain(db)`, `gain_to_db(gain)`.
- `rint(x)`: round to nearest, ties to even.
- `sin_cos_2pi(x)`, `reduce_turns(x)`, `reduce_pi(x)`.

### Behaviour changes

- `log2` (and `log`, `log10`, `gain_to_db`) treats subnormal input as 0 and returns -inf, with or
  without DAZ.
- `sin`, `cos`, `tan` return 0 for |x| >= 2^24, where every float is an even integer and the
  reduction has nothing left to work with.

### `rsqrt` is accurate

`vsl::rsqrt` is `1 / sqrt` on every backend (it was the hardware estimate on vectors). For
the estimate, use `vsl::rsqrt_estimate`.

---

## 1.x → 2.0

Notes for moving from the previous `main` to this release. Highlights only — the
test and benchmark infrastructure was substantially reworked, but the changes
below are the ones that affect calling code.

### 1. Header extension: `.h` → `.hpp`

Every header was renamed to `.hpp`. Update includes:

```cpp
#include <vsl/vsl.h>     // old
#include <vsl/vsl.hpp>   // new
```

Most code only includes the umbrella `<vsl/vsl.hpp>`. If you included individual
headers directly, note that two were also renamed beyond the extension:

| old | new |
|---|---|
| `vsl_conv.h` | `vsl_convert.hpp` |
| `vsl_rand.h` | `vsl_random.hpp` |

The rest are a straight `.h` → `.hpp` (`vsl_core`, `vsl_impl`, `vsl_logic`,
`vsl_math`, `vsl_complex`).

### 2. Trig functions take an explicit `needs_wrap` template argument

`cos`, `sin`, `tan`, `cos_2pi`, `sin_2pi` now require a `bool` template parameter
selecting whether the input still needs range reduction. There is no default.

```cpp
vsl::cos(x);          // old — always range-reduced
vsl::cos<true>(x);    // new — same behaviour (reduce first)
vsl::cos<false>(x);   // new — skip reduction; input already in range
```

Use `<true>` to preserve the old behaviour. Use `<false>` (faster) only when the
argument is already in the approximation's domain — `[-pi, pi]` for `cos`/`sin`,
`[-pi/2, pi/2]` for `tan`, `[0, 1]` for `cos_2pi`/`sin_2pi`.

### 3. Random number generator renamed

```cpp
vsl::Random_gen<vsl::float32x4, vsl::Random_engine::lin_cong>  // old
vsl::Generator<vsl::float32x4, vsl::Engine::Lin_cong>          // new
```

| old | new |
|---|---|
| `Random_gen<X, Engine>` | `Generator<X, Engine>` |
| `Random_engine` | `Engine` |
| `Random_engine::lin_cong` | `Engine::Lin_cong` |
| `Random_engine::xor_shift` | `Engine::Xor_shift` |

Constructors and members (`next()`, `reset(mask)`) are unchanged.

### 4. `cos_v2` removed

The experimental `vsl::cos_v2` was removed. Use `vsl::cos<needs_wrap>`.

### 5. `round` now matches `std::round` (ties away from zero)

`round` previously rounded halfway cases to the nearest *even* integer (banker's
rounding). It now rounds halves *away from zero*, matching `std::round`:

```cpp
vsl::round(0.5f);   // was 0, now 1
vsl::round(2.5f);   // was 2, now 3
```

Only exact `.5` inputs are affected.

### 6. vsl::No_denormals is gone.

You should be able to rely on the new Denormal_guard in tinyplug now present in all wrappers.

### New, additive (no action needed)

- `vsl::fma(a, b, c)` — fused multiply-add (`a*b + c`, single rounding).
- `vsl::rsqrt(x)` — reciprocal square root (native instruction where available).
- `wrap(x, k)` — symmetric wrap to `[-k, k]` is now a public overload of `wrap`.
- Domain-precondition checking was expanded across the math functions, via
  `assert` in debug builds (stripped under `NDEBUG`). Passing out-of-domain
  inputs (e.g. `log2(-1)`, `acos(2)`) may now trip an assertion in debug builds
  where it previously went unchecked.

### Contributor-facing

- Tests moved from included `tests/*.h` headers to self-registering
  `tests/source/*.cpp` translation units; `test_helpers.h` was removed.
- The original approximation scripts were archived under `work/v1/`.
