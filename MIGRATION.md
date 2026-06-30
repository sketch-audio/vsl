# Migration guide

Notes for moving from the previous `main` to this release. Highlights only — the
test and benchmark infrastructure was substantially reworked, but the changes
below are the ones that affect calling code.

## 1. Header extension: `.h` → `.hpp`

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
`vsl_math`, `vsl_complex`, `vsl_denormals`).

## 2. Trig functions take an explicit `needs_wrap` template argument

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

## 3. Random number generator renamed

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

## 4. `cos_v2` removed

The experimental `vsl::cos_v2` was removed. Use `vsl::cos<needs_wrap>`.

## 5. `round` now matches `std::round` (ties away from zero)

`round` previously rounded halfway cases to the nearest *even* integer (banker's
rounding). It now rounds halves *away from zero*, matching `std::round`:

```cpp
vsl::round(0.5f);   // was 0, now 1
vsl::round(2.5f);   // was 2, now 3
```

Only exact `.5` inputs are affected.

## New, additive (no action needed)

- `vsl::fma(a, b, c)` — fused multiply-add (`a*b + c`, single rounding).
- `vsl::rsqrt(x)` — reciprocal square root (native instruction where available).
- `wrap(x, k)` — symmetric wrap to `[-k, k]` is now a public overload of `wrap`.
- Domain-precondition checking was expanded across the math functions, via
  `assert` in debug builds (stripped under `NDEBUG`). Passing out-of-domain
  inputs (e.g. `log2(-1)`, `acos(2)`) may now trip an assertion in debug builds
  where it previously went unchecked.

## Contributor-facing

- Tests moved from included `tests/*.h` headers to self-registering
  `tests/source/*.cpp` translation units; `test_helpers.h` was removed.
- The original approximation scripts were archived under `work/v1/`.
