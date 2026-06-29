#pragma once

#include "vsl_conv.hpp"
#include "vsl_core.hpp"
#include "vsl_logic.hpp" // select

namespace vsl::random {

enum class Engine : uint32_t { Lin_cong = 0, Xor_shift };

template<typename X, Engine Eng = Engine::Lin_cong>
struct Generator {

    using S = scalar_t<X>;
    using Ux = unsigned_t<X>;
    using Us = unsigned_t<S>;

    explicit Generator(Ux seed = default_seed) : _seed{seed}, _min{0}, _max{1} {};
    Generator(S min, S max, Ux seed = default_seed) : _seed{seed}, _min{min}, _max{max} {};

    // Reset to seed where mask is true (default all lanes).
    auto reset(mask_t<X> mask = true_mask_v<S>) -> void
    {
        const auto cond = mask_to_bool(mask);
        _state = select(cond, _seed, _state);
    }

    // Get next random value in range [_min, _max).
    [[nodiscard]] auto next() -> X
    {
        constexpr auto shift = ieee_exp_bits_v<S>;
        constexpr auto one = Us{1};
        constexpr auto denom = one << (ieee_sig_bits_v<S> + 1);
        constexpr auto mask = denom - 1;
        
        // We're doing the thing where we shift down so x doesn't get "rounded" on conversion.
        const auto x = unsigned_to_float((_next<Eng>() >> shift) & mask);
        constexpr auto y = unsigned_to_float(denom); // Scalar so it can be constexpr.
        const auto norm = x / y;
        
        return (_max - _min) * norm + _min;
    }

private:

    static constexpr auto default_seed = Us{808};

    Ux _seed{default_seed};
    Ux _state{_seed};
    S _min{};
    S _max{1};

    template<Engine>
    auto _next() -> Ux { static_assert(deferred_false_v<X>); }
    
    // See: https://audiodev.blog/random-numbers/
    template<>
    auto _next<Engine::Lin_cong>() -> Ux
    {
        // 32-bit scalar type
        if constexpr (sizeof(S) == 4) {
            _state = 196314165 * _state + 907633515;
            return _state;
        }
        // 64-bit scalar type
        else if constexpr (sizeof(S) == 8) {
            _state = 6364136223846793005 * _state + 1442695040888963407;
            return _state;
        }
        else {
            static_assert(deferred_false_v<S>);
        }
    }

    // See: https://en.wikipedia.org/wiki/Xorshift
    template<>
    auto _next<Engine::Xor_shift>() -> Ux
    {
        // 32-bit scalar type
        if constexpr (sizeof(S) == 4) {
            auto x = _state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            _state = x;
            return x;
        }
        // 64-bit scalar type
        else if constexpr (sizeof(S) == 8) {
            auto x = _state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            _state = x;
            return x;
        }
        else {
            static_assert(deferred_false_v<S>);
        }
    }
};
static_assert(std::semiregular<Generator<float>>);

} // namespace vsl::random