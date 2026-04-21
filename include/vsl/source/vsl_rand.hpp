#pragma once

#include "vsl_conv.hpp"
#include "vsl_core.hpp"
#include "vsl_logic.hpp" // select

namespace vsl {

enum class Random_engine { lin_cong, xor_shift };

template<typename X, Random_engine Engine = Random_engine::lin_cong>
struct Random_gen {

    using S = scalar_t<X>;

    static constexpr auto default_seed = 808u;

    Random_gen(unsigned_t<X> seed = default_seed) : _seed{seed}, _min{0}, _max{1} {};
    Random_gen(S min, S max, unsigned_t<X> seed = default_seed) : _seed{seed}, _min{min}, _max{max} {};

    auto reset(mask_t<X> mask = true_mask_v<S>) -> void
    {
        const auto cond = mask_to_bool(mask);
        _state = select(cond, _seed, _state);
    }

    auto next() -> X
    {
        constexpr auto shift = ieee_exp_bits_v<S>;
        constexpr auto one = unsigned_t<S>{1};
        constexpr auto denom = one << (ieee_sig_bits_v<S> + 1);
        constexpr auto mask = denom - 1;
        
        // We're doing the thing where we shift down so x doesn't get "rounded" on conversion.
        const auto x = unsigned_to_float((_next<Engine>() >> shift) & mask);
        constexpr auto y = unsigned_to_float(denom);
        const auto norm = x / y;
        
        return (_max - _min) * norm + _min;
    }

private:

    const unsigned_t<X> _seed{default_seed};
    const S _min{};
    const S _max{1};

    unsigned_t<X> _state{_seed};

    template<Random_engine>
    auto _next() -> unsigned_t<X> { static_assert(deferred_false_v<X>); }
    
    // See: https://audiodev.blog/random-numbers/
    template<>
    auto _next<Random_engine::lin_cong>() -> unsigned_t<X>
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
    auto _next<Random_engine::xor_shift>() -> unsigned_t<X>
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

} // namespace vsl