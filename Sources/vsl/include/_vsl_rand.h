#ifndef _vsl_rand_h
#define _vsl_rand_h

#include <limits>

#include "_vsl_core.h"
#include "_vsl_utils.h" // unsigned_to_float

namespace vsl {

enum class Random_engine {
    linear_congruential, xorshift
};

template<typename X, size_t Seed = 808, Random_engine Engine = Random_engine::linear_congruential>
struct Random_gen {

    using S = scalar_t<X>;

    Random_gen() : _min{0}, _max{1} {}
    Random_gen(S min, S max) : _min{min}, _max{max} {}
    
    //
    auto reset(mask_t<X> mask = true_mask_v<X>) -> void
    {
        constexpr auto reset = uint_t<X>(Seed);
        _state = select(mask, reset, _state);
    }
    
    auto next() -> X
    {
        constexpr auto shift = ieee_exp_bits_v<S>;
        constexpr auto one = uint_t<S>{1};
        constexpr auto denom = one << (ieee_sig_bits_v<S> + 1);
        constexpr auto mask = denom - 1;
        
        // We're doing the thing where we shift down so x doesn't get "rounded" on conversion.
        const auto x = unsigned_to_float((_next<Engine>() >> shift) & mask);
        const auto y = unsigned_to_float(denom);
        const auto norm = x / y;
        
        return (_max - _min) * norm + _min;
    }
    
private:
    
    uint_t<X> _state = Seed;
    
    const S _min{};
    const S _max{};
    
    template<Random_engine>
    auto _next() -> uint_t<X> { static_assert(deferred_false_v<X>); }
    
    // see: https://audiodev.blog/random-numbers/
    template<>
    auto _next<Random_engine::linear_congruential>() -> uint_t<X>
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
    
    // see: https://en.wikipedia.org/wiki/Xorshift
    template<>
    auto _next<Random_engine::xorshift>() -> uint_t<X>
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

}

#endif /* _vsl_rand_h */
