#pragma once

#include "vsl_impl.hpp"

namespace vsl {

struct No_denormals {

    No_denormals() {
#if VSL_SSE4
        _flag = _mm_getcsr();
        _mm_setcsr(_flag | 0x8040);
#endif
    }

    ~No_denormals() {
#if VSL_SSE4
        _mm_setcsr(_flag);
#endif
    }

private:
#if VSL_SSE4
    unsigned int _flag{};
#endif
};

} // namespace vsl