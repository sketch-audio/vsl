#include <cassert>
#include <iostream>

#include "include/vsl.h"

#define VERBOSE 0

int main() {
    // MARK: - Test cxm

    const auto true_mask = vsl::true_mask_v<vsl::float4>;
    const auto tol = vsl::float4(1e-5f);

    // exp2
    const auto arg_exp2 = vsl::float4(3.5f);
    const auto res_exp2 = vsl::cxm::exp2(arg_exp2);
    const auto ref_exp2 = vsl::float4(std::exp2(3.5f));
    const auto mask_exp2 = vsl::about_equal(res_exp2, ref_exp2, tol);
    assert(vsl::all(mask_exp2 == true_mask));

    // log2
    const auto arg_log2 = vsl::float4(69.f);
    const auto res_log2 = vsl::cxm::log2(arg_log2);
    const auto ref_log2 = vsl::float4(std::log2(69.f));
    const auto mask_log2 = vsl::about_equal(res_log2, ref_log2, tol);
    assert(vsl::all(mask_log2 == true_mask));

    // MARK: - Test Random_gen

    // float
    auto frng = vsl::Random_gen<float>{-1, 1};
    
    for (size_t i = 0; i < 10; ++i) {
        [[maybe_unused]] const auto v = frng.next();
#if VERBOSE
        std::cout << v << std::endl;
#endif
    }

    // double
    auto drng = vsl::Random_gen<double>{-1, 1};
    
    for (size_t i = 0; i < 10; ++i) {
        [[maybe_unused]] const auto v = drng.next();
#if VERBOSE
        std::cout << v << std::endl;
#endif
    }

    // float4
    auto f4rng = vsl::Random_gen<vsl::float4>{-1, 1};

    for (size_t i = 0; i < 10; ++i) {
        if (i == 5) {
            f4rng.reset(vsl::int4{-1, 0, -1, 0});
        }

        [[maybe_unused]] const auto v = f4rng.next();
#if VERBOSE
        std::cout << "{" << v[0] << ", " << v[1] << ", "\
            << v[2] << ", " << v[3] << "}" << std::endl;
#endif
    }

    // double2
    auto d2rng = vsl::Random_gen<vsl::double2>{-1, 1};

    for (size_t i = 0; i < 10; ++i) {
        if (i == 5) {
            d2rng.reset(vsl::long2{-1, 0});
        }

        [[maybe_unused]] const auto v = d2rng.next();
#if VERBOSE
        std::cout << "{" << v[0] << ", " << v[1] << "}" << std::endl;
#endif
    }
    
    return 0;
}
