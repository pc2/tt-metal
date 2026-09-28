// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

template <bool HAS_BASE_SCALING>
sfpi_inline void _calculate_log_body_(const std::uint32_t log_base_scale_factor, const std::uint32_t dst_idx = 0)
{
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    sfpi::vFloat in = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat x  = setexp(in, 127); // set exp to exp bias (put in range of 1-2)

    ////////////////////////////
    // Minimax cubic approximation of ln(x) on x in [1, 2), Horner form:
    // x * (x * (x * A + B) + C) + D
    // A: 0.10968964, B: -0.72910421, C: 2.11263230, D: -1.49277612
    // max |error| over [1, 2] = 4.4e-04
    ////////////////////////////
    sfpi::vFloat a = sfpi::vConstFloatPrgm1;
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;
    sfpi::vFloat series_result = x * (x * (x * a + b) + 2.11263230f) + -1.49277612f;

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(in));

    sfpi::vFloat expf      = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat result    = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING)
    {
        result *= sfpi::sFloat16a(log_base_scale_factor);
    }

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if (in == 0.0F)
    { // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}

/**
 * @brief ln(in) of a value already in a register: the arithmetic of _calculate_log_body_<false>
 * (same constants, so _init_log_ must have run) without its Dest load and store.
 *
 * For a caller that holds its operand in a register, this avoids the Dest store/reload round
 * trips that _calculate_log_body_ would need (two SFPSTOREs and two SFPLOADs per row).
 *
 * The -inf lane is every input whose biased exponent is 0, i.e. +-0 *and* the denormals.
 * That is what a caller that stores its operand to Dest and runs _calculate_log_body_ gets:
 * SFPSTORE flushes a denormal to zero (measured on Blackhole, fp32 and bf16 Dest), so the
 * in-place body only ever sees 0 there. Testing `in == 0.0F` instead would return a finite
 * ~-87.3..-103.3 for those inputs.
 */
sfpi_inline sfpi::vFloat _calculate_log_body_on_reg_(const sfpi::vFloat in)
{
    sfpi::vFloat x             = setexp(in, 127); // set exp to exp bias (put in range of 1-2)
    sfpi::vFloat a             = sfpi::vConstFloatPrgm1;
    sfpi::vFloat b             = sfpi::vConstFloatPrgm2;
    sfpi::vFloat series_result = x * (x * (x * a + b) + 2.11263230f) + -1.49277612f;

    auto exp               = sfpi::convert<sfpi::vSMag>(sfpi::exexp(in));
    sfpi::vFloat expf      = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat result    = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    v_if (sfpi::exexp(in, sfpi::ExponentMode::Biased) == 0)
    {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return result;
}

sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base)
{
    // Normalize base to calculation range
    sfpi::vFloat x = setexp(base, 127); // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    auto exp          = sfpi::convert<sfpi::vSMag>(sfpi::exexp(base));
    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

    // De-normalize to original range
    sfpi::vFloat vConstLn2  = 0.692871f;
    sfpi::vFloat log_result = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf
    v_if (base == 0.0f)
    {
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return log_result;
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void _calculate_log_(const int iterations, std::uint32_t log_base_scale_factor)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        _calculate_log_body_<HAS_BASE_SCALING>(log_base_scale_factor);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_log_()
{
    sfpi::vConstFloatPrgm0 = 0.69314718f; // ln2

    // Minimax cubic for ln on [1, 2); see _calculate_log_body_.
    sfpi::vConstFloatPrgm1 = 0.10968964f;
    sfpi::vConstFloatPrgm2 = -0.72910421f;
}

} // namespace sfpu
} // namespace ckernel
