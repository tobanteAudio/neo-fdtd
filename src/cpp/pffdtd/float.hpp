// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include <limits>

#if (defined(__APPLE__) or defined(__clang__)) and not defined(__INTEL_LLVM_COMPILER)
  #define PFFDTD_HAS_FLOAT16
#endif

namespace pffdtd {

template<typename T>
struct FloatTraits;

#if defined(PFFDTD_HAS_FLOAT16)
template<>
struct FloatTraits<_Float16> {
  static constexpr auto digits      = 11;
  static constexpr auto minExponent = -13;
  static constexpr auto maxExponent = 16;
};
#endif

template<>
struct FloatTraits<float> {
  static constexpr auto digits      = std::numeric_limits<float>::digits;
  static constexpr auto minExponent = std::numeric_limits<float>::min_exponent;
  static constexpr auto maxExponent = std::numeric_limits<float>::max_exponent;
};

template<>
struct FloatTraits<double> {
  static constexpr auto digits      = std::numeric_limits<double>::digits;
  static constexpr auto minExponent = std::numeric_limits<double>::min_exponent;
  static constexpr auto maxExponent = std::numeric_limits<double>::max_exponent;
};

} // namespace pffdtd
