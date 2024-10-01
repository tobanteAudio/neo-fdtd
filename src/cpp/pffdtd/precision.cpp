// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "precision.hpp"

#include "pffdtd/exception.hpp"
#include "pffdtd/float.hpp"

namespace pffdtd {

template<typename T>
constexpr auto EPS = 0.0;

template<>
constexpr auto EPS<float> = 1.19209289e-07;

auto getEpsilon(Precision precision) -> double {
  switch (precision) {
    case Precision::Half: return EPS<float>;
    case Precision::Float: return EPS<float>;
    case Precision::Double: return EPS<double>;

    case Precision::DoubleHalf: return EPS<float>;
    case Precision::DoubleFloat: return EPS<float>;
    case Precision::DoubleDouble: return EPS<double>;

    default: break;
  }

  raisef<std::invalid_argument>("invalid precision = {}", int(precision));
}

auto getMinMaxExponent(Precision precision) -> std::pair<int, int> {
  using std::pair;
  switch (precision) {
    case Precision::Float: return pair{FloatTraits<float>::minExponent, FloatTraits<float>::maxExponent};
    case Precision::DoubleFloat: return pair{FloatTraits<float>::minExponent, FloatTraits<float>::maxExponent};

    case Precision::Double: return pair{FloatTraits<double>::minExponent, FloatTraits<double>::maxExponent};
    case Precision::DoubleDouble: return pair{FloatTraits<double>::minExponent, FloatTraits<double>::maxExponent};

#if defined(PFFDTD_HAS_FLOAT16)
    case Precision::Half: return pair{FloatTraits<_Float16>::minExponent, FloatTraits<_Float16>::maxExponent};
    case Precision::DoubleHalf: return pair{FloatTraits<_Float16>::minExponent, FloatTraits<_Float16>::maxExponent};
#endif

    default: break;
  }

  raisef<std::invalid_argument>("invalid precision = {}", int(precision));
}

} // namespace pffdtd
