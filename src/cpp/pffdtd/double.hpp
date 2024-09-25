// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch
#pragma once

#include <cmath>
#include <concepts>
#include <limits>

namespace pffdtd {

/// https://github.com/sukop/doubledouble/blob/master/doubledouble.py
template<std::floating_point Real>
struct Double {
  constexpr Double() = default;

  constexpr Double(Real x, Real y = 0.0) noexcept : _high{x}, _low{y} {}

  [[nodiscard]] constexpr auto high() const noexcept -> Real { return _high; }

  [[nodiscard]] constexpr auto low() const noexcept -> Real { return _low; }

  template<std::floating_point OtherReal>
  explicit constexpr operator OtherReal() const noexcept {
    return static_cast<OtherReal>(high());
  }

  template<std::floating_point OtherReal>
  explicit constexpr operator Double<OtherReal>() const noexcept {
    return Double<OtherReal>{
        static_cast<OtherReal>(high()),
        static_cast<OtherReal>(low()),
    };
  }

  friend constexpr auto operator+(Double x) noexcept -> Double { return x; }

  friend constexpr auto operator-(Double x) noexcept -> Double { return {-x.high(), -x.low()}; }

  friend constexpr auto operator+(Double lhs, Double rhs) noexcept -> Double {
    auto [r, e] = twoSum(lhs.high(), rhs.high());
    e += lhs.low() + rhs.low();
    return twoSumQuick(r, e);
  }

  friend constexpr auto operator-(Double lhs, Double rhs) noexcept -> Double {
    auto [r, e] = twoDifference(lhs.high(), rhs.high());
    e += lhs.low() - rhs.low();
    return twoSumQuick(r, e);
  }

  friend constexpr auto operator*(Double lhs, Double rhs) noexcept -> Double {
    auto [r, e] = twoProduct(lhs.high(), rhs.high());
    e += lhs.high() * rhs.low() + lhs.low() * rhs.high();
    return twoSumQuick(r, e);
  }

  friend constexpr auto operator/(Double lhs, Double rhs) noexcept -> Double {
    auto r      = lhs.high() / rhs.high();
    auto [s, f] = twoProduct(r, rhs.high());
    auto e      = (lhs.high() - s - f + lhs.low() - r * rhs.low()) / rhs.high();
    return twoSumQuick(r, e);
  }

  friend constexpr auto operator+=(Double& lhs, Double rhs) noexcept -> Double& {
    lhs = lhs + rhs;
    return lhs;
  }

  friend constexpr auto operator-=(Double& lhs, Double rhs) noexcept -> Double& {
    lhs = lhs - rhs;
    return lhs;
  }

  friend constexpr auto operator*=(Double& lhs, Double rhs) noexcept -> Double& {
    lhs = lhs * rhs;
    return lhs;
  }

  friend constexpr auto operator/=(Double& lhs, Double rhs) noexcept -> Double& {
    lhs = lhs / rhs;
    return lhs;
  }

  private:
  [[nodiscard]] static constexpr auto twoSum(Real x, Real y) noexcept -> Double {
    auto r = x + y;
    auto t = r - x;
    auto e = (x - (r - t)) + (y - t);
    return Double{r, e};
  }

  [[nodiscard]] static constexpr auto twoSumQuick(Real x, Real y) noexcept -> Double {
    auto r = x + y;
    auto e = y - (r - x);
    return Double{r, e};
  }

  [[nodiscard]] static constexpr auto twoDifference(Real x, Real y) noexcept -> Double {
    auto r = x - y;
    auto t = r - x;
    auto e = (x - (r - t)) - (y + t);
    return Double{r, e};
  }

  [[nodiscard]] static constexpr auto twoProduct(Real x, Real y) noexcept -> Double {
    auto r = x * y;
    auto e = std::fma(x, y, -r);
    return Double{r, e};
  }

  Real _high{static_cast<Real>(0)};
  Real _low{static_cast<Real>(0)};
};

} // namespace pffdtd

template<std::floating_point Real>
struct std::numeric_limits<pffdtd::Double<Real>> : std::numeric_limits<Real> {};
