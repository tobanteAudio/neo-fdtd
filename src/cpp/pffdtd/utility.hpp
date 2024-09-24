// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// Misc function definitions not specific to FDTD simulation
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifndef DIV_CEIL
  #define DIV_CEIL(x, y) (((x) + (y) - 1) / (y)) // this works for x≥0 and y>0
#endif
#define GET_BIT(var, pos)          (((var) >> (pos)) & 1)
#define SET_BIT(var, pos)          ((var) |= (1ULL << (pos)))
#define SET_BIT_VAL(var, pos, val) ((var) = ((var) & ~(1ULL << (pos))) | ((val) << (pos)))

namespace pffdtd {

template<typename T>
[[nodiscard]] auto allocate_zeros(std::integral auto count) -> T* {
  auto* const ptr  = new T[count];
  auto const bytes = static_cast<size_t>(count) * sizeof(T);
  std::memset(static_cast<void*>(ptr), 0, bytes);
  return ptr;
}

template<typename T>
[[nodiscard]] constexpr auto get_bit_as(std::integral auto word, std::integral auto pos) -> T {
  return static_cast<T>(GET_BIT(word, pos));
}

} // namespace pffdtd
