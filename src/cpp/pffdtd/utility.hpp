// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// Misc function definitions not specific to FDTD simulation
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#ifndef DIV_CEIL
  #define DIV_CEIL(x, y) (((x) + (y) - 1) / (y)) // this works for x≥0 and y>0
#endif
#define GET_BIT(var, pos)          (((var) >> (pos)) & 1)
#define SET_BIT(var, pos)          ((var) |= (1ULL << (pos)))
#define SET_BIT_VAL(var, pos, val) ((var) = ((var) & ~(1ULL << (pos))) | ((val) << (pos)))

namespace pffdtd {

template<typename T>
[[nodiscard]] auto allocate_zeros(std::integral auto count) -> std::unique_ptr<T[]> {
  auto ptr = std::make_unique<T[]>(static_cast<size_t>(count));
  std::memset(static_cast<void*>(ptr.get()), 0, static_cast<size_t>(count) * sizeof(T));
  return ptr;
}

template<typename T>
[[nodiscard]] constexpr auto get_bit_as(std::integral auto word, std::integral auto pos) -> T {
  return static_cast<T>(GET_BIT(word, pos));
}

} // namespace pffdtd
