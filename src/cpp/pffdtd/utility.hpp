// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// Misc function definitions not specific to FDTD simulation
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifndef DIV_CEIL
  #define DIV_CEIL(x, y) (((x) + (y) - 1) / (y)) // this works for x≥0 and y>0
#endif
#define GET_BIT(var, pos)          (((var) >> (pos)) & 1)
#define SET_BIT(var, pos)          ((var) |= (1ULL << (pos)))
#define SET_BIT_VAL(var, pos, val) ((var) = ((var) & ~(1ULL << (pos))) | ((val) << (pos)))

namespace pffdtd {
void allocate_zeros(void** arr, uint64_t Nbytes);
void sort_keys(int64_t* val_arr, int64_t* key_arr, int64_t N);

template<typename T>
[[nodiscard]] auto allocate(std::integral auto count) -> T* {
  return reinterpret_cast<T*>(std::malloc(static_cast<size_t>(count) * sizeof(T)));
}

} // namespace pffdtd
