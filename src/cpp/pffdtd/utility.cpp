// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

#include "utility.hpp"

#include "pffdtd/exception.hpp"

#include <cstdlib>

namespace pffdtd {

// sort and return indices
void sort_keys(int64_t* val_arr, int64_t* key_arr, int64_t N) {
  // for sorting int64 arrays and returning keys
  struct sort_int64_struct {
    int64_t val;
    int64_t idx;
  };

  // comparator with indice keys (for FCC ABC nodes)
  static constexpr auto cmpfunc_int64_keys = [](void const* a, void const* b) -> int {
    if ((*(sort_int64_struct const*)a).val < (*(sort_int64_struct const*)b).val) {
      return -1;
    }
    if ((*(sort_int64_struct const*)a).val > (*(sort_int64_struct const*)b).val) {
      return 1;
    }
    return 0;
  };

  auto* struct_arr = (sort_int64_struct*)malloc(N * sizeof(sort_int64_struct));
  if (struct_arr == nullptr) {
    raise<std::bad_alloc>();
  }
  for (int64_t i = 0; i < N; i++) {
    struct_arr[i].val = val_arr[i];
    struct_arr[i].idx = i;
  }
  qsort(struct_arr, N, sizeof(sort_int64_struct), cmpfunc_int64_keys);
  for (int64_t i = 0; i < N; i++) {
    val_arr[i] = struct_arr[i].val;
    key_arr[i] = struct_arr[i].idx;
  }
  free(struct_arr);
}

} // namespace pffdtd
