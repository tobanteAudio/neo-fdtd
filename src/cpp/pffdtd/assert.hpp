// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#undef NDEBUG
#include <cassert>

#define PFFDTD_ASSERT(stmt) assert(stmt)

namespace pffdtd {
template<typename>
inline constexpr auto always_false = false;
}
