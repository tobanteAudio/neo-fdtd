#pragma once

#include <fmt/format.h>

#include <stdexcept>

namespace pffdtd {

template<typename E, typename... Args>
[[noreturn]] auto raisef(fmt::format_string<Args...> str, Args&&... args)
    -> void {
  throw E{fmt::format(str, std::forward<Args>(args)...)};
}
} // namespace pffdtd
