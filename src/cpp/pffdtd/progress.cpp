// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

#include "progress.hpp"

#include <fmt/format.h>

#include <cstdio>
#include <ctime>

#if defined(_WIN32)
  #include <windows.h>
#else
  #include <sys/ioctl.h>
#endif

namespace pffdtd {

namespace {

[[nodiscard]] auto getConsoleWidth() -> int {
#if defined(_WIN32)
  auto info = CONSOLE_SCREEN_BUFFER_INFO{};
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) {
    return info.srWindow.Right - info.srWindow.Left + 1;
  } else {
    return 80;
  }
#else
  auto w = winsize{};
  ioctl(0, TIOCGWINSZ, &w);
  return w.ws_col;
#endif
}

} // namespace

// hacky print progress (like tqdm)..
// N.B. this conflicts with tmux scrolling (stdout needs to flush)
// and not great for piping output to log (better to disable or change for those
// cases)
auto print(ProgressReport const& progress) -> void {
  auto const& p = progress;

  // progress bar (doesn't impact performance unless simulation is really tiny)
  auto const ncols  = getConsoleWidth();
  auto const ncolsl = 80;

  double const pcnt = (100.0 * p.n) / p.Nt;
  int const nlines  = 6;
  if (p.n > 0) {
    // back up
    for (int nl = 0; nl < nlines; nl++) {
      fmt::print("\033[1A");
    }
    // clear lines
    for (int nl = 0; nl < nlines; nl++) {
      for (int cc = 0; cc < ncols; cc++) {
        fmt::print(" ");
      }
      fmt::print("\n");
    }
    // back up
    for (int nl = 0; nl < nlines; nl++) {
      fmt::print("\033[1A");
    }
  }
  fmt::print("\n");
  // progress bar
  fmt::print("Running [{:.1f}%]", pcnt);
  if (ncols >= ncolsl) {
    for (int cc = 0; cc < (0.01 * pcnt * ncolsl); cc++) {
      fmt::print("=");
    }
    fmt::print(">");
    for (int cc = (0.01 * pcnt * ncolsl); cc < ncolsl; cc++) {
      fmt::print(".");
    }
  }
  double const est_total = p.elapsed * p.Nt / p.n;

  auto const sec = (int)p.elapsed;
  auto const h_e = (sec / 3600);
  auto const m_e = (sec - (3600 * h_e)) / 60;
  auto const s_e = (sec - (3600 * h_e) - (m_e * 60));

  auto const sec_e = (int)est_total;
  auto const h_t   = (sec_e / 3600);
  auto const m_t   = (sec_e - (3600 * h_t)) / 60;
  auto const s_t   = (sec_e - (3600 * h_t) - (m_t * 60));

  // clang-format off
  fmt::print("[");
  fmt::print("{:02d}:{:02d}:{:02d}<{:02d}:{:02d}:{:02d}]", h_e, m_e, s_e, h_t, m_t, s_t);
  fmt::println("");
  fmt::print("T: {:06.1f}", 1e-6 * p.Npts * p.n / p.elapsed); //"total" Mvox/s (averaged up to current time)
  fmt::print(" - ");
  fmt::print("I: {:06.1f}", 1e-6 * p.Npts / p.elapsedSample); // instantaneous Mvox/s (per time-step)
  fmt::print(" | ");
  fmt::print("TPW: {:06.1f}", 1e-6 * p.Npts * p.n / p.elapsed / p.numWorkers); // total per worker
  fmt::print(" - ");
  fmt::print("IPW: {:06.1f}", 1e-6 * p.Npts / p.elapsedSample / p.numWorkers); // inst per worker
  fmt::println("");

  fmt::print("TA: {:06.1f}", 1e-6 * p.Npts * p.n / p.elapsedAir); // total for air bit
  fmt::print(" - ");
  fmt::print("IA: {:06.1f}", 1e-6 * p.Npts / p.elapsedSampleAir); // inst for air bit

  fmt::println("");
  fmt::print("TB: {:06.1f}", 1e-6 * p.Nb * p.n / p.elapsedBoundary); // total for bn
  fmt::print(" - ");
  fmt::print("IB: {:06.1f}", 1e-6 * p.Nb / p.elapsedSampleBoundary); // inst for bn

  fmt::println("");

  fmt::print("T: {:02.1f}%", 100.0 * p.elapsedAir / p.elapsed); //% for air (total)
  fmt::print(" - ");
  fmt::print("I: {:02.1f}%", 100.0 * p.elapsedSampleAir / p.elapsedSample); //% for air (inst)
  fmt::println("");
  // clang-format on

  fflush(stdout);
}

} // namespace pffdtd
