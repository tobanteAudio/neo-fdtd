#include "progress.hpp"

#include <cstdio>
#include <ctime>
#include <sys/ioctl.h> //terminal width

namespace pffdtd {

// hacky print progress (like tqdm)..
// N.B. this conflicts with tmux scrolling (stdout needs to flush)
// and not great for piping output to log (better to disable or change for those
// cases)
void print_progress(
    uint32_t n,
    uint32_t Nt,
    uint64_t Npts,
    uint64_t Nb,
    double time_elapsed,
    double time_elapsed_sample,
    double time_elapsed_air,
    double time_elapsed_sample_air,
    double time_elapsed_bn,
    double time_elapsed_sample_bn,
    int num_workers
) {
  // progress bar (doesn't impact performance unless simulation is really tiny)
  struct winsize w;
  ioctl(0, TIOCGWINSZ, &w);
  int ncols  = w.ws_col;
  int ncolsl = 80;
  // int ncolsl = 120;
  // int ncolsp = w.ws_col-ncolsl;

  double pcnt = (100.0 * n) / Nt;
  int nlines  = 6;
  if (n > 0) {
    // back up
    for (int nl = 0; nl < nlines; nl++) {
      printf("\033[1A");
    }
    // clear lines
    for (int nl = 0; nl < nlines; nl++) {
      for (int cc = 0; cc < ncols; cc++) {
        printf(" ");
      }
      printf("\n");
    }
    // back up
    for (int nl = 0; nl < nlines; nl++) {
      printf("\033[1A");
    }
  }
  printf("\n");
  // progress bar
  printf("Running [%.1f%%]", pcnt);
  if (ncols >= ncolsl) {
    for (int cc = 0; cc < (0.01 * pcnt * ncolsl); cc++) {
      printf("=");
    }
    printf(">");
    for (int cc = (0.01 * pcnt * ncolsl); cc < ncolsl; cc++) {
      printf(".");
    }
  }
  double est_total = time_elapsed * Nt / n;

  int sec, h_e, m_e, s_e, h_t, m_t, s_t;
  sec = (int)time_elapsed;
  h_e = (sec / 3600);
  m_e = (sec - (3600 * h_e)) / 60;
  s_e = (sec - (3600 * h_e) - (m_e * 60));

  sec = (int)est_total;
  h_t = (sec / 3600);
  m_t = (sec - (3600 * h_t)) / 60;
  s_t = (sec - (3600 * h_t) - (m_t * 60));

  // clang-format off
  printf("[");
  printf("%02d:%02d:%02d<%02d:%02d:%02d]", h_e, m_e, s_e, h_t, m_t, s_t);
  printf("\n");
  printf("T: %06.1f", 1e-6 * Npts * n / time_elapsed); //"total" Mvox/s (averaged up to current time)
  printf(" - ");
  printf("I: %06.1f", 1e-6 * Npts / time_elapsed_sample); // instantaneous Mvox/s (per time-step)
  printf(" | ");
  printf("TPW: %06.1f", 1e-6 * Npts * n / time_elapsed / num_workers); // total per worker
  printf(" - ");
  printf("IPW: %06.1f", 1e-6 * Npts / time_elapsed_sample / num_workers); // inst per worker
  printf("\n");

  printf("TA: %06.1f", 1e-6 * Npts * n / time_elapsed_air); // total for air bit
  printf(" - ");
  printf("IA: %06.1f", 1e-6 * Npts / time_elapsed_sample_air); // inst for air bit

  printf("\n");
  printf("TB: %06.1f", 1e-6 * Nb * n / time_elapsed_bn); // total for bn
  printf(" - ");
  printf("IB: %06.1f", 1e-6 * Nb / time_elapsed_sample_bn); // inst for bn

  printf("\n");

  printf("T: %02.1f%%", 100.0 * time_elapsed_air / time_elapsed); //% for air (total)
  printf(" - ");
  printf("I: %02.1f%%", 100.0 * time_elapsed_sample_air / time_elapsed_sample); //% for air (inst)
  printf("\n");
  // clang-format on

  fflush(stdout);
}

} // namespace pffdtd
