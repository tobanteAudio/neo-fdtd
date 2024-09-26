// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

namespace pffdtd {

#define MMb 12 // maximum number of RLC branches in freq-dep (FD) boundaries
#define MNm 64 // maximum number of materials allows

// see python code and 2016 ISMRA paper
template<typename Real>
struct MatQuad {
  Real b;   // b
  Real bd;  // b*d
  Real bDh; // b*D-hat
  Real bFh; // b*F-hat
};

} // namespace pffdtd
