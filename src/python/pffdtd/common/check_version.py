# SPDX-License-Identifier: MIT

import sys

ATLEASTVERSION39 = sys.version_info >= (3, 9)
ATLEASTVERSION38 = sys.version_info >= (3, 8)
ATLEASTVERSION37 = sys.version_info >= (3, 7)
ATLEASTVERSION36 = sys.version_info >= (3, 6)
assert ATLEASTVERSION39
