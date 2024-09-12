# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton

import multiprocessing as mp
from pathlib import Path


def get_default_nprocs():
    return max(1, int(0.8*mp.cpu_count()))


def clear_dat_folder(dat_folder_str=None):
    # clear dat folder
    dat_path = Path(dat_folder_str)
    if not dat_path.exists():
        dat_path.mkdir(parents=True)
    else:
        assert dat_path.is_dir()

    assert dat_path.exists()
    assert dat_path.is_dir()
    for f in dat_path.glob('*'):
        try:
            f.unlink()
        except OSError as e:
            print(f"Error: {f} : {e.strerror}")


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False
