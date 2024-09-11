# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
"""This is a timer in a dict. Makes it easy to set up little timers.
Has a deconstructor to let you know if a timer started (tic) didn't end (toc).
"""
import time

#timer class using dict for sub-timers
class TimerDict:
    def __init__(self):
        self.d = {} #actual dict
        self.t = {} #to check toc'ed

    def __del__(self):
        for key in self.t.keys():
            if not self.t[key]:
                print(f'TimerDict: "{key}" never toc\'ed')

    #tic (start)
    def tic(self,key=0):
        self.d[key] = time.time()
        self.t[key] = False

    def inc(self,key=0,delta=0):
        self.d[key] -= delta

    #toc and print
    def toc(self,key=0,print_elapsed=True):
        assert key in self.d.keys()
        delta = time.time()-self.d[key]
        self.t[key] = True
        if print_elapsed:
            print('*TIMER %s: elapsed = %.4f s' % (key,delta))
        return delta

    #toc and pass back f-string
    def ftoc(self,key=0):
        assert key in self.d.keys()
        delta = time.time()-self.d[key]
        self.t[key] = True
        return f'** TIMED {key}: elapsed = {delta:.4f} s'

    #toc quietly
    def tocq(self,key=0):
        assert key in self.d.keys()
        self.t[key] = True
        return time.time()-self.d[key]
