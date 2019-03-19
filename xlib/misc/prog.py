#-*- coding: utf-8 -*-

"""
Authors:
-------------------------------------------------------------------------------

    Alexander Stark (DLR), alexander.stark@dlr.de

Description:
-------------------------------------------------------------------------------

    The class "Prog" provides the ability to perform progress reports of an
    long lasting calculation.


Input:
-------------------------------------------------------------------------------

    After initialization the progress is reported through a call of the method
    "print_Prog". The input parameters of this method are:

        loop     :: (Integer)
                    the value of a counting variable within the loop(s)
        step     :: (Integer)
                    at which percentage values to report the progreess
        kind     :: (Character)
                    percentage or progress bar output
        etc      :: (Boolean)
                    whether to print remaining Estimated Time of Computation
        eta      :: (Boolean)
                    whether to print Estimated Time of Arrival
        appendix :: (String)
                    some text to append to the status report

Usage:
-------------------------------------------------------------------------------

    loop = 0
    if (verbose):
        p = pr.Prog(n * m, step=10, kind='#', etc=False, end=True)
    for i in range(n):
        for j in range(m):
            loop += 1
            # do tasks
        if (verbose):
            p.print_Prog(loop, appendix='| processing i = '+str(i))
    p.close_Prog()

Version and Date
-------------------------------------------------------------------------------

   Version 1.0 -- December 13, 2016 -- Alexander Stark, DLR

      Initial release.

   2019-03-19 -- Gregory Ng UTIG
      Added ability to configure progress symbol.  Current symbol can cause
      UnicodeEncodeErrors exceptions in some cases.

"""
import time
import sys
import numpy as np

__version__ = '1.0'
__all__ = ['Prog']


class Prog():
    def __init__(self, length, step=0, kind='#',
        etc=True, eta=True,
        prog_symbol = '▨'):
        def isiterable(x):
            try:
                iter(x)
            except TypeError:
                return False
            else:
                return True

        self.t0 = time.time()
        if isiterable(length):
            self.length = len(length)
            self.keys = list(length)
        else:
            self.length = length
            self.keys = None
        self.thrsh = 0
        self.step = step
        self.kind = kind
        self.etc = etc
        self.eta = eta
        self.prog_symbol = prog_symbol

    def set_prog_symbol(self, symbol):
        self.prog_symbol = symbol

    def print_Prog(self, loop, appendix=''):
        if (self.keys is not None):
            loop = self.keys.index(loop)
        prog = 100.0 * (loop+1)/self.length
        if (prog >= self.thrsh):
            self.thrsh += self.step
            self.thrsh = min(self.thrsh, 100-1E-10)
            prog_t = (time.time()-self.t0)*(100.0/prog-1)/3600.0
            if (self.eta):
                eta_str = time.strftime(
                                "%a %d %b %Y %H:%M:%S",
                                time.localtime(time.time()+3600*prog_t))
                eta_str = ' | ETA ' + eta_str
            else:
                eta_str = ''
            if (self.etc):
                etc_str = " | -{:n}:{:02n}:{:02n}".format(
                               np.floor(prog_t),
                               np.floor(60*(prog_t - np.floor(prog_t))),
                               np.floor(60.0*(prog_t*60-np.floor(prog_t*60))))
            else:
                etc_str = ''
            if (self.kind == '%'):
                sys.stdout.write("\r%5.1f%%%s%s %s"
                                 % (prog,
                                    etc_str,
                                    eta_str,
                                    appendix))
            if (self.kind == '#'):
                # prog_symbol = '▬'
                # prog_symbol = '■'
                # prog_symbol = self.kind
                sys.stdout.write("\r[%s%s] %3i%%%s%s%s"
                                 % (self.prog_symbol * int(round(prog/5)),
                                    ' ' * (20-int(round(prog/5))),
                                    int(round(prog)),
                                    etc_str,
                                    eta_str,
                                    appendix))
            sys.stdout.flush()

    def close_Prog(self):
        print ('')
