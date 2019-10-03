#!/bin/env python2.7

#import math
#import argparse
#import os

import numpy as np

#import dechirp
#import focus

def load_sharad(labelpath, radpath):
    """ Load sharad data and labels """
    rad_lbl = np.getfromtext(labelpath, delimiter='=')
    rad = np.fromfile(radpath, dtype='i1')
