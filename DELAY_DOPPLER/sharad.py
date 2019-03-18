#!/bin/env python2.7


import numpy as np
import dechirp
import focus
import math
import argparse
import os


def load_sharad(thePath):
	rad_lbl=np.getfromtext(theLabelPath,delimiter='=')
	rad=np.fromfile(thePath,dtype='i1')
	


