#!/usr/bin/env python3
__authors__ = ['Kirk Scanlan, kirk.scanlan@utexas.edu', 'Gregory Ng, ngg@utexas.edu']
__version__ = '2.0'
__history__ = {
    '1.0': {
        'date': 'June 27, 2019',
        'author': 'Gregory Ng, UTIG',
        'info': 'Function for smoothing data that shows sample-and-hold steps'
    },
}

"""
This module implements algorithms to estimate smoothed, discrete-time data
from discrete-time data that has been sampled using a sample-and-hold method.

Example:

Let g(t) be a continuous function defined by

   g(t) = 10*t

Without loss of generality, t can represent any of time, ground-track distance,
angular position, etc.

Let y = h[t] be a discrete function defined by sampling g(t) for t as
shown below:

T = [0,  1,  2,  3,  4, 5,  6,  7,  8,  9,  10]
Y = [0, 10, 20, 30, 40 50, 60, 70, 80, 90, 100]

Sample-and-hold (also known as boxcar interpolation) can be implemented
by defining as a function h2[t]

h2[t] = { h[t1] for the greatest t1 in X such that t1 <= t} 



Mathematics of Sample-and-hold:
https://en.wikipedia.org/wiki/Zero-order_hold


Sample-and-hold reconstruction algorithms:

Interval reconstruction

(Kirk, this is a mathematical description of your approach)

One way to approximate an improvement to linear interpolation is to
find intervals with repeated y values, assume that the function's actual
value is the repeated value, at the center of that interval. That is,
If a discrete function g[t] = c for t in [t0, t1], let t2 = (t0+t1)/2.
We assume f( t2 ) = c is the actual value of the function.

After reconstructing a functional sampling, we can use this sampling
to estimate the original functional values using some interpolation such
as linear interpolation, spline interpolation, or even sinc interpolation.


Bandlimited interpolation



One way to reconstruct the original discrete-time function g[t] is to
assume that the original function g(t) contains no frequency components
greater than some frequency f.

Thus, you can take the function g[t] and apply a lowpass filter
with cutoff of 2*f.  This doesn't require us to make the "central value" assumption
from above.




TODO: explore whether this can be justified to be used in cases of amplitude quantization.

The algorithms in this module are not specifically optimized for situations you wish
to interpolate for effects due to amplitude quantization, but will probably give
pretty good results in those cases, too.

"""


import logging
import scipy.interpolate
import numpy as np


def vector_interp(vect):
    '''
    This is the function we're trying to rewrite in this module.

    Description:

    Given a vector that represents the values of a discrete-time function that has
    been upsampled using sample-and-hold interpolation, return a vector that 
    "improves" the interpolation method to linear interpolation, assuming that
    the center of the observed sample-and-hold intervals is the 
   



    Original description (kms2):

    Linear interpolation of a vector to smooth out steps. Values are plotted in
    middle of any discretized zones.

    Inputs:
    -------------
      vect: one-dimensional vector to be interpolated

     Outputs:
    -------------
     out: interpolated vector
    '''

    # interpolate a vector
    t1 = np.unique(vect)
    unix = np.zeros((len(t1), 1), dtype=float)
    len_vect = len(vect)
    for ii in range(len(t1)):
        t2 = np.abs(vect - t1[ii])
        t3 = np.argmin(t2) # find the farthest neighbor?
        # Finding the nearest neighbors to the current index
        t4 = abs(len_vect - np.argmin(np.flipud(t2)))
        unix[ii] = np.round(np.mean([t3, t4]))
    out = np.interp(np.arange(0, len(vect), 1), unix[:, 0], t1)

    return out

def vector_interp_wrapped(vect):
    v2 = vector_interp(vect)
    return v2, range(len(v2))

def get_minimums(arr):
    """
    Find the minimum value in the array, and the indices of
    all elements that match this value.
    """
    minv = None
    minidx = None
    for i, x in enumerate(arr):
        if minv is None or x < minv:
            minidx = [i]
            minv = x
        elif x == minv:
            minidx.append(i)
    return minv, minidx


def vector_interp2(vect, t=None, type='linear'):
    '''

    TODO: rename as reconstruct_sample_hold

    (previously vector_interp())

    Inputs:
    -------------
      vect: one-dimensional vector to be interpolated

     Outputs:
    -------------
     out: interpolated vector
    '''


    #vmin = np.amin(vect)
    #vmax = np.amax(vect)
    #step = (vmax-vmin) / (len(vect)-1)
    #out = np.arange(vmin, vmax + step, step)
    #assert len(out) == len(vect)
    #return out

    # interpolate a vector

    # v_uniq is a list of the unique values in this vector
    v_uniq = np.unique(vect)
    unix = []    

    len_vect = len(vect)
    for ii in range(len(v_uniq)):
        # Generate an array with the distance of each other array value
        # to this array value
        t2 = np.abs(vect - v_uniq[ii])

        #_, minidx = get_minimums(t2)
        # t3 is the distance of this value to the beginning of the vector
        t3 = np.argmin(t2)
        #print("ii={:d} t3={:d}".format(ii,t3))
        # Finding the nearest neighbors to the current index
        # flipud to get the lowest highest index in the original array that is closest
        # t4 is the index distance from the highest-indexed closest-argument to the end of the vector
        #t4 = abs(len_vect - minidx[-1]) # 
        t4 = abs(len_vect - np.argmin(np.flipud(t2)))
        #unix.append(  np.round(np.mean([t3, t4])) )
        unix.append(round( (t3+t4) / 2.0))
        #unix[ii] = np.round(np.mean([t3, t4]))
        #print("ii={:d} t3={:d} t4={:d}".format(ii,t3, t4))
    out = np.interp(np.arange(0, len_vect, 1), unix, v_uniq)
    #print(vect)
    #print(out)


    return out


def smooth(y, t=None, kind='linear', place='mid', epsilon=0.0):
    """
    Smooth data in a discrete-time series that has been
    interpolated using sample-and-hold interpolation.

    This algorithm first calls reduce_samphold, then interpolates
    points for x values given in t.

    Inputs:
    -------------
    y: numpy array representing data samples acquired using sample-and-hold, to be reduced
    t: nupy array of time (independent variable) corresponding to y. If
       t is none, then t is internally generated to be the indices of y
4~    kind: string
        String defining type of interpolation to perform for smoothing.
        See scipy interp1d.
    place: 
        Controls where to place returned samples.  See reduce_samphold.

    epsilon:
        Consecutive values with difference less than or equal to epsilon are considered to be equal.
     Outputs:
    -------------
     out: tuple (y2, t2)

       y2: Data samples corresponding to endpoints
       t2: Time values of each data sample returned. If t is provided, same as t.
    """

    if t is None:
        t = np.array(range(len(y)))

    y1, t1 = reduce_samphold(y, t, place, epsilon)

    # We allow extrapolation because it should only be for a short distance, but
    # hopefully this assumption doesn't break down.  
    fsmooth = scipy.interpolate.interp1d(t1, y1, kind, fill_value="extrapolate")
    try:
        y2 = fsmooth(t)
    except ValueError as e:
        logging.error("y1: " + str(y1))
        logging.error("t1: " + str(t1))
        logging.error("t:  " + str(t))
        raise e

    return y2, t


def reduce_samphold(y, t=None, place='mid', epsilon=0.0):
    """
    Reduce data to unique data points in a discrete-time series that has been
    interpolated using sample-and-hold interpolation.

    This algorithm finds intervals of equal values, and assumes them to be
    values repeated using sample-and-hold.

    TODO: make this algorithm a streaming generator

    Inputs:
    -------------
    y: numpy array representing data samples acquired using sample-and-hold, to be reduced
    t: nupy array of time (independent variable) corresponding to y. If
       t is none, then t is internally generated to be the indices of y
    place: 
         Controls where to place returned samples.
        'mid': place reduced samples in the middle of the interval (default)
        'low': place reduced samples at the least t value in the interval
        'high': place reduced samples at the greatest t value in the interval

    epsilon:
        Consecutive values with difference less than or equal to epsilon are considered to be equal.
     Outputs:
    -------------
     out: tuple (y2, t2)

       y2: Data samples corresponding to endpoints
       t2: Time values of each data sample returned


    """

    if t is None:
        t = np.array(range(len(y)))

    assert len(y) == len(t)
    outfunc = None

    # difference between consecutive samples
    dy =  np.abs(np.diff(y)) 

    # Find points where the value changes
    endpoints_idx = np.array(np.argwhere(dy > epsilon)[:, 0])
    endpoints_y = np.insert( y[endpoints_idx + 1], 0, y[0] )

    if place == 'mid':
        # get t for the low end of intervals
        endpoints_t1 = np.insert( t[endpoints_idx+1], 0, t[0] )
        # get t for the high end of intervals
        endpoints_t2 = np.append( t[endpoints_idx], t[-1])
        # get the average
        endpoints_t = (endpoints_t1 + endpoints_t2) / 2

        if len(endpoints_t) == 1:
            # for a constant function, we need more than one point
            endpoints_t = np.insert(endpoints_t, 0, t[0])
            endpoints_y = np.insert(endpoints_y, 0, y[0])

    elif place == 'low':
        raise Exception('Mode "low" has not been tested')
        # Return the point for each interval
        # Add one to offset from the first sample.
        # get t for the low end of the interval
        endpoints_t = np.concatenate(([np.array(t[0]), t[endpoints_idx+1]]))
        #endpoints_t = np.array([t[0], t[endpoints_idx+1]])
    elif place == 'high':
        raise Exception('Mode "high" has not been tested')
        # get t for the high end of the interval
        endpoints_t = np.concatenate((t[endpoints_idx], np.array(t[-1]) ))        

    return endpoints_y, endpoints_t

               



def test_vector_interp(ntests=100):
    # Test that vector_interp is equivalent to vector_interp2
    v = np.array(list(range(80)) + list(range(90,100,2)))

    w1 = vector_interp(v)
    w2 = vector_interp2(v)
    try:
        assert np.array_equal(w1, w2)
    except AssertionError as e:
        print(v)
        print(w1)
        print(w2)
        raise e

    for i in range(ntests):
        v = np.random.randint(1,500, 100)

        w1 = vector_interp(v)
        w2 = vector_interp2(v)
        try:
            assert np.array_equal(w1, w2)
        except AssertionError as e:
            print(v)
            print(w1)
            print(w2)
            raise e


def demo_vector_interp(funcname, func):
    logging.info("demo_vector_interp begin " + funcname)

    # TODO: if running reduce_samphold, then also try the different placements


    for hold_interval in (1,2,3,4):
        y1 = []

        for i in range(10):
            y1.extend([10*i,]*hold_interval )
        t = np.array(range(len(y1)))

        # Demonstrate problems with a strictly monotonic function
        y2,t2 = func(np.array(y1))
        plt.clf()
        plt.plot(t, y1, 'x')
        plt.plot(t2, y2, '+')
        plt.legend(['Original',funcname])
        plt.title('Function with repeat step {:d}'.format(hold_interval))
        plt.savefig('demo_{:s}{:d}.png'.format(funcname, hold_interval))


    # Show what happens in some pathological cases

    y1 = np.array([0,1,2,3,4,5,6,7,8,7,6,5,4,3,2,1])
    t = np.array(range(len(y1)))
    y2, t2 = func(y1)
    plt.clf()
    plt.plot(t, y1, 'x')
    plt.plot(t2, y2, '+')
    plt.legend(['Original',funcname])
    plt.title('Triangle function')
    plt.savefig('demo_{:s}_tri.png'.format(funcname))


    # Constant function

    y1 = np.array([0] * 10 )
    t = np.array(range(len(y1)))
    y2,t2 = func(y1)
    plt.clf()
    plt.plot(t, y1, 'x')
    plt.plot(t2, y2, '+')
    plt.legend(['Original',funcname])
    plt.title('Triangle function')
    plt.savefig('demo_{:s}_const.png'.format(funcname))

    logging.info("demo_vector_interp end")


def test_argmin(ntests=1000):
    print("test_argmin() start")
    for i in range(ntests):
        v = np.random.randint(1, 99, 400)


        #np.argmin(np.flipud(t2)
        #min1a = np.argmin(v)
        min2a = np.argmin(np.flipud(v))
        min2b = np.flip(np.argmin(          v ))

        try:
            assert np.array_equal(min2a, min2b)
        except AssertionError as e:
            print(v)
            print(min2a)
            print(min2b)
            raise e
    print("test_argmin() end")
    



def main():
    """ Test sar.py functions """
    parser = argparse.ArgumentParser(description='Sample-and-hold smoothing')
    parser.add_argument('-o', '--output', default='.', help="Output directory")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Display verbose output")

    args = parser.parse_args()


    loglevel = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format="sar: [%(levelname)-7s] %(message)s")

    # test_argmin()
    demo_vector_interp('vector_interp',vector_interp_wrapped)
    demo_vector_interp('reduce_samphold',reduce_samphold)
    demo_vector_interp('smooth',smooth)
    test_vector_interp()


if __name__ == "__main__":
    # execute only if run as a script
    import argparse
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main()


