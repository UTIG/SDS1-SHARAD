#!/usr/bin/env python3

"""
Peak interpolation

# TODO: Unify qint3 and qint5 methods to have a more uniform prototype that we can swap

"""
import numpy as np

# From Julius Smith Spectral Audio Signal Processing
# peak finding algorithm (qint.m)
# https://ccrma.stanford.edu/~jos/sasp/Matlab_listing_qint_m.html
def qint3(ym1, y0, yp1):
    """
    QINT   Quadratic interpolation of 3 uniformly spaced samples
    Returns the extremum location p, height y, and half-curvature
    of a parabolic fit through 3 points.

    %       returns extremum-location p, height y, and half-curvature a
    %       of a parabolic fit through three points.
    %       The parabola is given by y(x) = a*(x-p)^2+b,
    %       where y(-1)=ym1, y(0)=y0, y(1)=yp1.

    given y[m] == max(y) for discretely sampled signal y

    return value:
    p is the value of x at the extremum (min or max)
    y is the value of the signal at y(p)
    a is the half-curvature of a parabolic fit through these three points

    """

    p = (yp1 - ym1)/(2*(2*y0 - yp1 - ym1))
    y = y0 - 0.25*(ym1 - yp1)*p
    a = 0.5*(ym1 - 2*y0 + yp1)

    return (p,y,a)


def qint5(y):
    """ Quadratic interpolation using 5 points.  Based on
    routines in pk3.c (by sdkempf) and qint.m by Julius Smith
    repackaged.
    Return value is position relative to
    /* Five point parabolic least means square residual fit */
    /* x = [-2, -1, 0, 1, 2]; */
    /* y = a + b * x + c * x .^ 2; */

    Returns the maximum position p, maximum value (y), half-curvature (TODO)
    """
    # double a, b, c;
    a = (-6.0*y[0] + 24.0*y[1] + 34.0*y[2] + 24.0*y[3] - 6.0*y[4])/70;
    b = (-2.0 * y[0] - y[1] + y[3] + 2.0 * y[4])/10;
    c = (y[0] + y[1] + y[2] + y[3] + y[4] - 5.0*a) / 10;

    # assert that this is a max, not a min. optional.
    # assert c < 0
    vmax_pos = -b / (2*c)
    vmax_val = a + b*vmax_pos + c * vmax_pos*vmax_pos
    # Should always be within one sample of the max.
    # assert -1 < vmax_pos < 1
    # TODO: verify half-curvature
    return (vmax_pos, vmax_val, c)

def qint(y, maxpos):
    if maxpos >= 2 and maxpos <= len(y) - 3:
        # Use 5-point fit
        return qint5(y[maxpos-2:maxpos+3])
    elif 0 < maxpos < len(y) - 1:
        return qint3(y[max(0, maxpos-1)], y[maxpos], y[min(len(y)-1, maxpos+1)])
    else: # return exactly; can't interpolate
        return (0, y[maxpos], None)


def test3():
    eps = 1e-6
    # TODO: test boundaries
    x = np.arange(-1, 13, 1)
    print("start test3()")
    for a in np.arange(-10, -0.1, 0.1):
        for h in np.arange(0.0, 10.0, 0.1):
            for k in np.arange(-10, 10.0, 2):
                # known vertex h, k
                data = a * ((x - h) ** 2) + k
                x1 = np.argmax(data)
                p, y, a = qint3(data[x1 - 1], data[x1], data[x1 + 1])
                p1  = x[x1] + p
                try:
                    assert abs(p1 - h) < eps
                    assert abs(y - k) < eps
                except AssertionError: #pragma: no cover
                    print("h,k=({:f},{:f}) p1, y, a = {:f} {:f} {:f}".format(h, k, p1, y, a))
                    raise
    print("end test3()")
def test5():
    """ test 5-point method """
    # TODO: test boundaries
    eps = 1e-6
    x = np.arange(-2, 15, 1)
    print("start test5()")
    for a in np.arange(-10, -0.1, 0.1):
        for h in np.arange(0.0, 10.0, 0.1):
            for k in np.arange(-10, 10.0, 2):
                # known vertex h, k
                data = a * ((x - h) ** 2) + k
                x1 = np.argmax(data)
                p, y, a = qint5(data[x1 - 2:x1 + 3])
                p1  = x[x1] + p
                try:
                    assert abs(p1 - h) < eps
                    assert abs(y - k) < eps
                except AssertionError: #pragma: no cover
                    print("h,k=({:f},{:f}) p1, y, a = {:f} {:f} {:f}".format(h, k, p1, y, a))
                    raise
    print("end test5()")


def test_qint():
    eps = 1e-6

    testdata = [
        [0.5, 1.0, 2.0, 1.0, 0.5],
        [0.5, 2.0, 1.0, 1.0, 0.5],
        # This fails equivalence if equal max values
        #[2.0, 2.0, 1.0, 1.0, 0.5],
        [3.0, 2.0, 1.0, 1.0, 0.5],
    ]

    for data in testdata:
        npdata = np.array(data)
        x1 = np.argmax(npdata)
        p1, y1, a1 = qint(npdata, x1)

        data.reverse()
        npdata = np.array(data)
        x2 = np.argmax(npdata)
        p2, y2, a2 = qint(npdata, x2)

        try:
            assert abs(p1 + p2) < eps # p1 == -p2
            assert abs(y1 - y2) < eps # y1 == y2
            assert a1 is None or abs(a1 - a2) < eps # a1 == a2
        except AssertionError: #pragma: no cover
            print("data={:s}; x1={:d} x2={:d} p1={:f} p2={:f}, y1={:f} y2={:f} "
                  "a1={:s} a2={:s}".format(str(data), x1, x2, p1, p2, y1, y2, str(a1), str(a2)))
            raise
        # TODO: check result

def main():
    """ testing functions for peak interpolation """
    test_qint()
    test3()
    test5()

if __name__=="__main__":
    main()
