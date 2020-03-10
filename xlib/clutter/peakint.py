#!/usr/bin/env python3

"""
Peak interpolation
"""

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
    return (vmax_pos, vmax_val, None)

def qint(y, maxpos):
    if maxpos >= 2 and maxpos <= len(y) - 3:
        # Use 5-point fit
        return qint5(y[maxpos-2:maxpos+3])
    elif 0 < maxpos < len(y) - 1:
        return qint3(y[max(0, maxpos-1)], y[maxpos], y[min(len(y)-1, maxpos+1)])
    else: # return exactly; can't interpolate
        return (0, y[maxpos], None)


def main():
    # TODO: testing functions for peak interpolation
    pass

if __name__=="__main__":
    main()
