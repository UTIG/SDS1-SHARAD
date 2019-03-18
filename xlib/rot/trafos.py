# Authors:
# -------
#    Alexander Stark (DLR), alexander.stark@dlr.de (AS)
#
# Version and Date:
# -----------------
# 0.0 - 24.10.2016 - AS
#       Porting from IDL code.
# 1.0 - 15.05.2017 - AS
#       Included documentation of P_rot class.
#

import numpy as np
from scipy.constants import degree as deg
# Julian century
cy = 36525

norm = np.linalg.norm
nsin = np.sin
ncos = np.cos
nasin = np.arcsin
nacos = np.arccos
natan2 = np.arctan2
nsqrt = np.sqrt


class P_rot():
    """
    Defines a rotational reference frame and provides methods for computation
    body-fixed from inertial (ICRF) coordinates and viceversa.

    Parameters
    ----------
    ra : array_like
         Right ascension secular terms up to degree 2.
    dec : array_like
          Declination secular terms up to degree 2.
    pm : array_like
         Prime meridian terms up to degree 2.
    lib_pm : array_like, optional
             Longitudinal librations amplitudes.
    lib_angles : [array_like, array_like], optional
                 Longitudinal librations phase and frequency.
                 Should be provided as [phase, frequencies].
                 Frequencies are expressed as degrees per day.
    nut_prec_ra : [array_like], optional
                  Amplitudes of precession and nutation of the rotation axis in
                  right ascension.
    nut_prec_dec : [array_like], optional
                   Amplitudes of precession and nutation of the rotation axis
                   in declination.
    nut_prec_angles: [array_like, array_like], optional
                     Precession and nutation of the rotation axis phase and
                     frequency. When the nut_prec_ra_angles or
                     nut_prec_dec_angles are not provided these values are
                     applied to right ascension and declination.
                     Should be provided as [phase, frequencies].
                     Frequencies are expressed as degrees per Julian centuries.
    nut_prec_ra_angles : [array_like, array_like], optional
                         Precession and nutation of the rotation axis phase and
                         frequency in right ascension. This overwrites values
                         of nut_prec_angles for the right ascension.
                         Should be provided as [phase, frequencies].
                         Frequencies are expressed as degrees per Julian
                         centuries.
    nut_prec_dec_angles : [array_like, array_like], optional
                          Precession and nutation of the rotation axis phase
                          and frequency in declination. This overwrites values
                          of nut_prec_angles for the declination.
                          Should be provided as [phase, frequencies].
                          Frequencies are expressed as degrees per Julian
                          centuries.
    i_dr : [array_like], optional
           Offset of the rotation center from the origin of the coordinate
           system.

    Examples
    --------
    >>> import pydlr.rot.trafos as rot_trafos
    >>> p199 = rot_trafos.P_rot([281.0103, -0.0328],
                         [61.4155, -0.0049],
                         [329.5988, 6.1385108])

    """
    def __init__(self, ra=0, dec=0, pm=0,
                 lib_pm=[], lib_angles=[],
                 nut_prec_ra=[], nut_prec_dec=[],
                 nut_prec_angles=[],
                 nut_prec_ra_angles=[], nut_prec_dec_angles=[],
                 i_dr=[0, 0, 0]):
        self.epoch = 0  # J2000.0
        self.ra = np.array(ra, dtype='d')
        self.dec = np.array(dec, dtype='d')
        self.pm = np.array(pm, dtype='d')
        self.nut_prec_ra = np.array(nut_prec_ra, dtype='d')
        self.nut_prec_dec = np.array(nut_prec_dec, dtype='d')
        if (len(nut_prec_angles) > 0):
            self.nut_prec_ra_angles = np.array(nut_prec_angles, dtype='d')
            self.nut_prec_dec_angles = np.array(nut_prec_angles, dtype='d')
        else:
            self.nut_prec_ra_angles = np.array(nut_prec_ra_angles, dtype='d')
            self.nut_prec_dec_angles = np.array(nut_prec_dec_angles, dtype='d')
        self.lib_pm = np.array(lib_pm, dtype='d')
        self.lib_angles = np.array(lib_angles, dtype='d')
        self.i_dr = np.array(i_dr, dtype='d')

    def librationParameters(self, resonance, moi_ratio, eccentricity, n0,
                            forcing=[]):
        """
        Computes the libration amplitudes, frequencies and phases for a
        specified resonance, orbit eccentricity ,mean motion value "n0", and
        moments of inertia ratio "moi_ratio" (expressed as (B-A)/C). If a
        "forcing" is specified the amplitudes at the forced frequencies are
        also computed.
        """
        def kaula_g(e, n):
            return np.array([
             (1 - 11 * (e**2) + (959 * (e**4))/48 - (3641 * (e**6))/288 +
              (11359 * (e**8))/2880 - (247799 * (e**10))/345600),
             (-(e/8) - (421 * (e**3))/96 + (32515 * (e**5))/3072 -
              (312409 * (e**7))/36864 + (61199959 * (e**9))/17694720),
             (-((533 * (e**4))/144) + (4609 * (e**6))/480 -
              (34709 * (e**8))/3840 + (146017 * (e**10))/32256),
             ((e**3)/768 - (57073 * (e**5))/15360 +
              (7678157 * (e**7))/737280 - (99360199 * (e**9))/8847360),
             ((e**4)/600 - (18337 * (e**6))/4500 +
              (12412549 * (e**8))/1008000 - (12903227 * (e**10))/864000)][0:n])
        a = []
        phi = []
        nu = []
        if (resonance == 1.5):
            harmonics = 5
            a = kaula_g(eccentricity, harmonics)/deg * moi_ratio * 1.5
            nharmonics = (np.arange(harmonics) + 1)
            phi = (n0[0] * nharmonics) % 360
            nu = n0[1] * nharmonics
        if (resonance == 1.0):
            # TODO: Implement libration for synchronous rotation
            print('Librations not implemented yet!')
        self.lib_pm = a
        self.lib_angles = np.asarray([phi, nu])

    def change_epoch(self, new_epoch):
        """
        Transforms the rotation parameters to the new reference epoch.
        """
        dt = new_epoch - self.epoch
        ra = []
        # update rotation parameters to new epoch
        for i in range(len(self.ra)):
            value = 0
            for j in range(i, len(self.ra)):
                value += pow(dt/cy, j-i) * self.ra[j]
            ra.append(value)
        self.ra = ra
        dec = []
        for i in range(len(self.dec)):
            value = 0
            for j in range(i, len(self.dec)):
                value += pow(dt/cy, j-i) * self.dec[j]
            dec.append(value)
        self.dec = dec
        pm = []
        for i in range(len(self.pm)):
            value = 0
            for j in range(i, len(self.pm)):
                value += pow(dt, j-i) * self.pm[j]
            pm.append(value)
        self.pm = pm
        if (len(self.lib_angles) > 0):
            self.lib_angles[0] = ((self.lib_angles[0] +
                                   self.lib_angles[1]*dt) % 360)
        if (len(self.nut_prec_ra_angles) > 0):
            self.nut_prec_ra_angles[0] = (
                (self.nut_prec_ra_angles[0] +
                 self.nut_prec_ra_angles[1]*(dt/cy)) % 360)
        if (len(self.nut_prec_dec_angles) > 0):
            self.nut_prec_dec_angles[0] = (
                (self.nut_prec_dec_angles[0] +
                 self.nut_prec_dec_angles[1]*(dt/cy)) % 360)
        self.epoch = new_epoch

    def euler_angles(self, t, angles=[1, 2, 3], indeg=True):
        """
        Computes the Euler angles for the specified epochs in "t".

        Parameters
        ----------
        t : array
            Time array in days.

        Output
        ------
        Array with Euler angles as specified by the "angels" keyword.
        """
        scl = deg
        if (indeg):
            scl = 1.0
        t = t - self.epoch
        nt = len(t)
        alpha = None
        delta = None
        omega = None
        if (1 in angles):
            # pole and secular term of right ascension
            alpha = np.zeros(nt)
            for i in range(len(self.ra)):
                alpha = alpha + pow(t/cy, i) * self.ra[i]
            # precession and nutation of right ascension
            for i in range(len(self.nut_prec_ra)):
                alpha = alpha + (
                            self.nut_prec_ra[i] *
                            nsin((self.nut_prec_ra_angles[0][i] +
                                  self.nut_prec_ra_angles[1][i] * t/cy)*deg))
            alpha = (alpha + 90) * scl
        if (2 in angles):
            # pole and secular term of declination
            delta = np.zeros(nt)
            for i in range(len(self.dec)):
                delta = delta + pow(t/cy, i) * self.dec[i]
            # precession and nutation of declination
            for i in range(len(self.nut_prec_dec)):
                delta = delta + (
                            self.nut_prec_dec[i] *
                            ncos((self.nut_prec_dec_angles[0][i] +
                                  self.nut_prec_dec_angles[1][i] * t/cy)*deg))
            delta = (90 - delta) * scl
        if (3 in angles):
            # rotation
            omega = np.zeros(nt)
            for i in range(len(self.pm)):
                omega = omega + pow(t, i) * self.pm[i]
            # libration
            for i in range(len(self.lib_pm)):
                omega = omega + (
                            self.lib_pm[i] *
                            nsin((self.lib_angles[0][i] +
                                  self.lib_angles[1][i] * t) * deg))
            omega = omega * scl
        return np.asarray([alpha, delta, omega])

    def dr_bf(self, coord, angle, t=None, t_col=3, par=None,
              euler_angles=None, indeg=False):
        """
        Computes the partial derivative of the rotation parameters for the
        Cartesian vectors specified by "coord".
        In order to avoid re-computation of Euler angles they can be provided
        by the keyword "euler_angles" (the time vector is not needed in this
        case and will be ignored). Using the keyword "indeg" the unit of the
        angles can be specified.
        The rotation parameters are categorized in to Euler angles angle and
        the index "par" as follows:
        angle φ = (1,2,3) -> (α,δ,ω)
        par = (1,2,3) -> (φ_0, φ_1, φ_2)
        of the parametrization φ(t) = φ_0  + φ_1  t + φ_2  t^2.
        """
        from scipy.constants import degree as deg
        scl = 1.0
        if (indeg):
            scl = deg
        ntime = coord.shape[0]
        if ((angle < 0) | (angle > 2)):
            print('ERROR: The specified value = '+str(angle) +
                  ' for the parameter angle is' +
                  ' not between 0 and 2!')
            return None
        # coord = np.reshape(coord, (-1, 3))
        if (euler_angles is None):
            if (t is None):
                if (coord.shape[1] > t_col):
                    t = coord[:, t_col]
                else:
                    print('ERROR: Either Euler angles nor ' +
                          'time vector provided!')
                    return None
                # endif time vector is in coord
            # endif time vector provided
            if (ntime == len(t)):
                [a, d, w] = self.euler_angles(t/86400, indeg=False)
            else:
                print('ERROR: Dimension of time vector and coordinates ' +
                      'do not match: '+str(len(t))+' != '+str(ntime)+' !')
            # endif dimensions do not match
        else:
            if (ntime == euler_angles.shape[1]):
                [a, d, w] = euler_angles*scl
            else:
                print('ERROR: Dimension of euler angles and coordinates ' +
                      'do not match: '+str(euler_angles.shape[1]) +
                      ' != '+str(ntime)+' !')
            # endif dimensions do not match
        # endif euler angles provided
        ca = ncos(a)
        sa = nsin(a)
        cd = ncos(d)
        sd = nsin(d)
        cw = ncos(w)
        sw = nsin(w)
        drbf = np.empty([ntime, 3])
        # TODO: Improve matrix multiplication
        for i in range(ntime):
            if(angle == 0):
                Ra = np.transpose(np.matrix([
                                             [-sa[i], -ca[i],   0   ],
                                             [ ca[i], -sa[i],   0   ],
                                             [  0   ,    0  ,   0   ]]))
            else:
                Ra = np.transpose(np.matrix([
                                             [ca[i], -sa[i],   0   ],
                                             [sa[i],  ca[i],   0   ],
                                             [  0  ,    0  ,   1   ]]))
            #
            if (angle == 1):
                Rd = np.transpose(np.matrix([[  0  ,    0  ,   0   ],
                                             [  0  , -sd[i], -cd[i]],
                                             [  0  ,  cd[i], -sd[i]]]))
            else:
                Rd = np.transpose(np.matrix([[  1  ,    0  ,   0   ],
                                             [  0  ,  cd[i], -sd[i]],
                                             [  0  ,  sd[i],  cd[i]]]))
            #
            if (angle == 2):
                Rw = np.transpose(np.matrix([[-sw[i], -cw[i],   0   ],
                                             [ cw[i], -sw[i],   0   ],
                                             [  0   ,   0   ,   0   ]]))
            else:
                Rw = np.transpose(np.matrix([[cw[i], -sw[i],   0   ],
                                             [sw[i],  cw[i],   0   ],
                                             [  0  ,   0   ,   1   ]]))
            #
            # if indeg=True the partial derivative would be given as length per
            # degree
            drbf[i] = scl*np.dot(Rw * Rd * Ra, coord[i] + self.i_dr)
        if not(par is None):
            if (angle == 0):
                drbf = (t[:, np.newaxis]/86400/cy)**par * drbf
            if (angle == 1):
                drbf = -(t[:, np.newaxis]/86400/cy)**par * drbf
            if (angle == 2):
                drbf = (t[:, np.newaxis]/86400)**par * drbf
        return drbf

    def r_bf(self, coord, t=None, t_col=3, euler_angles=None, indeg=False):
        """
        Transforms inertial Cartesian coordinates "coord" to body-fixed
        Cartesian coordinates. If "t" is not specified the time to compute the
        Euler angles is assumed to be part of the coord matrix at the column
        specified by "t_col".
        In order to avoid re-computation of Euler angles they can be provided
        by the keyword "euler_angles" (the time vector is not needed in this
        case and will be ignored). Using the keyword "indeg" the unit of the
        angles can be specified.
        """
        from scipy.constants import degree as deg
        # coord = np.reshape(coord, (-1, 3))
        ntime = coord.shape[0]
        if (euler_angles is None):
            if (t is None):
                if (coord.shape[1] > t_col):
                    t = coord[:, t_col]
                else:
                    print('ERROR: Either Euler angles nor ' +
                          'time vector provided!')
                    return None
                # endif time vector is in coord
            # endif time vector provided
            if (ntime == len(t)):
                [a, d, w] = self.euler_angles(t/86400, indeg=False)
            else:
                print('ERROR: Dimension of time vector and coordinates ' +
                      'do not match: '+str(len(t))+' != '+str(ntime)+' !')
            # endif dimensions do not match
        else:
            if (ntime == euler_angles.shape[1]):
                if indeg:
                    [a, d, w] = euler_angles*deg
                else:
                    [a, d, w] = euler_angles
            else:
                print('ERROR: Dimension of euler angles and coordinates ' +
                      'do not match: '+str(euler_angles.shape[1]) +
                      ' != '+str(ntime)+' !')
            # endif dimensions do not match
        # endif euler angles provided
        ca = ncos(a)
        sa = nsin(a)
        cd = ncos(d)
        sd = nsin(d)
        cw = ncos(w)
        sw = nsin(w)
        rbf = np.empty([ntime, 3])
        # TODO: Improve matrix multiplication
        for i in range(ntime):
            Ra = np.transpose(np.matrix([[ca[i], -sa[i],   0   ],
                                         [sa[i],  ca[i],   0   ],
                                         [  0  ,    0  ,   1   ]]))
            #
            Rd = np.transpose(np.matrix([[  1  ,    0  ,   0   ],
                                         [  0  ,  cd[i], -sd[i]],
                                         [  0  ,  sd[i],  cd[i]]]))
            #
            Rw = np.transpose(np.matrix([[cw[i], -sw[i],   0   ],
                                         [sw[i],  cw[i],   0   ],
                                         [  0  ,   0   ,   1   ]]))
            #
            rbf[i] = np.dot(Rw * Rd * Ra, coord[i] + self.i_dr)
        return rbf

    def r_i(self, coord, t=None, t_col=3, euler_angles=None, indeg=False):
        """
        Transforms body-fixed Cartesian coordinates to intertial Cartesian
        coordinates. If "t" is not specified transforms inertial Cartesian
        coordinates "coord" to body-fixed Cartesian coordinates.
        If "t" is not specified the time to compute the Euler angles is assumed
        to be part of the coord matrix at the column specified by "t_col".
        In order to avoid re-computation of Euler angles they can be provided
        by the keyword "euler_angles" (the time vector is not needed in this
        case and will be ignored). Using the keyword "indeg" the unit of the
        angles can be specified.
        """
        # coord = np.reshape(coord, (-1, 3))
        ntime = coord.shape[0]
        if (euler_angles is None):
            if (t is None):
                if (coord.shape[1] > t_col):
                    t = coord[:, t_col]
                else:
                    print('ERROR: Either Euler angles nor ' +
                          'time vector provided!')
                    return None
                # endif time vector is in coord
            # endif time vector provided
            if (ntime == len(t)):
                [a, d, w] = self.euler_angles(t/86400, indeg=False)
            else:
                print('ERROR: Dimension of time vector and coordinates ' +
                      'do not match: '+str(len(t))+' != '+str(ntime)+' !')
            # endif dimensions do not match
        else:
            if (ntime == euler_angles.shape[1]):
                if indeg:
                    [a, d, w] = euler_angles*deg
                else:
                    [a, d, w] = euler_angles
            else:
                print('ERROR: Dimension of euler angles and coordinates ' +
                      'do not match: '+str(euler_angles.shape[1]) +
                      ' != '+str(ntime)+' !')
            # endif dimensions do not match
        # endif euler angles provided
        ca = ncos(a)
        sa = nsin(a)
        cd = ncos(d)
        sd = nsin(d)
        cw = ncos(w)
        sw = nsin(w)
        ri = np.empty([ntime, 3])
        for i in range(ntime):
            Ra = np.transpose(np.matrix([[ca[i], -sa[i],   0   ],
                                         [sa[i],  ca[i],   0   ],
                                         [  0  ,    0  ,   1   ]]))
            #
            Rd = np.transpose(np.matrix([[  1  ,    0  ,   0   ],
                                         [  0  ,  cd[i], -sd[i]],
                                         [  0  ,  sd[i],  cd[i]]]))
            #
            Rw = np.transpose(np.matrix([[cw[i], -sw[i],   0   ],
                                         [sw[i],  cw[i],   0   ],
                                         [  0  ,   0   ,   1   ]]))
            #
            ri[i] = np.dot(np.transpose(Rw * Rd * Ra), coord[i]) - self.i_dr
        return ri


class P_q():
    """
    Constructor for a Helmert transformation (rotation, scaling and
    translation). If "quat" and "dr" are not provided the class is initialized
    with a unit transformation and no translation.
    """
    def __init__(self, quat=[1, 0, 0, 0], dr=[0, 0, 0]):
        self.quat = np.array(quat, dtype=float)
        self.dr = np.array(dr, dtype=float)

    def inverse(self):
        """
        Computes the inverse transformation and stores the resulting
        quaternion and translation vector as class attributes.
        """
        self.dr = -np.asarray(np.dot(quat2mat(self.quat), self.dr)).flatten()
        q = self.quat
        n_q = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
        self.quat = np.array([abs(q[0])/n_q,
                              -abs(q[1]) * np.sign(q[0]*q[1])/n_q,
                              -abs(q[2]) * np.sign(q[0]*q[2])/n_q,
                              -abs(q[3]) * np.sign(q[0]*q[3])/n_q])

    def r_q(self, rbf):
        """
        Applies the Helmert transformation on the Cartesian vectors in "rbf".
        """
        return np.asarray(np.dot(quat2mat(self.quat), (rbf + self.dr).T).T)

    def dr_q(self, rbf, par):
        """
        Computes the partial derivative of the transformation parameters for
        the Cartesian vectors specified by "rbf". The transformation parameters
        "par" are denoted as follows:
        (0,1,2,3) -> (q0, qx, qy, qz)
        (4,5,6) -> (dx, dy, dz)
        """
        q = self.quat
        n_q = norm(q)
        if (par == 0):
            # q0
            sq0 = q[1]**2 + q[2]**2 + q[3]**2
            R1 = q[0] * np.diag([q[0]**2 + 3*sq0]*3)
            R2 = -2*q[0]*np.matrix([[q[1]*q[1], q[1]*q[2], q[1]*q[3]],
                                    [q[2]*q[1], q[2]*q[2], q[2]*q[3]],
                                    [q[3]*q[1], q[3]*q[2], q[3]*q[3]]])
            R3 = 2*sq0*np.matrix([[    0  , -q[3],  q[2] ],
                                   [  q[3],   0  , -q[1] ],
                                   [ -q[2],  q[1],   0   ]])
            R = (R1 + R2 + R3)/(n_q**3)
            return np.asarray(np.dot(R, (rbf[:] + self.dr).T).T)
        elif (par == 1):
            # qx
            sq0 = q[1]**2 + q[2]**2 + q[3]**2
            sq1 = q[0]**2 + q[2]**2 + q[3]**2
            R1 = -q[1] * np.diag([3*q[0]**2 + sq0]*3)
            R2 = 2*np.matrix([[q[1]*(2*sq1+q[1]**2), q[2]*sq1, q[3]*sq1],
                              [q[2]*sq1, -q[1]*q[2]*q[2], -q[1]*q[2]*q[3]],
                              [q[3]*sq1, -q[1]*q[2]*q[3], -q[1]*q[3]*q[3]]])
            R3 = 2*q[0]*np.matrix([[     0    , q[1]*q[3],  -q[1]*q[2]],
                                   [-q[1]*q[3],    0     ,     -sq1   ],
                                   [ q[1]*q[2],   sq1    ,       0    ]])
            R = (R1 + R2 + R3)/(n_q**3)
            return np.asarray(np.dot(R, (rbf[:] + self.dr).T).T)
        elif (par == 2):
            # qy
            sq0 = q[1]**2 + q[2]**2 + q[3]**2
            sq2 = q[0]**2 + q[1]**2 + q[3]**2
            R1 = -q[2] * np.diag([3*q[0]**2 + sq0]*3)
            R2 = 2*np.matrix([[-q[1]*q[1]*q[2], q[1]*sq2, -q[1]*q[2]*q[3]],
                              [q[1]*sq2, q[2]*(2*sq2+q[2]**2), q[3]*sq2],
                              [-q[1]*q[2]*q[3], q[3]*sq2, -q[2]*q[3]*q[3]]])
            R3 = 2*q[0]*np.matrix([[     0    ,  q[2]*q[3],   sq2    ],
                                   [-q[2]*q[3],     0     , q[1]*q[2]],
                                   [   -sq2   , -q[1]*q[2],    0     ]])
            R = (R1 + R2 + R3)/(n_q**3)
            return np.asarray(np.dot(R, (rbf[:] + self.dr).T).T)
        elif (par == 3):
            # qz
            sq0 = q[1]**2 + q[2]**2 + q[3]**2
            sq3 = q[0]**2 + q[1]**2 + q[2]**2
            R1 = -q[3] * np.diag([3*q[0]**2 + sq0]*3)
            R2 = 2*np.matrix([[-q[1]*q[1]*q[3], -q[1]*q[2]*q[3], q[1]*sq3],
                              [-q[1]*q[2]*q[3], -q[2]*q[2]*q[3], q[2]*sq3],
                              [q[1]*sq3, q[2]*sq3, q[3]*(2*sq3+q[3]**2)]])
            R3 = 2*q[0]*np.matrix([[     0    ,   -sq3    , -q[2]*q[3]],
                                   [    sq3   ,     0     ,  q[1]*q[3]],
                                   [ q[2]*q[3], -q[1]*q[3],     0     ]])
            R = (R1 + R2 + R3)/(n_q**3)
            return np.asarray(np.dot(R, (rbf[:] + self.dr).T).T)
        elif (par == 4):
            # dx
            rx = quat2mat(self.quat)[:, 0]
            return np.array([list(np.asarray(rx).flatten())]*len(rbf))
        elif (par == 5):
            # dy
            ry = quat2mat(self.quat)[:, 1]
            return np.array([list(np.asarray(ry).flatten())]*len(rbf))
        elif (par == 6):
            # dz
            rz = quat2mat(self.quat)[:, 2]
            return np.array([list(np.asarray(rz).flatten())]*len(rbf))
        else:
            print('ERROR: Unknown parameter '+str(par)+'!')
            print('Valid parameter values are: {0..3} -> {q0, qx, qy, qz} ' +
                  'and {4,5,6} -> {dx, dy, dz}')
            return None
        # endif par


def quat2mat(q):
    """
    Computes a transformation matrix (rotation + scaling) from the quaternion
    "q".
    """
    q = np.asarray(q)
    scl = norm(q)
    q = q/scl
    f1 = np.dot(q**2, np.array([1, -1, -1, -1]))
    Rq = scl*(np.diag(np.array([f1, f1, f1])) +
              2 * np.outer(q[1:4], q[1:4]) +
              2 * q[0] * np.matrix([[   0  , -q[3],  q[2] ],
                                    [  q[3],   0  , -q[1] ],
                                    [ -q[2],  q[1],   0   ]]))
    return Rq


def mat2quat(mat):
    """
    Computes the quaternion associated to the transformation "mat".
    If the matrix is singular an error is signaled.
    """
    det = np.linalg.det(mat)
    if not(det == 0):
        scl = 0.5 * det**(1/3)
    else:
        print('ERROR: Matrix is not unitary!')
        return None
    mat = mat * det**(-1/3)
    tr = np.trace(mat)
    d = np.insert(np.diag(mat), 0, 1)
    q1 = scl * nsqrt(1+tr)
    qx = scl * np.sign(mat[2, 1]-mat[1, 2])*nsqrt(np.dot(d, [1, 1, -1, -1]))
    qy = scl * np.sign(mat[0, 2]-mat[2, 0])*nsqrt(np.dot(d, [1, -1, 1, -1]))
    qz = scl * np.sign(mat[1, 0]-mat[0, 1])*nsqrt(np.dot(d, [1, -1, -1, 1]))
    return [q1, qx, qy, qz]
