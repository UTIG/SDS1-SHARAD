def mars_rot_model(model):
    '''
    This routine contains three different rotation models for Mars.

    Input
    -----
    model :
        'IAU2000' according to [1]_
        'Kuchynka2014' according to [2]_
        'Konopliv2016' rotation parameters from the kernel
                       mars_konopliv_2016.tpc

    Output
    ------
    return the rotation model to be applied as p499_modelname.

    References
    ----------
    .. [1] REPORT OF THE IAU/IAG WORKING GROUP ON CARTOGRAPHIC COORDINATES AND
           ROTATIONAL ELEMENTS OF THE PLANETS AND SATELLITES: 2000
           P. K. SEIDELMANN (CHAIR) 1 , V. K. ABALAKIN 2 , M. BURSA 3 ,
           M. E. DAVIES 4,† , C. DE BERGH 5 , J. H. LIESKE 6 , J. OBERST 7 ,
           J. L. SIMON 8 , E. M. STANDISH 6 , P. STOOKE 9 and P. C. THOMAS 10
    .. [2] Kuchynka et al. 2014, Icarus, 229, 340–347
           http://dx.doi.org/10.1016/j.icarus.2013.11.015

    Examples
    --------
    >>> import pydlr.la.mola as la_mola
    >>> import pydlr.misc.coord as crd
    >>> sph_coord = la_mola.mola_latlon(extent=[0, 5, 337, 340], verbose=True)
    >>> et = sph_coord[:, 3]
    >>> cartesian = crd.sph2cart(sph_coord[:, 0:3])
    >>> rot_model = mars_rot_model('IAU2000')
    >>> r_i = rot_model.r_i(cartesian, t=et)
    >>> rot_model = mars_rot_model('Kuchynka')
    >>> new_bodyfixed1 = rot_model.r_bf(r_i, t=et)
    >>> rot_model_new = mars_rot_model('Konopliv')
    >>> new_bodyfixed2 = rot_model_new.r_bf(r_i, t=et)
    '''
#
# More details:
#
# Change rotation model -> EXAMPLE:
# spherical = [lat, lon, radial]
# cartesian = crd.sph2cart(spherical)
# change of reference frame from IAU2000 to inertial J2000 (since all the
# coordinates from MOLA are always in IAU2000, we have to first convert to
# inertial J2000 and then to the body fixed coordinates of the new rot model)
# iner = p499.r_i(cartesian, t=ets)
# change to bodyfixed in the new rotation model
# new_bodyfixed = p499_K****.r_bf(iner, t=ets)
#
    import rot.trafos as rot_trafos
    if (model == 'IAU2000'):
        p499 = rot_trafos.P_rot([317.68143, -0.1061, 0],
                                [52.88650, -0.0609, 0.],
                                [176.630, 350.89198226, 0.],
                                i_dr=[0, 0, 0])
        return p499
    elif (model == 'Kuchynka2014'):
        p499_Kuchynka = rot_trafos.P_rot(
                        [317.269202, -0.10927547],
                        [54.432516, -0.05827105],
                        [176.049863, 350.891982443297],
                        nut_prec_ra=[0.000068, 0.000238, 0.000052,
                                     0.000009, 0.419057],
                        nut_prec_dec=[0.000051, 0.000141, 0.000031,
                                      0.000005, 1.591274],
                        nut_prec_ra_angles=[[198.991226, 226.292679,
                                             249.663391, 266.183510,
                                             79.398797],
                                            [19139.4819985, 38280.8511281,
                                            57420.7251593, 76560.636795,
                                            0.5042615]],
                        nut_prec_dec_angles=[[122.433576, 43.058401,
                                             57.663379, 79.476401, 166.325722],
                                             [19139.9407476, 38280.8753272,
                                              57420.7517205, 76560.6495004,
                                              0.5042615]],
                        lib_pm=[0.000145, 0.000157, 0.000040, 0.000001,
                                0.000001, 0.584542],
                        lib_angles=[[129.071773, 36.352167, 56.668646,
                                     67.364003, 104.792680, 95.391654],
                                    [19140.0328244/36525, 38281.0473591/36525,
                                     57420.9295360/36525, 76560.2552215/36525,
                                     95700.4387578/36525, 0.5042615/36525]])
        return p499_Kuchynka
    elif (model == 'Konopliv2016'):
        p499_Konopliv = rot_trafos.P_rot(
                        [315.34551871, -0.108649712784, 0],
                        [61.69239825, -0.061587333591, 0],
                        [173.30879242, 350.891982519523, 0],
                        nut_prec_ra=[0., 0., 0., 0., 0.,
                                     2.33559631, 0.00004628, -0.00001031,
                                     0.00013117, 0.,
                                     0.00001882, 0.00001116, -0.00001014,
                                     -0.00000041, -0.00008977,
                                     -0.00008600, -0.00011513, -0.00000051,
                                     -0.00000136, -0.00001764, -0.00004755,
                                     -0.00000059, -0.00000873, 0.00000035,
                                     -0.00000134],
                        nut_prec_dec=[0., 0., 0., 0., -8.80604547,
                                      0., -0.00080268, 0.00012392,
                                      0, 0.00079170, -0.00001272,
                                      -0.00000141, 0.00000251, 0.00000082,
                                      -0.00005458, 0.00020651, -0.00004026,
                                      -0.00000048, 0.00000772, 0.00001712,
                                      0.00001857, 0.00000097, 0.00000523,
                                      0., 0.00000083],
                        nut_prec_angles=[[169.51, 192.93, 53.47, 36.53,
                                         0, 90, 0, 90,
                                          190.02859433, 354.26708690,
                                          0, 90, 0, 90, 0,
                                          41.18790047, 90, 0, 90, 0,
                                          90, 0, 90, 0, 90],
                                         [-15916.2801, 41215163.19675,
                                          -662.965275, 662.965275,
                                          0.21134279, 0.21134279,
                                          19139.86461912,
                                          19139.81084919, 19139.85801553,
                                          19139.85801553, 19140.99045156,
                                          19141.16081386, 38279.76346293,
                                          38279.64898292, 38280.78360991,
                                          38280.88273809, 38280.96773580,
                                          57413.23685793, 57420.61182408,
                                          57420.61254870, 57420.76966903,
                                          76560.22756307, 76560.60395345,
                                          95700.82351052, 95700.45229604]],
                        lib_pm=[0., 0., 0., 0., -0.75667792, 3.32310358,
                                -0.00230232, -0.00025587, -0.00220746,
                                0.00006338, -0.00001442, -0.00000909,
                                -0.00000076, -0.00002912, 0.00019723,
                                -0.00009194, 0.00018709, 0.00000142,
                                -0.00003743, 0.00002073, 0.00007035,
                                0.00000270, 0.00000419, -0.00000028,
                                0.00000107],
                        lib_angles=[[169.51, 192.93, 53.47, 36.53,
                                     0.00, 90, 0, 90,
                                     190.02859433, 354.26708690,
                                     0, 90, 0, 90, 0,
                                     41.18790047, 90.00000000, 0.00000000,
                                     90, 0, 90, 0, 90, 0, 90],
                                    [-15916.2801/36525, 41215163.19675/36525,
                                     -662.965275/36525, 662.965275/36525,
                                     0.21134279/36525, 0.21134279/36525,
                                     19139.86461912/36525,
                                     19139.81084919/36525,
                                     19139.85801553/36525,
                                     19139.85801553/36525,
                                     19140.99045156/36525,
                                     19141.16081386/36525,
                                     38279.76346293/36525,
                                     38279.64898292/36525,
                                     38280.78360991/36525,
                                     38280.88273809/36525,
                                     38280.96773580/36525,
                                     57413.23685793/36525,
                                     57420.61182408/36525,
                                     57420.61254870/36525,
                                     57420.76966903/36525,
                                     76560.22756307/36525,
                                     76560.60395345/36525,
                                     95700.82351052/36525,
                                     95700.45229604/36525]])
        return p499_Konopliv
    else:
        print('unknown model for Mars')
        return None
