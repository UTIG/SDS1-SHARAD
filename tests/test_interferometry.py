#!/usr/bin/env python3

""" Interferometry function library testing """
__authors__ = ['Gregory Ng, ngg@ig.utexas.edu']


import unittest
import sys
import os
from pathlib import Path
import logging

p1 = Path(__file__).parent / '..' / 'src' / 'sds1_sharad' / 'xlib' / 'clutter'
sys.path.insert(1, str(p1.absolute()))

#import sds1_sharad.xlib.clutter.interferometry_funclib as ifunc
from interferometry_funclib import * #denoise_and_dechirp, load_marfa, convert_to_complex

WAIS = os.getenv('WAIS', '/disk/kea/WAIS')

class TestLoad(unittest.TestCase):

    def test_load_marfa_s4foc(self):
        line = 'DEV2/JKB2t/Y81a'
        path = '/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S4_FOC'
        mag0, phs0 = load_marfa(line, '1', pth=path)

        assert mag0.shape == phs0.shape

        # Test a file that has a bad length
        line = 'ICP10/JKB2u/F01T01a'
        path = '/disk/kea/WAIS/targ/xtra/ICP10/FOC/Best_Versions/S4_FOC'
        mag, phs = load_marfa(line, '1', pth=path)
        return mag0, phs0

    def test_stacked_power_image(self):
        line = 'DEV2/JKB2t/Y81a'
        path = os.path.join(WAIS, 'targ/xtra/SRH1/FOC/Best_Versions/S4_FOC')
        mag, phs = load_marfa(line, '1', pth=path)

        assert mag.shape == phs.shape
        cmpA = convert_to_complex(mag, phs)
        cmpB = convert_to_complex(mag, -phs)
        rollphase = np.random.normal(loc=0, scale=0.1, size=(cmpA.shape[1],))
        for fresnel_stack in (1, 4, 15):
            for method in ('averaged', 'summed'):
                with self.subTest(fresnel_stack=fresnel_stack, method=method):
                    _ = stacked_power_image(mag, phs, mag, phs, fresnel_stack, method)

                    _ = stacked_correlation_map(cmpA, cmpB, fresnel_stack)
                    _ = stacked_interferogram(cmpA, cmpB, fresnel_stack, rollphase)




    def test_load_S2_bxds(self):
        # TODO: randomize for random file testing
        basepath = '/disk/kea/WAIS/targ/xtra'
        files = list(find_S2_bxds(basepath))
        logging.debug("Found {:d} files".format(len(files)))
        for i, filepath in enumerate(files, start=1):
            with self.subTest(file=filepath):
                logging.debug("[%2d] Loading %s", i, filepath)
                dirpath = os.path.dirname(filepath)
                channel = filepath[-3]
                bxds = load_S2_bxds(dirpath, channel)
                assert len(bxds)
            if i >= 9: # only process the first 10 files
                break


    def test_load_pik1(self):
        # TODO: test some RADnh5
        # /disk/kea/WAIS/targ/xped/ICP6/quality/xlob/pyk1.RADnh3/ICP6/JKB2k/F01T04a/MagLoResInco1
        # /disk/kea/WAIS/targ/xped/ICP6/quality/xlob/pyk1.RADnh3/ICP6/JKB2l/F02T01a/MagLoResInco1

        pth = os.path.join(os.getenv('WAIS', '/disk/kea/WAIS'), 'targ/xped/ICP6/quality/xlob/pyk1.RADnh3') + '/'
        line = 'DVG/MKB2l/Y06a/'

        logging.debug("pth=" + pth)
        logging.debug("line=" + line)

        list_channels = ['1', '2', '5', '6', '7', '8']

        for channel in list_channels:
            for IQ in ('mag','phs'):
                load_pik1(line, channel, pth=pth, IQ=IQ)
            with self.assertRaises(ValueError):
                load_pik1(line, channel, pth=pth, IQ='invalidtype')

    def test_load_power_image(self):
        logging.info("test_load_power_image()")
        path = os.path.join(WAIS, 'targ/xtra/SRH1/FOC/Best_Versions/S4_FOC/')

        # default value for trim
        trim = [0, 1000, 0, 0]
        #chirpwin = [0, 200]
        fresnel_stack = 15
        method = 'summed'

        for line in ('DEV2/JKB2t/Y81a/', 'DEV2/JKB2t/Y81a'): # allow either
            img = load_power_image(line, '1', trim, fresnel_stack, method, pth=path)

        line = 'DEV2/JKB2t/Y81a'
        for fresnel_stack in (1, 2, 15):
            for method in ('averaged','summed'):
                img = load_power_image(line, '1', trim, fresnel_stack, method, pth=path)


    def test_raw_bxds_load(self):
        """ Load orig data from MARFA """
        # /disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/AGAE/JKB2i/X5Aa/Xo
        testcases = [
            {
            'raw_path': "/disk/kea/WAIS/orig/xlob/DEV2/JKB2t/Y81a/RADnh5",
            'geo_path': "/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S2_FIL/DEV2/JKB2t/Y81a",
            'sigwin': [0, 1000, 0, 0],
            #}, { # 1 tear
            #'raw_path': "/disk/kea/WAIS/orig/xlob/NAQLK/JKB2j/ZY1a/RADnh3/",
            #'geo_path': "/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/NAQLK/JKB2j/ZY1a/",
            #'sigwin': [0, 1000, 0, 0]
            }, { # 1 tear
            'raw_path': "/disk/kea/WAIS/orig/xlob/NAQLK/JKB2j/ZY1b/RADnh3/",
            'geo_path': "/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/NAQLK/JKB2j/ZY1b/",
            'sigwin': [0, 1000, 0, 12000]
            #}, { # 0 tears -- no metadata.
            #'raw_path': "/disk/kea/WAIS/orig/xlob/CLEM/JKB2j/COL01a",
            #'geo_path': "/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/CLEM/JKB2j/COL01a",
            #'sigwin': [0, 1000, 0, 0]
            #}, { # 49 tears and a short read at the end
            ##/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S2_FIL/DEV/JKB2t/Y49a/NRt
            #'raw_path': "/disk/kea/WAIS/orig/xlob/DEV/JKB2t/Y49a/RADnh5",
            #'geo_path': "/disk/kea/WAIS/targ/xtra/SRH1/FOC/Best_Versions/S2_FIL/DEV/JKB2t/Y49a",
            #'sigwin': [0, 1000, 0, 0]

            },
        ]
        chan = '5'
        #testcases = (testcases[0],)
        for i, rec in enumerate(testcases, start=1):
            with self.subTest(raw_path=rec['raw_path']):
                logging.info("[%d of %d] Testing raw_bxds_load with %s",
                             i, len(testcases), rec['raw_path'])

                logging.debug("raw_bxds_load()")
                bxds1 = raw_bxds_load(rec['raw_path'], rec['geo_path'], chan, rec['sigwin'])

                """
                logging.debug("raw_bxds_load2()")
                bxds2 = raw_bxds_load2(rec['raw_path'], rec['geo_path'], chan, rec['sigwin'])

                assert bxds2.shape == bxds1.shape
                rmse = np.sqrt(np.square(abs(bxds2 - bxds1)).mean())
                logging.debug("RMSE(raw_bxds_load - raw_bxds_load2) = {:0.3g}".format(rmse))
                assert rmse < 1e-9
                """



class TestInterpolateResample(unittest.TestCase):
    """ Test functions related to interpolating and resampling radargrams """

    def test_frequency_shift(self, plot=False):
        for dx in (0.2, 0.1, 0.05):
            x1 = np.arange(0, 2.0, dx)
            y1 = fshiftfunc(x1)

            for upsamp in np.arange(2, 10):
                for offset in np.arange(0, upsamp):
                    with self.subTest(dx=dx, upsamp=upsamp, offset=offset):
                        self.run_frequency_shift(dx, upsamp, offset, x1, y1)

    def run_frequency_shift(self, dx:float, upsamp:int, offset:int, x1, y1):
        plot = False
        # Compute the actual functional value with a shift
        y2 = fshiftfunc(x1 - dx*offset/upsamp)  # calculate upsampled sine directly
        y3 = frequency_shift(y1, upsamp, offset) # calculate by old method
        y4 = frequency_shift2(y1, offset/upsamp) # calculate by new method

        try:
            assert (np.abs(y2 - y3) < 1e-6).all()
            assert (np.abs(y2 - y4) < 1e-6).all()
        except AssertionError as e: #pragma: no cover
            logging.warning("Assert error with upsamp=%d offset=%f", upsamp, offset)
            print(e)
            plot = True

        if plot or offset == 0 and upsamp == 2: # calculate at least once, for coverage
            plt.clf()
            plt.subplot(3,1,1)
            plt.plot(x1, y1, label='orig', marker='o', linewidth=0)
            plt.plot(x1, np.real(y2), label='y2real')
            plt.plot(x1, np.real(y3), label='y3real', marker='x', linewidth=0)
            plt.plot(x1, np.real(y4), label='y4real', marker='v', linewidth=0)
            plt.legend()
            plt.grid(True)
            plt.title('{:0.0f} / {:0.0f}'.format(offset, upsamp))
            plt.subplot(3,1,2);
            plt.plot(x1, np.zeros_like(x1), label='orig', marker='.', linewidth=0)
            plt.plot(x1, np.imag(y2), label='y2imag')
            plt.plot(x1, np.imag(y3), label='y3imag', marker='x', linewidth=0)
            plt.plot(x1, np.imag(y4), label='y4imag', marker='v', linewidth=0)
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(x1, np.real(y3 - y2), label='y3 - y2', marker='x', linewidth=1, color='g')
            plt.plot(x1, np.real(y4 - y2), label='y4 - y2', marker='v', linewidth=1, color='r')
            plt.title('{:0.0f} / {:0.0f} - Real Residual'.format(offset, upsamp))
            plt.legend()
            plt.grid(True)

        if plot: # pragma: no cover
            plt.show()



    def test_interpolate(self, bplot=False):
        """ Test equivalence of interpolation algorithms, and run with
        a variety of input sizes
        Test that average value and amplitude of signals mathes
        """

        for repeatsize in (128, 255, 77):
            self.run_repeat(repeatsize)

    def run_repeat(self, repeatsize:int):
        logging.debug("repeatsize = %d", repeatsize)
        meanval = 0.0

        # Series of step functions
        sig = np.repeat([0.5, 0., 1., 1., 0., 1., 0., 0., 1., 0.5], repeatsize)
        sig -= sig.mean() + meanval
        x = np.linspace(0, 100.0, len(sig))
        # noisy signal
        sig_noise = sig + 1e-3 * np.random.randn(len(sig))
        osi = 1 / 50e6 # original signal interval
        for ifactor in (2, 5, 10, 13, 21):
            with self.subTest(repeatsize=repeatsize, ifactor=ifactor):
                self.run_ifactor(repeatsize, ifactor, sig, sig_noise, osi, x)

    def run_ifactor(self, repeatsize:int, ifactor:int, sig, sig_noise, osi:float, x):
        x2 = np.linspace(0, max(x), len(sig_noise)*ifactor) # get interpolated indices
        sig_interp0 = np.interp(x2, x, sig_noise) # linear interpolation
        sig_interp1 = np.real(sinc_interpolate(sig_noise, osi, ifactor))
        sig_interp2 = np.real(frequency_interpolate(sig_noise, ifactor))

        rms1 = np.sqrt(np.square(abs(sig_interp0 - sig_interp1)).mean())
        rms2 = np.sqrt(np.square(abs(sig_interp0 - sig_interp2)).mean())
        rms3 = np.sqrt(np.square(abs(sig_interp1 - sig_interp2)).mean())
        logging.info("interpolate: ifactor=%0.1f RMS(lin-sinc)=%0.4g"
                     " RMS(lin-freq)=%0.4g RMS(sinc-freq)=%0.4g",
                     ifactor, rms1, rms2, rms3)


        statso = np.array([np.mean(sig_noise), np.std(sig_noise)])
        stats0 = np.array([np.mean(sig_interp0), np.std(sig_interp0)])
        stats1 = np.array([np.mean(sig_interp1), np.std(sig_interp1)])
        stats2 = np.array([np.mean(sig_interp2), np.std(sig_interp2)])

        #assert (np.abs(stats0 - stats1) < 1e-3).all()
        assert (np.abs(stats1 - stats2) < 1e-2).all(), \
        "statso=%r stats0=%r stats1=%r stats2=%r" % (statso, stats0, stats1, stats2)
        #assert (np.abs(statso - stats2) < 1e-3).all()

        assert rms3 < 5e-4, \
            "repeatsize=%d ifactor=%d RMS interpolation " \
            "difference: %f (limit %f)" % (repeatsize, ifactor, rms3, 5e-4)
        #bplot = True
        bplot = False

        if bplot: #pragma: no cover
            plt.subplot(211)
            plt.plot(x2, sig_interp0, x2, sig_interp1, x2, sig_interp2)
            plt.legend(['linear', 'sinc', 'frequency'])
            plt.subplot(212)
            plt.plot(abs(sig_interp1-sig_interp2))
            plt.title("Sinc vs Frequency Interpolation error")
            plt.show()


class TestCoregisterRadargrams(unittest.TestCase):
    """ Depends on interpolation """
    def test_coregistration(self):
        chanlist = {'low': ('5','7')} # not all high gain data exists, 'high': ('6','8')}
        # Load the focused SLC 1m port and starboard radargrams

        #"/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S2_FIL/NAQLK/JKB2j/ZY1a/"
        line = "NAQLK/JKB2j/ZY1a/"
        path = "/disk/kea/WAIS/targ/xtra/GOG3/FOC/Best_Versions/S4_FOC"
        trim = [0, 1000, 0, 12000]
        for name, chans in chanlist.items():
            logging.info("Coregistration " + name)
            cmpa = convert_to_complex(*load_marfa(line, chans[0], pth=path, trim=trim))
            cmpb = convert_to_complex(*load_marfa(line, chans[1], pth=path, trim=trim))
            logging.info("Coregistration done loading data")
            # orig_sample_interval is unused; TODO: remove
            for ifactor in range(1, 2, 4):
                for method in (0, 7):
                    with self.subTest(channel_name=name, ifactor=ifactor, method=method):
                        _ = coregistration(cmpa, cmpb, orig_sample_interval=None,
                                           upsample_factor=ifactor, method=method)


class TestUnfocDechirp(unittest.TestCase):
    """ Test functions that came from the unfoc library """
    def test_denoise_and_dechirp(self):
        gain = 'low'
        trim_default = [0, 1000, 0, 0]

        # Test with a RADnh3 line
        snms = {'SRH1': 'RADnh5'}

        inputs = [
            {'prj': 'SRH1', 'line': 'DEV2/JKB2t/Y81a', 'trim': trim_default },
            {'prj': 'GOG3', 'line': 'NAQLK/JKB2j/ZY1b', 'trim': [0, 1000, 0, 12000] },
            # Trimming at trim[2] == 15000 isn't supported yet
            #{'prj': 'GOG3', 'line': 'GOG3/JKB2j/BWN01a/', 'trim': [0, 1000, 15000, 27294] },
            {'prj': 'GOG3', 'line': 'GOG3/JKB2j/BWN01a/', 'trim': [0, 1000, 0, 27294] },
            {'prj': 'GOG3', 'line': 'GOG3/JKB2j/BWN01b/', 'trim': [0, 1000, 0, 15000] },
        ]

        for rec in inputs:
            with self.subTest(line=rec['line']):
                logging.debug("Processing line %s", rec['line'])
                snm = snms.get(rec['prj'], 'RADnh3') # get stream name
                path = os.path.join(WAIS, 'targ/xtra', rec['prj'], 'FOC/Best_Versions/')
                rawpath = os.path.join(WAIS, 'orig/xlob', rec['line'], snm)
                geopath = os.path.join(path, 'S2_FIL', rec['line'])
                chirppath = os.path.join(path, 'S4_FOC', rec['line'])
                chirp_bp = snm == 'RADnh5' # not strictly true, but pretty close
                #(gain, sigwin, raw_path, geo_path, chirp_path,
                #                    output_samples=3200, do_cinterp=True, bp=True):
                da, db = denoise_and_dechirp(gain, rec['trim'], rawpath, geopath, chirppath, do_cinterp=False, bp=chirp_bp)





def oldmain():
    parser = argparse.ArgumentParser(description='Interferometry Function Library Test')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose script output')
    parser.add_argument('--plot', action='store_true', help='Show debugging plots')

    args = parser.parse_args()

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=loglevel, stream=sys.stdout)

    magphs = test_load_marfa()
    test_stacked_power_image(*magphs)
    del magphs

    test_load_pik1()
    test_raw_bxds_load()
    test_load_S2_bxds()
    test_load_power_image()
    test_denoise_and_dechirp()
    test_frequency_shift(plot=args.plot)
    test_interpolate(bplot=args.plot)
    test_coregistration()


if __name__ == "__main__":
    unittest.main()

