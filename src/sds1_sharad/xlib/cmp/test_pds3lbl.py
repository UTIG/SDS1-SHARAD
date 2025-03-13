#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


import pds3lbl as pds3

SDS = os.getenv('SDS', '/disk/kea/SDS')
ORIG = os.path.join(SDS, "orig/supl/xtra-pds/SHARAD/EDR")

class TestBasic(unittest.TestCase):
    def test_read_science(self):
        """ Test basic PDS file read ancillary functionality """
        datafile = os.path.join(ORIG, "mrosh_0001/data/edr05xxx/edr0518201/e_0518201_001_ss19_700_a_s.dat")
        labelfile = os.path.join(ORIG, "mrosh_0001/label/science_ancillary.fmt")
        assert os.path.exists(datafile), datafile + " does not exist"
        assert os.path.exists(labelfile), labelfile + " does not exist"
        data = pds3.read_science(datafile, labelfile)


    def test_read_aux(self):
        """ Test basic PDS file read in non-science mode """
        datafile = os.path.join(ORIG, "mrosh_0001/data/edr05xxx/edr0518201/e_0518201_001_ss19_700_a_a.dat")
        labelfile = os.path.join(ORIG, "mrosh_0001/label/auxiliary.fmt")
        assert os.path.exists(datafile), datafile + " does not exist"
        assert os.path.exists(labelfile), labelfile + " does not exist"
        aux = pds3.read_science(datafile, labelfile)


    def test_missing_aux(self):
        """ Show what happens if you feed it a missing data or label file """
        datafile = os.path.join(ORIG, "mrosh_0001/data/edr05xxx/edr0518201/e_0518201_001_ss19_700_a_a.dat")
        labelfile = os.path.join(ORIG, "mrosh_0001/label/auxiliary.fmt")
        assert os.path.exists(datafile), datafile + " does not exist"
        assert os.path.exists(labelfile), labelfile + " does not exist"

        with self.assertRaises(FileNotFoundError):
            aux = pds3.read_science("ZZZMISSINGZZZ_a_a.dat", labelfile)

        with self.assertRaises(FileNotFoundError):
            aux = pds3.read_science(datafile, "ZZZMISSINGZZZ")

        with self.assertRaises(FileNotFoundError):
            aux = pds3.read_science("YYYMISSINGYYY_a_s.dat", "ZZZMISSINGZZZ")

    def test_missing_science(self):
        datafile = os.path.join(ORIG, "mrosh_0001/data/edr05xxx/edr0518201/e_0518201_001_ss19_700_a_s.dat")
        labelfile = os.path.join(ORIG, "mrosh_0001/label/science_ancillary.fmt")
        assert os.path.exists(datafile), datafile + " does not exist"
        assert os.path.exists(labelfile), labelfile + " does not exist"

        with self.assertRaises(FileNotFoundError):
            _ = pds3.read_science("ZZZMISSINGZZZ_a_s.dat", labelfile)

        with self.assertRaises(FileNotFoundError):
            _ = pds3.read_science(datafile, "ZZZMISSINGZZZ_a.lbl")

        with self.assertRaises(FileNotFoundError):
            _ = pds3.read_science("YYYMISSINGYYY_a_a.dat", "ZZZMISSINGZZZ")


    def test_diffname_samedir(self):
        """ Test case where there is a label in the same directory but it doesn't
        have the same name or capitalization """
        #E_0522001_001_SS19_700_A000.LBL

        datafile = os.path.join(ORIG, "mrosh_0001/data/edr05xxx/edr0518201/e_0518201_001_ss19_700_a_s.dat")
        labelfile = os.path.join(ORIG, "mrosh_0001/label/science_ancillary.fmt")
        aux = pds3.read_science(datafile, labelfile)


    def test_refchirp(self):
        chirppat = os.path.join(ORIG, 'mrosh_0001/calib/reference_chirp_*.dat')
        chirpfiles = sorted(glob.glob(chirppat))
        assert chirpfiles, "No chirp files found in %s" % chirppat
        for f in chirpfiles:
            with self.subTest(chirp=os.path.basename(f)):
                arr = pds3.read_refchirp(f)
                assert arr.shape[0] == 2048, "Unexpected chirp length: %r" % (arr.shape,)

class TestPseudoBits(unittest.TestCase):
    def setUp(self):
        self.datafiles = {
            # INSTRUMENT_MODE_ID=SS04 (8-bit)
            "SS04": [
                "mrosh_0001/data/edr19xxx/edr1945703/e_1945703_001_ss04_700_a_s.dat",
                "mrosh_0001/label/science_ancillary.fmt"],
            # INSTRUMENT_MODE_ID=SS11 (6-bit)
            "SS11": [
                "mrosh_0001/data/edr10xxx/edr1063402/e_1063402_001_ss11_700_a_s.dat",
                "mrosh_0001/label/science_ancillary.fmt"],
            # INSTRUMENT_MODE_ID=SS18 (4-bit)
            "SS18": [
                "mrosh_0001/data/edr01xxx/edr0169401/e_0169401_001_ss18_700_a_s.dat",
                "mrosh_0001/label/science_ancillary.fmt"],
        }
    def test_pseudo_bits(self):
        """ Test and write decoded binary data to temporary directory """
        with TemporaryDirectory() as fd:
            self.run_pseudo_bits(Path(fd))

    def run_pseudo_bits(self, outputdir=None):
        """ Execute optionally outputting to directory """
        orig = os.path.join(SDS, "orig/supl/xtra-pds/SHARAD/EDR")
        labelfile = os.path.join(orig, "mrsh_0004/label/science_ancillary.fmt")
        for name, (datarel, labelrel) in self.datafiles.items():
            with self.subTest(name=name):
                datafile = os.path.join(orig, datarel)
                labelfile = os.path.join(orig, labelrel)

                sreader = pds3.SHARADDataReader(labelfile, datafile)
                radar_samples = sreader.get_radar_samples()

                if outputdir is not None:
                    # Write to an output file
                    outfile = outputdir / Path(datarel).with_suffix('.bin').name
                    outputdir.mkdir(parents=True, exist_ok=True)
                    with outfile.open('wb') as fout:
                        radar_samples.tofile(fout)


if __name__ == "__main__":
    unittest.main()
