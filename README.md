# Introduction

SDS1-SHARAD is an implementation of processing algorithms for the
SHARAD planetary sounding radar.

SDS1-SHARAD starts with data products received from the
PDS archive at https://pds-geosciences.wustl.edu/mro/mro-m-sharad-3-edr-v1/
and processes them into various derived data products.

## Processing Step Scripts

The primary processing scripts which produce named
data products are in the SHARAD directory, and include

- cmp - `run_rng_cmp.py`
- alt - `run_altimetry.py`
- srf - `run_surface.py`
- rsr - `run_rsr.py`
- rng - `run_ranging.py`
- foc - `run_sar2.py`

`pipeline.py` orchestrates the running of these scripts in a sensible
sequence based on what processing steps depend on other steps.

These scripts assume that a mirrored copy of the SHARAD EDR data
archive at https://pds-geosciences.wustl.edu/mro/mro-m-sharad-3-edr-v1/ been
has been placed at `$SDS/orig/supl/xtra-pds/SHARAD/EDR/`, where
`$SDS` is an environment variable.

These processing step scripts may be called with the name of a data product, such as:

```
./run_rng_cmp.py e_1592001_001_ss11_700_a
```

This will look to the SHARAD EDR data archive, get the science data
for this data product (which it finds at `$SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr15xxx/edr1592001/e_1592001_001_ss11_700_a.lbl`),
pulse compress it, and place it in a parallel directory structure at
 `$SDS/targ/xtra/SHARAD/cmp/mrosh_0001/data/edr15xxx/edr1592001/ion/`.

Multiple data products can be specified on the command line, and they can
also be provided via a text file from the `--tracklist` option.



# Getting SDS

```
git clone /disk/kea/SDS/repo/sds.git
```

# Dependencies

pip Package dependencies:

See `requirements.txt`.  To install, run

```
pip install -r requirements.txt
```

# Environment

Before you run code in SDS, in each terminal, to setup your environment, including path, run

```
$ source env/cshrc
```
