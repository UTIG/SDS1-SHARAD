# Introduction

SDS1-SHARAD is an implementation of processing algorithms for the
SHARAD planetary sounding radar.

SDS1-SHARAD starts with data products received from the
PDS archive at https://pds-geosciences.wustl.edu/mro/mro-m-sharad-3-edr-v1/
and processes them into various derived data products.

## Processing Step Scripts

The primary processing scripts which produce named
data products are in the `SHARAD` directory, and include

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

### Processing script standard usage and options

These processing step scripts may be called with the name of a data product, such as:

```
./run_rng_cmp.py e_1592001_001_ss11_700_a
```

This will look to the SHARAD EDR data archive, get the science data
for this data product (which it finds at `$SDS/orig/supl/xtra-pds/SHARAD/EDR/mrosh_0001/data/edr15xxx/edr1592001/e_1592001_001_ss11_700_a.lbl`),
pulse compress it, and place it in a parallel directory structure at
 `$SDS/targ/xtra/SHARAD/cmp/mrosh_0001/data/edr15xxx/edr1592001/ion/`.

Multiple data products can be specified on the command line, and they can
also be provided via a text file from the `--tracklist` option.  Products
specified in the text file are appended after those specified on the
command line.

Processing outputs are automatically organized by data product type and SHARAD EDR product ID.
To change the base output directory, use the `--output` flag.  Providing
`--output $SDS/targ/xtra/SHARAD` is equivalent to the default behavior.

The path to the processing input can't be changed directly, but changing
the `$SDS` environment variable or providing `--SDS` option can effectively
allow you to change the input path.

By default, processing scripts will check whether output files
are up to date compared to their direct input prerequisites.
If they are up to date, or the required inputs do not exist,
the script will skip processing.  You can force outputs to be
overwritten with the `--overwrite` flag, or see what processing would
occur with the `-n` or `--dryrun` flag.


# Show processing status

The script `SHARAD/show_product_status.py`
follows a similar argument convention as the processing scripts,
and can show processing completion status for all data products.

# Getting SDS

```
git clone git@github.austin.utexas.edu:utig-reason/SDS1-SHARAD.git
```

# Dependencies

pip Package dependencies:

See `requirements.txt`.  To install, run

```
pip install -r requirements.txt
```

# Environment

Before you run code in SDS, in each terminal, to setup your environment, including path, run

```sh
$ source env/cshrc
```
