#!/bin/bash -e


# Requires coverage pip package
# pip install coverage
# to use:
# ./coverage.sh
# coverage3 report -m


S0=`basename $0`
D0=`dirname $0`

echo "Changing to $D0"
pushd $D0

RCFILE=`pwd`/.coveragerc
FLAGS="--rcfile=$RCFILE"
COV=coverage3


 
$COV run $FLAGS ../xlib/clutter/parse_channels.py
$COV run $FLAGS -a  ../xlib/rdr/solar_longitude.py
echo $S0: CMD $COV run $FLAGS -a  ../SHARAD/SHARADEnv.py
$COV run $FLAGS -a  ../SHARAD/SHARADEnv.py

echo $S0: CMD $COV run $FLAGS -a ../xlib/cmp/pds3lbl.py
$COV run $FLAGS -a ../xlib/cmp/pds3lbl.py

RNGDATA=./rng_cmp_data/
rm -rf $RNGDATA

$COV run $FLAGS -a ../xlib/cmp/rng_cmp.py --maxtracks 1 --ofmt none
$COV run $FLAGS -a ../xlib/cmp/rng_cmp.py --maxtracks 1
rm -rf $RNGDATA
$COV run $FLAGS -a ../xlib/cmp/rng_cmp.py --maxtracks 1 --ofmt hdf5

rm -rf $RNGDATA
echo "$S0: run_rsr"
RSRDATA=./rsr_data/
rm -rf $RSRDATA
$COV run $FLAGS -a ../SHARAD/run_rsr.py -n all
$COV run $FLAGS -a ../SHARAD/run_rsr.py -n --delete all
nice $COV run $FLAGS -a ../SHARAD/run_rsr.py --ofmt none --output $RSRDATA 1920301
rm -rf $RSRDATA



$COV report -m
