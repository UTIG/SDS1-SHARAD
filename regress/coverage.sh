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



rm -rf ./covdata/
 
$COV run $FLAGS ../xlib/clutter/parse_channels.py
$COV run $FLAGS -a ../xlib/misc/hdf_test.py -o ./covdata/
$COV run $FLAGS -a  ../xlib/rdr/solar_longitude.py
echo $S0: CMD $COV run $FLAGS -a  ../SHARAD/SHARADEnv.py
$COV run $FLAGS -a  ../SHARAD/SHARADEnv.py

echo $S0: CMD $COV run $FLAGS -a ../xlib/cmp/pds3lbl.py
$COV run $FLAGS -a ../xlib/cmp/pds3lbl.py -o ./covdata/

echo $S0: CMD $COV run $FLAGS -a ../xlib/sar/smooth.py
$COV run $FLAGS -a ../xlib/sar/smooth.py
echo $S0: CMD $COV run $FLAGS -a ../MARFA/zfile.py
$COV run $FLAGS -a ../MARFA/zfile.py

RNGDATA=./covdata/rng_cmp/
$COV run $FLAGS -a ../xlib/cmp/rng_cmp.py --maxtracks 1 --ofmt none
$COV run $FLAGS -a ../xlib/cmp/rng_cmp.py --maxtracks 1
$COV run $FLAGS -a ../xlib/cmp/rng_cmp.py --maxtracks 1 --ofmt hdf5

#DRYRUN=-n
DRYRUN=

RUN_SAR2_FLAGS="-o ./covdata/run_sar2_data -j 1 --maxtracks 1  --tracklist ./tracks_coverage.txt --params run_sar2_cov.json"
#$COV run $FLAGS -a ../SHARAD/run_sar2.py -o ./covdata/run_sar2_data -j 1 --maxtracks 1 --ofmt none --tracklist ./tracks_coverage.txt
echo $S0: run_sar2 -n
$COV run $FLAGS -a ../SHARAD/run_sar2.py $RUN_SAR2_FLAGS --ofmt none --focuser ddv2 -n
echo $S0: run_sar2 ddv2
$COV run $FLAGS -a ../SHARAD/run_sar2.py $RUN_SAR2_FLAGS --ofmt none --focuser ddv2
# Run ddv2 with no interpolation
$COV run $FLAGS -a ../SHARAD/run_sar2.py -j 1 --maxtracks 1 --tracklist ./tracks_coverage.txt  --ofmt hdf5 --focuser ddv2 --params run_sar2_cov2.json

echo $S0: run_sar2 mf
$COV run $FLAGS -a ../SHARAD/run_sar2.py -j 1 --maxtracks 1 --tracklist ./tracks_coverage.txt $RUN_SAR2_FLAGS --ofmt none --focuser mf --params run_sar2_mf_cov.json

# This one still takes too long with default params
#echo $S0: run_sar2 ddv1
$COV run $FLAGS -a ../SHARAD/run_sar2.py $RUN_SAR2_FLAGS --ofmt none --focuser ddv1


echo $S0: run_rng_cmp
$COV run $FLAGS -a ../SHARAD/run_rng_cmp.py -o ./covdata/rng_cmp_data -j 1 --maxtracks 1 --tracklist ./tracks_coverage.txt

echo $S0: run_altimetry
$COV run $FLAGS -a ../SHARAD/run_altimetry.py -o ./covdata/altimetry_data -j 1 --maxtracks 1 --tracklist ./tracks_run_altimetry_cov.txt -n
$COV run $FLAGS -a ../SHARAD/run_altimetry.py -o ./covdata/altimetry_data -j 1 --maxtracks 1 --tracklist ./tracks_run_altimetry_cov.txt

echo "$S0: run_rsr"
# show all files
$COV run $FLAGS -a ../SHARAD/run_rsr.py -n all
# show all files that would be deleted
$COV run $FLAGS -a ../SHARAD/run_rsr.py -n --delete all
# Run an orbit
nice $COV run $FLAGS -a ../SHARAD/run_rsr.py --ofmt none --output ./covdata/rsr_data/ -s 2000 1920301



$COV report -m

