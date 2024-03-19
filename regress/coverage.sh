#!/bin/bash -ex


# Requires coverage pip package
# pip install coverage
# to use:
# ./coverage.sh
# coverage3 report -m

# A recommended report command, which excludes reporting of some external libraries:
# coverage report -m --omit \
# '/disk/kea/SDS/code/work/ngg/sds_master/xlib/rsr/*,\
# /disk/kea/SDS/code/work/ngg/sds_master/xlib/subradar/*.py,\
# /disk/kea/SDS/code/work/ngg/sds_master/SHARAD/data_visualization.py'



S0=`basename $0`
D0=`dirname $0`

echo "$S0: Start coverage: " `date`
echo "$S0: Changing to $D0"
pushd $D0

RCFILE=`pwd`/.coveragerc
FLAGS="--rcfile=$RCFILE"
COV=coverage3

# Turn off display variable for regressions. (prevent graph windows from activating)
export DISPLAY=""

OUT1=./covdata/targ/xtra/SHARAD
OUT2=./covdata/targ/xtra/SHARAD2

# Suppress numexpr messages
export NUMEXPR_MAX_THREADS=8

rm -rf ./covdata/
$COV run $FLAGS ../xlib/clutter/parse_channels.py > /dev/null

######################################
# Run placeholders
# TODO: run_specularity
for NAME in ../SHARAD/pipeline.py ../SHARAD/data_visualization.py ../SHARAD/run_ranging.py \
../SHARAD/run_rng_cmp.py ../SHARAD/run_rsr.py ../SHARAD/run_altimetry.py \
../xlib/clutter/unfoc_KMS2.py
do
    echo "$S0: Running $NAME -h"
    $COV run $FLAGS -a $NAME -h > /dev/null
done

# These are really slow, so allow them to be disabled during testing
RUNSLOW=0
if [ "$RUNSLOW" -eq "1" ]
then
    $COV run $FLAGS -a ../xlib/clutter/filter_ra.py --selftest 1 1 1 1 1 1 1
    for NAME in ../xlib/clutter/interferometry_funclib.py \
    ../xlib/rng/icsim.py
    do
        $COV run $FLAGS -a $NAME
    done
fi

for NAME in ../xlib/sar/sar.py ../xlib/altimetry/treshold.py ../xlib/altimetry/beta5.py \
../xlib/clutter/radargram_reprojection_funclib.py ../xlib/clutter/interface_picker.py \
../xlib/clutter/peakint.py \
../xlib/misc/coord.py ../xlib/misc/hdf.py ../xlib/rot/trafos.py  \
../xlib/rot/mars.py ../xlib/cmp/plotting.py \
../xlib/cmp/rng_cmp.py ../xlib/cmp/pds3lbl.py
do
    $COV run $FLAGS -a $NAME
done
# end placeholders
#########################################


$COV run $FLAGS -a ../xlib/misc/hdf_test.py -o ./covdata/
$COV run $FLAGS -a ../xlib/rdr/solar_longitude.py
$COV run $FLAGS -a ../xlib/misc/prog_test.py
$COV run $FLAGS -a ../xlib/cmp/test_pds3lbl.py

$COV run $FLAGS -a ../SHARAD/SHARADEnv.py > /dev/null
$COV run $FLAGS -a ./test_sharadfiles.py

$COV run $FLAGS -a ../xlib/cmp/pds3lbl.py -o ./covdata/

echo $S0: CMD $COV run $FLAGS -a ../xlib/sar/smooth.py
$COV run $FLAGS -a ../xlib/sar/smooth.py
# No longer here
#echo $S0: CMD $COV run $FLAGS -a ../MARFA/zfile.py
#$COV run $FLAGS -a ../MARFA/zfile.py

echo $S0: pipeline
# Run pipeline on a fresh (uninitialized) output.
$COV run $FLAGS -a ../SHARAD/pipeline.py -n -vv --tracklist ./tracks_coverage.txt
# Don't do foc because it takes forever
$COV run $FLAGS -a ../SHARAD/pipeline.py -j 1 -o $OUT2 --maxtracks 1 --tracklist ./tracks_coverage.txt --tasks rsr
$COV run $FLAGS -a ../SHARAD/pipeline.py -j 1 -o $OUT2 --ignoretimes --maxtracks 1 --tracklist ./tracks_coverage.txt --tasks cmp
$COV run $FLAGS -a ../SHARAD/pipeline.py -j 1 -o $OUT2 -n --tracklist ./tracks_coverage.txt
# Cause srf to be out of date
sleep 1
touch $OUT2/alt/mrosh_0001/data/edr19xxx/edr1920301/beta5/*.h5
$COV run $FLAGS -a ../SHARAD/pipeline.py -j 1 -o $OUT2 --tracklist ./tracks_coverage.txt
# Run pipeline on a partially complete output
rm -rf $OUT2/alt
$COV run $FLAGS -a ../SHARAD/pipeline.py -j 1 -o $OUT2 -n --tracklist ./tracks_coverage.txt

RNGDATA=./covdata/rng_cmp/
$COV run $FLAGS -a ../xlib/cmp/rng_cmp.py -o $RNGDATA -n --tracklist ./tracks_coverage.txt --maxtracks 1



echo $S0: run_rng_cmp
$COV run $FLAGS -a ../SHARAD/run_rng_cmp.py -n e_1920301_001_ss04_700_a DOESNOTEXIST || true
$COV run $FLAGS -a ../SHARAD/run_rng_cmp.py -o $OUT1 -j 1 --maxtracks 2 --tracklist ./tracks_coverage.txt
$COV run $FLAGS -a ../SHARAD/run_rng_cmp.py -o $OUT1 -j 1 e_0187401_007_ss19_700_a

echo $S0: run_altimetry
$COV run $FLAGS -a ../SHARAD/run_altimetry.py -o $OUT1 -j 1 --tracklist ./tracks_coverage.txt -n
$COV run $FLAGS -a ../SHARAD/run_altimetry.py -o $OUT1 -j 1 --maxtracks 1 --tracklist ./tracks_coverage.txt
# This doesn't work because we haven't updated our MROSH index since 2018 (see tracklist file)
#$COV run $FLAGS -a ../SHARAD/run_altimetry.py -o ./covdata/altimetry_data -j 1 \
#    --tracklist ./tracks_run_altimetry_fix202110.txt


echo "$S0: run_surface"
# show all files that would be processed and overwritten
$COV run $FLAGS -a ../SHARAD/run_surface.py -n --overwrite all
# show all files that would be processed if not done
$COV run $FLAGS -a ../SHARAD/run_surface.py -n all
$COV run $FLAGS -a ../SHARAD/run_surface.py -o $OUT1 -j 1 e_1920301_001_ss04_700_a

echo "$S0: run_rsr"
# show what happens processing all files
$COV run $FLAGS -a ../SHARAD/run_rsr.py -n all > /dev/null
# show all files that would be overwritten
$COV run $FLAGS -a ../SHARAD/run_rsr.py -n --overwrite all > /dev/null
# Run an orbit

nice $COV run $FLAGS -a ../SHARAD/run_rsr.py --output $OUT1 -s 2000 1920301


echo "$S0: run_ranging"
$COV run $FLAGS -a ../SHARAD/run_ranging.py --tracklist ./run_ranging__xover_idx.dat -o ./covdata/ranging_data/ --maxtracks 4 --jobs 1 -n
if [ "$RUNSLOW" -eq "1" ]
then
    $COV run $FLAGS -a ../SHARAD/run_ranging.py --noprogress --tracklist ./run_ranging__xover_idx.dat -o ./covdata/ranging_data/ --maxtracks 2 --jobs 1
fi


echo "$S0: interferometry"
$COV run $FLAGS -a ../MARFA/run_interferometry.py \
                 --project GOG3 --line GOG3/JKB2j/BWN01b/ \
                 --pickfile ../regress/pick_FOI_GOG3_JKB2j_BWN01b.npz \
                 --plot --save out_ri1
$COV run $FLAGS -a ../MARFA/run_interferometry.py \
                 --project GOG3 --line GOG3/JKB2j/BWN01b/ --mode Reference \
                 --pickfile ../regress/pick_FOI_GOG3_JKB2j_BWN01b.npz \
                 --refpickfile ../regress/pick_ref_GOG3_JKB2j_BWN01b_Stack15_KMS2.npz \
                 --plot --save out_ri2


####################################
RUN_SAR2_FLAGS="-o ./covdata/run_sar2_data -j 1 --maxtracks 1  --tracklist ./tracks_coverage.txt --params run_sar2_cov.json"
#$COV run $FLAGS -a ../SHARAD/run_sar2.py -o ./covdata/run_sar2_data -j 1 --maxtracks 1 --ofmt none --tracklist ./tracks_coverage.txt
echo $S0: run_sar2 -n
$COV run $FLAGS -a ../SHARAD/run_sar2.py $RUN_SAR2_FLAGS --ofmt none --focuser ddv2 -n
echo $S0: run_sar2 ddv2
$COV run $FLAGS -a ../SHARAD/run_sar2.py $RUN_SAR2_FLAGS --ofmt none --focuser ddv2
# Run ddv2 with no interpolation
$COV run $FLAGS -a ../SHARAD/run_sar2.py -j 1 --maxtracks 2 --tracklist ./tracks_coverage.txt  --ofmt hdf5 --focuser ddv2 --params run_sar2_cov2.json

echo $S0: run_sar2 mf
$COV run $FLAGS -a ../SHARAD/run_sar2.py -j 1 --maxtracks 3 --tracklist ./tracks_coverage.txt $RUN_SAR2_FLAGS --ofmt none --focuser mf --params run_sar2_mf_cov.json

# This one still takes too long with default params
#echo $S0: run_sar2 ddv1
$COV run $FLAGS -a ../SHARAD/run_sar2.py $RUN_SAR2_FLAGS --ofmt none --focuser ddv1

# End SAR
##########################################################


echo "$S0: data_visualization.py"
# These require a connection to an X11 display. allow them to error out
$COV run $FLAGS -a  ../SHARAD/data_visualization.py --selftest && true
$COV run $FLAGS -a  ../SHARAD/data_visualization.py --product cmp && true
$COV run $FLAGS -a  ../SHARAD/data_visualization.py --input '/disk/kea/SDS/targ/xtra/SHARAD/foc/mrosh_0001/data/edr10xxx/edr1058901/5m/3 range lines/30km/e_1058901_001_ss19_700_a_s.h5' && true
#---------------------------------------------

echo "$S0: End coverage: " `date` 
echo "$S0: coverage tests completed."
$COV report -m

