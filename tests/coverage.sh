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
XLIB=../src/sds1_sharad/xlib
SHARAD=../src/sds1_sharad/sharad
MARFA=../src/sds1_sharad/marfa
rm -rf ./covdata/


$COV run $FLAGS $XLIB/clutter/parse_channels.py > /dev/null

######################################
# Run placeholders
# TODO: run_specularity
for NAME in $SHARAD/pipeline.py \
$SHARAD/run_rng_cmp.py $SHARAD/run_rsr.py $SHARAD/run_altimetry.py \
$SHARAD/run_surface.py $SHARAD/show_product_status.py \
$XLIB/clutter/unfoc_KMS2.py
do
    echo "$S0: Running $NAME -h"
    $COV run $FLAGS -a $NAME -h > /dev/null
done


# These are really slow, so allow them to be disabled during testing
RUNSLOW=0
if [ "$RUNSLOW" -eq "1" ]
then
    $COV run $FLAGS -a $XLIB/clutter/filter_ra.py --selftest 1 1 1 1 1 1 1
    for NAME in $XLIB/clutter/interferometry_funclib.py \
    $SHARAD/run_ranging.py \
    $XLIB/rng/icsim.py $XLIB/rng/icd_test.py
    do
        $COV run $FLAGS -a $NAME
    done
fi
ENABLE_GDAL=0
ENABLE_TK=1
# These scripts require tk to be working properly
if [ "${ENABLE_TK}" -eq "1" ]
then
    $COV run $FLAGS -a $SHARAD/data_visualization.py -h
fi
for NAME in \
$XLIB/clutter/radargram_reprojection_funclib.py $XLIB/clutter/interface_picker.py
do
    $COV run $FLAGS -a $NAME
done



for NAME in $XLIB/sar/sar.py $XLIB/altimetry/treshold.py $XLIB/altimetry/beta5.py \
$XLIB/clutter/peakint.py \
$XLIB/misc/coord.py $XLIB/misc/hdf.py $XLIB/rot/trafos.py  \
$XLIB/rot/mars.py $XLIB/cmp/plotting.py \
$XLIB/cmp/rng_cmp.py $XLIB/cmp/pds3lbl.py \
$XLIB/rdr/solar_longitude.py \
$XLIB/misc/prog_test.py \
$XLIB/sar/smooth.py
do
    $COV run $FLAGS -a $NAME
done
# end placeholders
#########################################


$COV run $FLAGS -a $XLIB/misc/hdf_test.py -o ./covdata/
$COV run $FLAGS -a $XLIB/cmp/test_pds3lbl.py

$COV run $FLAGS -a ./test_sharadfiles.py
$COV run $FLAGS -a ./test_sharadenv.py
$COV run $FLAGS -a ./test_interferometry.py

$COV run $FLAGS -a $XLIB/cmp/pds3lbl.py -o ./covdata/

#echo $S0: CMD $COV run $FLAGS -a $XLIB/sar/smooth.py
# No longer here
#echo $S0: CMD $COV run $FLAGS -a $MARFA/zfile.py
#$COV run $FLAGS -a $MARFA/zfile.py

echo $S0: pipeline
# Run pipeline on a fresh (uninitialized) output.
$COV run $FLAGS -a $SHARAD/pipeline.py -n -vv --tracklist ./tracks_coverage.txt
# Don't do foc because it takes forever
$COV run $FLAGS -a $SHARAD/pipeline.py -j 1 -o $OUT2 --maxtracks 1 --tracklist ./tracks_coverage.txt --tasks rsr
$COV run $FLAGS -a $SHARAD/pipeline.py -j 1 -o $OUT2 --ignoretimes --maxtracks 1 --tracklist ./tracks_coverage.txt --tasks cmp
$COV run $FLAGS -a $SHARAD/pipeline.py -j 1 -o $OUT2 -n --ignoretimes --tracklist ./tracks_coverage.txt
# Cause srf to be out of date
sleep 1
touch $OUT2/alt/mrosh_0001/data/edr02xxx/edr0224401/beta5/*.{h5,i}
$COV run $FLAGS -a $SHARAD/pipeline.py -j 1 -o $OUT2 --tracklist ./tracks_coverage.txt
# Run pipeline on a partially complete output
rm -rf $OUT2/alt
$COV run $FLAGS -a $SHARAD/pipeline.py -j 1 -o $OUT2 -n --tracklist ./tracks_coverage.txt

# Show product status using actual hierarchy
$COV run $FLAGS -a $SHARAD/show_product_status.py --jsonout $OUT2/pstat/product_status.json --textout $OUT2/product_status.txt --maxtracks 1000
$COV run $FLAGS -a $SHARAD/show_product_status.py --maxtracks 10 > /dev/null


echo $S0: run_rng_cmp
$COV run $FLAGS -a $SHARAD/run_rng_cmp.py -n e_DOESNOTEXIST_a || true
$COV run $FLAGS -a $SHARAD/run_rng_cmp.py -n e_1920301_001_ss04_700_a
$COV run $FLAGS -a $SHARAD/run_rng_cmp.py -o $OUT1 -j 1 --maxtracks 2 --tracklist ./tracks_coverage.txt
$COV run $FLAGS -a $SHARAD/run_rng_cmp.py -o $OUT1 -j 1 e_0187401_007_ss19_700_a

echo $S0: run_altimetry
$COV run $FLAGS -a $SHARAD/run_altimetry.py -o $OUT1 -j 1 --tracklist ./tracks_coverage.txt -n
$COV run $FLAGS -a $SHARAD/run_altimetry.py -o $OUT1 -j 1 --maxtracks 1 --tracklist ./tracks_coverage.txt
$COV run $FLAGS -a $SHARAD/run_altimetry.py -o $OUT1 e_0224401_007_ss05_700_a
# This doesn't work because we haven't updated our MROSH index since 2018 (see tracklist file)
#$COV run $FLAGS -a $SHARAD/run_altimetry.py -o ./covdata/altimetry_data -j 1 \
#    --tracklist ./tracks_run_altimetry_fix202110.txt


echo "$S0: run_surface"
# show all files that would be processed and overwritten
$COV run $FLAGS -a $SHARAD/run_surface.py -n --overwrite all
# show all files that would be processed if not done
$COV run $FLAGS -a $SHARAD/run_surface.py -n all
$COV run $FLAGS -a $SHARAD/run_surface.py -o $OUT1 -j 1 e_0224401_007_ss05_700_a

echo "$S0: run_rsr"
# show what happens processing all files
$COV run $FLAGS -a $SHARAD/run_rsr.py -n all > /dev/null
# show all files that would be overwritten
$COV run $FLAGS -a $SHARAD/run_rsr.py -n --overwrite all > /dev/null
# Run an orbit

nice $COV run $FLAGS -a $SHARAD/run_rsr.py --output $OUT1 -s 2000 e_0224401_007_ss05_700_a

if [ "$ENABLE_GDAL" -eq "1" ]
then
    echo "$S0: run_clutter"
    $COV run $FLAGS -a $SHARAD/run_clutter.py --tracklist ./run_ranging__xover_idx.dat -o $OUT1 --maxtracks 2 --jobs 1
    echo "$S0: run_ranging"
    $COV run $FLAGS -a $SHARAD/run_ranging.py --tracklist ./run_ranging__xover_idx.dat -o $OUT1 --maxtracks 4 --jobs 1 -n
fi

if [ "$RUNSLOW" -eq "1" ]
then
    # Ensure cmp data is up-to-date for this track list
    $COV run $FLAGS -a $SHARAD/run_rng_cmp.py --tracklist ./run_ranging__xover_idx.dat -o $OUT1 --maxtracks 2 --jobs 1
    $COV run $FLAGS -a $SHARAD/run_ranging.py --noprogress --tracklist ./run_ranging__xover_idx.dat -o $OUT1 --maxtracks 2 --jobs 1 --qcdir $OUT1/rng/qc
fi


echo "$S0: interferometry"
$COV run $FLAGS -a $MARFA/run_interferometry.py \
                 --project GOG3 --line GOG3/JKB2j/BWN01b/ \
                 --pickfile ../tests/pick_FOI_GOG3_JKB2j_BWN01b.npz \
                 --plot --save out_ri1
$COV run $FLAGS -a $MARFA/run_interferometry.py \
                 --project GOG3 --line GOG3/JKB2j/BWN01b/ --mode Reference \
                 --pickfile ../tests/pick_FOI_GOG3_JKB2j_BWN01b.npz \
                 --refpickfile ../tests/pick_ref_GOG3_JKB2j_BWN01b_Stack15_KMS2.npz \
                 --plot --save out_ri2


####################################
RUN_SAR2_FLAGS="-o $OUT1 -j 1 --maxtracks 1  --tracklist ./tracks_coverage.txt --params run_sar2_cov.json"
# These require cmp to have succeeded
#$COV run $FLAGS -a $SHARAD/run_sar2.py -o ./covdata/run_sar2_data -j 1 --maxtracks 1 --ofmt none --tracklist ./tracks_coverage.txt
echo $S0: run_sar2 -n
$COV run $FLAGS -a $SHARAD/run_sar2.py $RUN_SAR2_FLAGS --ofmt none --focuser ddv2 -n
echo $S0: run_sar2 ddv2
$COV run $FLAGS -a $SHARAD/run_sar2.py $RUN_SAR2_FLAGS --ofmt none --focuser ddv2
# Run ddv2 with no interpolation
$COV run $FLAGS -a $SHARAD/run_sar2.py -j 1 --maxtracks 2 --tracklist ./tracks_coverage.txt  --ofmt hdf5 --focuser ddv2 --params run_sar2_cov2.json

echo $S0: run_sar2 mf
$COV run $FLAGS -a $SHARAD/run_sar2.py -j 1 --maxtracks 3 --tracklist ./tracks_coverage.txt $RUN_SAR2_FLAGS --ofmt none --focuser mf --params run_sar2_mf_cov.json

# This one still takes too long with default params
#echo $S0: run_sar2 ddv1
$COV run $FLAGS -a $SHARAD/run_sar2.py $RUN_SAR2_FLAGS --ofmt none --focuser ddv1

# End SAR
##########################################################


echo "$S0: data_visualization.py"
# These require a connection to an X11 display. allow them to error out
# These scripts require tk to be working properly
if [ "${ENABLE_TK}" -eq "1" ]
then
    $COV run $FLAGS -a  $SHARAD/data_visualization.py --selftest && true
    $COV run $FLAGS -a  $SHARAD/data_visualization.py --product cmp && true
    $COV run $FLAGS -a  $SHARAD/data_visualization.py \
                        --input '$SDS/targ/xtra/SHARAD/foc/mrosh_0001/data/edr10xxx/edr1058901/5m/3 range lines/30km/e_1058901_001_ss19_700_a_s.h5' && true
fi
#---------------------------------------------

echo "$S0: End coverage: " `date` 
echo "$S0: coverage tests completed."
$COV report -m

