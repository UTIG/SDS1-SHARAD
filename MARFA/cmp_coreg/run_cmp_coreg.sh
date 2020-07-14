#!/bin/bash -e
S0=`basename $0`
export DISPLAY=''
#rm -rf coregmethod{0,1,2}
#mkdir -p coregmethod{0,1,2}
METHODS=$*

for METHOD in $METHODS
do
    rm -rf coregmethod${METHOD}_if??
    # Iterate through different interpolation values for method 0
    #for IFACTOR in 10 8 6 4 2 1 15 20 30 40 50  60 70 80 90
    for IFACTOR in 2 4 10 20 40 80
    #for IFACTOR in 90
    do
        echo "$S0: Running coreg method $METHOD, ifactor=$IFACTOR"
        CDIR=`printf coregmethod%d_if%02d ${METHOD} ${IFACTOR}`
        mkdir -p $CDIR
        #../run_interferometry.py --project SRH1 --line DEV2/JKB2t/Y82a \
        ../run_interferometry.py --project GOG3 --line NAQLK/JKB2j/ZY1b/ \
        --plot --save $CDIR --pickfile ../run_interferometry__FOI.npz --coregifactor $IFACTOR --coregmethod $METHOD --mode Roll
    done
done

