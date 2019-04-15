#!/bin/bash

# Show the results of a python3 profiling run.
# Usage Example:
# Collect python profiling data to a file myrundata.profile
#
# $ python3 -m cProfile -o myrundata.profile ./run_myscript.py option1 option2 ...
#
# Then use this script to show the profiling results sorted by cumulative time
#
# $ ./show_profile.sh myrundata.profile
#
# Or show the results, sorted by total time
#
# $ ./show_profile.sh myrundata.profile tottime

PFILE=$1

if [ "$#" -gt "1" ]
then
    TYPE=$2
else
    TYPE=cumtime
fi

python3 -c "import pstats ; pstats.Stats('$PFILE').strip_dirs().sort_stats('$TYPE').print_stats(20)"
