#!/bin/bash
# Michael Hirsch
# runs all tests in this folder, in order of complexity from simplest to most
# complicated

flist=(omti_test.py subplots_test.py altitudeSlicev2.py plottingtest3d.py)

for f in ${flist[*]}; do
echo -e "\n *** python2 test $f ***\n"
python2 $f
echo -e "\n *** python3 test $f ***\n"
python3 $f
done


