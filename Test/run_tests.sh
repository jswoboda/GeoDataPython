#!/bin/bash
# Michael Hirsch
# runs all tests in this folder, in order of complexity from simplest to most
# complicated

flist=(omti_test.py subplots_test.py)

for f in ${flist[*]}; do
python2 $f
python3 $f
done


