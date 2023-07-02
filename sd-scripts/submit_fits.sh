#!/bin/bash

readonly -a min=(600 350 100 560 300 50 450 100 0)
readonly -a max=(1000 1000 1000 1250 1250 1250 1750 1750 1750)
readonly len="${#min[@]}"
readonly exec=${SCRATCH}/ml-conex-xfirst/sd-scripts/make_fits.srm

for i in $(seq 0 $((${len}-1)))
do
  sbatch ${exec} ${min[${i}]} ${max[${i}]}
done
