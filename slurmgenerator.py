#!/usr/bin/env python3

import numpy as np
import os
import sys
from subprocess import run
from astropy.io import fits
import time

if __name__ == '__main__':
    nnodes = 20
    start = 7000
    stop = start+1000
    galpernode = (stop-start)//nnodes
    print(galpernode, 'galaxies per file')

    drp = fits.open('/home/bdigiorg/drpall-v3_1_1.fits')[1].data
    plates = drp['plate'][start:]
    ifus = np.array(drp['ifudsgn'], dtype=int)[start:]

    for i in range(nnodes):
        platesi = plates[galpernode * i:galpernode * (i+1)]
        ifusi = ifus[galpernode * i:galpernode * (i+1)]
        fname = f'/home/bdigiorg/slurms/nirvana_{platesi[0]}-{platesi[-1]}.slurm'
        if os.path.isfile(fname):
            raise FileExistsError(f'File already exists: {fname}')
        with open(fname, 'a') as f:
            f.write(f'\
#!/bin/bash \n\
#SBATCH --job-name={platesi[0]}-{platesi[-1]}_nirvana \n\
#SBATCH --partition=windfall \n\
#SBATCH --account=windfall \n\
#SBATCH --mail-type=END,FAIL,REQUEUE \n\
#SBATCH --mail-user=bdigiorg@ucsc.edu \n\
#SBATCH --ntasks=1 \n\
#SBATCH --cpus-per-task=40 \n\
#SBATCH --nodes=1 \n\
#SBATCH --output=/data/users/bdigiorg/logs/nirvana_{platesi[0]}-{platesi[-1]}.log \n\
#SBATCH --requeue \n \n\
\
pwd; hostname; date \n\n\
\
module load python/3.9.0 \n\
module load nirvana/0.0.1dev \n\
module load fftw/3.3.8 \n\n\
\
export SAS_BASE_DIR=/data/users/bdigiorg/\n\
export MANGA_SPECTRO=/data/users/bdigiorg/mangawork/manga/spectro\n\
export MANGA_SPECTRO_REDUX=$MANGA_SPECTRO/redux/\n\
export MANGA_SPECTRO_ANALYSIS=$MANGA_SPECTRO/analysis/\n\n')


            for j in range(len(platesi)):
                f.write(f'\
echo {platesi[j]} {ifusi[j]} gas \n\
mkdir /data/users/bdigiorg/progress/{platesi[j]}/ \n\
mkdir /data/users/bdigiorg/progress/{platesi[j]}/{ifusi[j]}/ \n\
touch /data/users/bdigiorg/progress/{platesi[j]}/{ifusi[j]}/gas.start \n\
nirvana {platesi[j]} {ifusi[j]} -c 40 --root /data/users/bdigiorg/ --dir /data/users/bdigiorg/fits/ --remote /data/users/bdigiorg/download/ --fits \n\
touch /data/users/bdigiorg/progress/{platesi[j]}/{ifusi[j]}/gas.finish \n \n\
\
echo {platesi[j]} {ifusi[j]} stellar \n\
touch /data/users/bdigiorg/progress/{platesi[j]}/{ifusi[j]}/stellar.start \n\
        nirvana {platesi[j]} {ifusi[j]} -c 40 --root /data/users/bdigiorg/ --dir /data/users/bdigiorg/fits/ --remote /data/users/bdigiorg/download/ --fits -s \n\
touch /data/users/bdigiorg/progress/{platesi[j]}/{ifusi[j]}/stellar.finish \n\
date\n\n')
        run(['sbatch',fname])
        time.sleep(.1)
