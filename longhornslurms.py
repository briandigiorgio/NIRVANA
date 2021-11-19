#!/usr/bin/env python3

import numpy as np
import os
import sys
from subprocess import run
from astropy.io import fits
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-t', '--hours', type=int, default = 10, 
                    help='Max number of hours to run')
parser.add_argument('--start', type=int, default=0,
                    help='Index of DRPall file to start at')
parser.add_argument('--stop', type=int, default=-1,
                    help='Index of DRPall file to stop at')
parser.add_argument('--ad', action='store_true',
                    help='Use the plateifus from nirvana_test_sample.txt')
parser.add_argument('--nosmear', action='store_true',
                    help="Don't smear with PSF")
args = parser.parse_args()

if __name__ == '__main__':
    homedir = '/home/08271/bdigiorg/'
    datadir = homedir + 'data/'

    if args.ad:
        plates, ifus = np.genfromtxt(homedir + 'nirvana_testing_sample.txt').T
        drp = {'plate':plates, 'ifudsgn':ifus}
    else:
        drp = fits.open(homedir + 'nirvana/drpall-v3_1_1.fits')[1].data

    lendrp = len(drp['plate'])
    args.stop = lendrp if args.stop == -1 else args.stop

    outdir = datadir + 'fits/'
    remotedir = datadir + 'download/'
    progressdir = datadir + 'progress/'

    plates = np.array(drp['plate'], dtype=int)[args.start:args.stop]
    ifus = np.array(drp['ifudsgn'], dtype=int)[args.start:args.stop]

    for i in range(len(plates)):
        for vtype in ['Gas', 'Stars']:
            if ifus[i] < 4000: partition = 'development'
            else: partition = 'v100'

            fname = homedir + f'/slurms/nirvana_{plates[i]}-{ifus[i]}_{vtype}.slurm'
            progresspath = f'{progressdir}/{plates[i]}/{ifus[i]}/'
            if os.path.isfile(fname):
                os.remove(fname)
            with open(fname, 'a') as f:
                f.write(f'\
#!/bin/bash \n\
#SBATCH --job-name={plates[i]}-{ifus[i]}_{vtype} \n\
#SBATCH --partition={partition} \n\
#SBATCH --mail-type=END,FAIL,REQUEUE \n\
#SBATCH --mail-user=bdigiorg@ucsc.edu \n\
#SBATCH --ntasks=1 \n\
#SBATCH --cpus-per-task=40 \n\
#SBATCH --nodes=1 \n\
#SBATCH --requeue \n \n\
#SBATCH --output={datadir}logs/nirvana_{plates[i]}-{ifus[i]}.log \n\
#SBATCH -t {args.hours}:00:00 \n\
\
pwd; hostname; date \n\
source {homedir}.bashrc \n\n\
\
echo {plates[i]} {ifus[i]} {vtype} \n\
mkdir {progressdir}/{plates[i]}/ \n\
mkdir {progressdir}/{plates[i]}/{ifus[i]}/ \n\
touch {progresspath}/{vtype}.start \n\
nirvana {plates[i]} {ifus[i]} -c 40 {"--stellar" * (vtype == "Stars")} --root {datadir} --dir {outdir} --remote {remotedir} {"--nosmear" * args.nosmear} > {progresspath}/{vtype}.log 2> {progresspath}/{vtype}.err\n\
touch {progressdir}/{plates[i]}/{ifus[i]}/{vtype}.finish \n \n\
date\n\n')
            run(['sbatch',fname])
            time.sleep(.1)
