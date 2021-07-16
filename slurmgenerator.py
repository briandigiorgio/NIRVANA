#!/usr/bin/env python3

import numpy as np
import os
import sys
from subprocess import run
from astropy.io import fits
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-n' ,'--nnodes', type=int, default=30,
                    help='Number of nodes to take up')
parser.add_argument('-r', '--reverse', default=False, action='store_true',
                    help='Whether to run galaxies in reverse order')
parser.add_argument('-t', '--hours', type=int, default=12,
                    help='Max number of hours to run. Only applies to reverse')
parser.add_argument('--start', type=int, default=0,
                    help='Index of DRPall file to start at')
parser.add_argument('--stop', type=int, default=-1,
                    help='Index of DRPall file to stop at')
parser.add_argument('--clobber', action='store_true',
                    help='Overwrite existing slurm files')
parser.add_argument('--ad', action='store_true',
                    help='Use the plateifus from nirvana_test_sample.txt')
parser.add_argument('--nosmear', action='store_true',
                    help="Don't smear with PSF")
args = parser.parse_args()

if __name__ == '__main__':
    if args.ad:
        plates, ifus = np.genfromtxt('/home/bdigiorg/nirvana_testing_sample.txt').T
        drp = {'plate':plates, 'ifudsgn':ifus}
    else:
        drp = fits.open('/home/bdigiorg/drpall-v3_1_1.fits')[1].data

    lendrp = len(drp['plate'])
    args.stop = lendrp if args.stop == -1 else args.stop
    galpernode = (args.stop - args.start)//args.nnodes
    print(galpernode, 'galaxies per file')

    rootdir = '/data/users/bdigiorg/'
    outdir = '/data/users/bdigiorg/fits/'
    remotedir = '/data/users/bdigiorg/download/'
    progressdir = '/data/users/bdigiorg/progress/'

    plates = np.array(drp['plate'], dtype=int)[args.start:args.stop]
    ifus = np.array(drp['ifudsgn'], dtype=int)[args.start:args.stop]
    if args.reverse:
        plates = plates[::-1]
        ifus = ifus[::-1]
        timelimit = f'#SBATCH --time={args.hours}:00:00'
    else: timelimit = ''

    for i in range(args.nnodes):
        platesi = plates[galpernode * i:galpernode * (i+1)]
        ifusi = ifus[galpernode * i:galpernode * (i+1)]
        fname = f'/home/bdigiorg/slurms/nirvana_{platesi[0]}-{platesi[-1]}.slurm'
        if os.path.isfile(fname):
            if args.clobber:
                os.remove(fname)
            else:
                raise FileExistsError(f'File already exists: {fname}')
        with open(fname, 'a') as f:
            f.write(f'\
#!/bin/bash \n\
#SBATCH --job-name={args.reverse * "r"}{platesi[0]}-{platesi[-1]}_nirvana \n\
#SBATCH --partition=windfall \n\
#SBATCH --account=windfall \n\
#SBATCH --mail-type=END,FAIL,{"REQUEUE" * (1-args.reverse)} \n\
#SBATCH --mail-user=bdigiorg@ucsc.edu \n\
#SBATCH --ntasks=1 \n\
#SBATCH --cpus-per-task=40 \n\
#SBATCH --nodes=1 \n\
#SBATCH --requeue \n \n\
#SBATCH --output=/data/users/bdigiorg/logs/nirvana_{args.reverse * "r"}{platesi[0]}-{platesi[-1]}.log \n\
{timelimit} \n\
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
                progresspath = f'{progressdir}/{platesi[j]}/{ifusi[j]}/'
                f.write(f'\
echo {platesi[j]} {ifusi[j]} gas \n\
mkdir {progressdir}/{platesi[j]}/ \n\
mkdir {progressdir}/{platesi[j]}/{ifusi[j]}/ \n\
touch {progresspath}/gas.start \n\
nirvana {platesi[j]} {ifusi[j]} -c 40 --root {rootdir} --dir {outdir} --remote {remotedir} {"--nosmear" * args.nosmear} > {progresspath}/gas.log 2> {progresspath}/gas.err\n\
touch {progressdir}/{platesi[j]}/{ifusi[j]}/gas.finish \n \n\
ln -s {outdir}/nirvana_{platesi[j]}-{ifusi[j]}_Gas.fits {progresspath}/gas.fits\n\
\
echo {platesi[j]} {ifusi[j]} stellar \n\
touch {progresspath}/stellar.start \n\
nirvana {platesi[j]} {ifusi[j]} -s -c 40 --root {rootdir} --dir {outdir} --remote {remotedir} {"--nosmear" * args.nosmear} > {progresspath}/stellar.log 2> {progresspath}/stellar.err\n\
touch {progresspath}/stellar.finish \n\
ln -s {outdir}/nirvana_{platesi[j]}-{ifusi[j]}_Stars.fits {progresspath}/stellar.fits\n\
date\n\n')
        run(['sbatch',fname])
        time.sleep(.1)
