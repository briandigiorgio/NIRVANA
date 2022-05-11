#!/usr/bin/env python

import os, glob, datetime
import numpy as np

if __name__ == '__main__':
    verbose = False
    user = 'bdigiorg'
    ntot = 1593 #22546

    #create file, find number of processes currently running for user
    nprocs = int(os.popen(f'squeue -u {user} | wc -l').read()) - 1
    outf = open(f'progcheck_{datetime.datetime.now().isoformat(timespec="minutes")}.txt', 'w')
    outf.write(f"Nirvana progress as of {datetime.datetime.now().isoformat(timespec='minutes')} \n{nprocs} processes currently running\n\n")

    #iterate through slurm files to check progress
    fs = sorted(glob.glob('slurms/*'))
    tot = 0
    for fi in fs:
        plates = []
        ifus = []
        done = []
        times = []
        progress = ''

        #read all of the lines of the slurm file
        try:
            with open(fi, 'r') as f:
                lines = f.readlines()
        except Exception: continue

        #iterate through lines to assemble list of galaxies
        for l in lines:
            if 'echo' in l:
                #reconstruct filename from info in line
                split = l.split(' ')
                plates += [split[1]]
                ifus += [split[2]]
                typ = 'Gas' if split[3] == 'gas' else 'Stars'
                typ2 = 'gas' if split[3] == 'gas' else 'stellar'
                fname = f'data/fits/nirvana_{split[1]}-{split[2]}_{typ}.fits'

                #fill out progress bar with status of galaxy
                if len(glob.glob(f'data/progress/{split[1]}/{split[2]}/{typ2}*finish')):
                    # '#' symbol for galaxy with successful outfile
                    if os.path.isfile(fname):
                        progress += '#'
                        done += [True]
                        times += [os.path.getmtime(fname)]

                    # 'X' for galaxy that is has finish but has no outfile
                    else:
                        progress += 'X'
                        done += [False]
                        times += [None]

                # '.' for galaxies that haven't started
                else:
                    progress += '.'
                    done += [False]
                    times += [None]

        #count successful galaxies, find last run and last successful and total
        success = np.sum(done)
        last = np.where([p != '.' for p in progress])[-1][-1] if 'X' in progress or '#' in progress else 0
        lastgood = np.where(done)[-1][-1] if any(done) else 0
        tot += last + 1

        #find finish time of last successful galaxy
        lasttime = datetime.datetime.fromtimestamp(times[lastgood]) if lastgood else None

        #for active slurms, print progress
        if last + 1 != len(plates) or verbose:
            if verbose and last+1 == len(plates): last -= 1
            outf.write(f'\n{fi}:\n Finished: {last+1}/{len(plates)}, Successful: {success}/{len(plates)}, Last saved: {lasttime}, Current: {plates[last+1]}-{ifus[last+1]}\n')
            outf.write(progress)
            outf.write('\n\n')

    #print total progress
    nfits = int(os.popen('ls /home/bdigiorg/data/fits/*fits | wc -l').read())
    outf.write(f"\n\n\nTotal fits attempted: {tot}, Percent done: {tot*100/ntot:.1f}%, Total successful fits: {nfits}, Success rate: {nfits*100/tot:.1f}%\n")
    outf.close()
