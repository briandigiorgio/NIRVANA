#!/usr/bin/env python

import os, glob, datetime
import numpy as np

verbose = False

nprocs = int(os.popen('squeue -u bdigiorg | wc -l').read()) - 1
outf = open(f'progcheck_{datetime.datetime.now().isoformat(timespec="minutes")}.txt', 'w')
outf.write(f"Nirvana progress as of {datetime.datetime.now().isoformat(timespec='minutes')} \n{nprocs} processes currently running\n\n")

fs = sorted(glob.glob('slurms/*'))
tot = 0
for fi in fs:
    plates = []
    ifus = []
    done = []
    times = []
    progress = ''
    try:
        with open(fi, 'r') as f:
            lines = f.readlines()
    except Exception: continue
    for l in lines:
        if 'echo' in l:
            split = l.split(' ')
            plates += [split[1]]
            ifus += [split[2]]
            typ = 'Gas' if split[3] == 'gas' else 'Stars'
            typ2 = 'gas' if split[3] == 'gas' else 'stellar'
            fname = f'data/fits/nirvana_{split[1]}-{split[2]}_{typ}.fits'

            if len(glob.glob(f'data/progress/{split[1]}/{split[2]}/{typ2}*finish')):
                if os.path.isfile(fname):
                    progress += '#'
                    done += [True]
                    times += [os.path.getmtime(fname)]
                else:
                    progress += 'X'
                    done += [False]
                    times += [None]
            else:
                progress += '.'
                done += [False]
                times += [None]
    success = np.sum(done)
    last = np.where([p != '.' for p in progress])[-1][-1] if 'X' in progress or '#' in progress else 0
    lastgood = np.where(done)[-1][-1] if any(done) else 0
    lasttime = datetime.datetime.fromtimestamp(times[lastgood]) if lastgood else None
    tot += last + 1

    if last + 1 != len(plates) or verbose:
        if verbose and last+1 == len(plates): last -= 1
        outf.write(f'\n{fi}:\n Finished: {last+1}/{len(plates)}, Successful: {success}/{len(plates)}, Last saved: {lasttime}, Current: {plates[last+1]}-{ifus[last+1]}\n')
        outf.write(progress)
        outf.write('\n\n')
    #for i in range(len(plates)):
    #    print(plates[i], ifus[i], done[i])

nfits = int(os.popen('ls /home/bdigiorg/data/fits/*fits | wc -l').read())
outf.write(f"\n\n\nTotal fits attempted: {tot}, Percent done: {tot*100/22546:.1f}%, Total successful fits: {nfits}, Success rate: {nfits*100/tot:.1f}%\n")
outf.close()
