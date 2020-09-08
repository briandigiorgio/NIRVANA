#!/usr/bin/env python

from barfit import *

if __name__ == '__main__':
	samp = barfit(8078,12704,cores=20, nbins = 5, dyn=False, steps = 20000)
	np.save('em8078-5', samp.chain)
	#pickle.dump(samp.results, open('dyn8979-6','wb'))
