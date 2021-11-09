import numpy as np

from .beam import smear, ConvolveFFTW
from ..data.util import unpack
from .geometry import projected_polar

import matplotlib.pyplot as plt

def bisym_model(args, paramdict, plot=False, relative_pab=False):
    '''
    Evaluate a bisymmetric velocity field model for given parameters.

    The model for this is a second order nonaxisymmetric model taken from
    Leung (2018) who in turn took it from Spekkens & Sellwood (2007). It
    evaluates the specified models at the desired coordinates.

    Args:
        args (:class:`~nirvana.data.fitargs.FitArgs`):
            Object containing all of the data and settings needed for the
            galaxy.  
        paramdict (:obj:`dict`): 
            Dictionary of galaxy parameters that are being fit. Assumes the
            format produced :func:`nirvana.fitting.unpack`.
        plot (:obj:`bool`, optional): 
            Flag to return resulting models as 2D arrays instead of 1D for 
            plotting purposes.
        relative_pab (:obj:`bool`, optional):
            Whether to define the second order position angle relative to the
            first order position angle (better for fitting) or absolutely
            (better for output).

    Returns:
        :obj:`tuple`: Tuple of two objects that are the model velocity field and
        the model velocity dispersion (if `args.disp = True`, otherwise second
        object is `None`). Arrays are 1D unless specified otherwise and should
        be rebinned to match the data.

    '''

    #convert angles to polar and normalize radial coorinate
    inc, pa, pab = np.radians([paramdict['inc'], paramdict['pa'], paramdict['pab']])
    pab = (pab - pa) % (2*np.pi)
    r, th = projected_polar(args.kin.grid_x-paramdict['xc'], args.kin.grid_y-paramdict['yc'], pa, inc)

    #interpolate the velocity arrays over full coordinates
    if len(args.edges) != len(paramdict['vt']):
        raise ValueError(f"Bin edge and velocity arrays are not the same shape: {len(args.edges)} and {len(paramdict['vt'])}")
    vtvals  = np.interp(r, args.edges, paramdict['vt'])
    v2tvals = np.interp(r, args.edges, paramdict['v2t'])
    v2rvals = np.interp(r, args.edges, paramdict['v2r'])

    #spekkens and sellwood 2nd order vf model (from andrew's thesis)
    velmodel = paramdict['vsys'] + np.sin(inc) * (vtvals * np.cos(th) \
             - v2tvals * np.cos(2 * (th - pab)) * np.cos(th) \
             - v2rvals * np.sin(2 * (th - pab)) * np.sin(th))


    #define dispersion and surface brightness if desired
    if args.disp: 
        sigmodel = np.interp(r, args.edges, paramdict['sig'])
        #sb = args.kin.remap('sb', masked=False)
    else: 
        sigmodel = None
        #sb = None
    debug=False
    if debug:
        plt.figure(figsize=(8,8))
        plt.subplot(221)
        plt.imshow(velmodel, cmap='jet', origin='lower', vmin=-200, vmax=200)
        #plt.subplot(222)
        #plt.imshow(sigmodel, cmap='jet', origin='lower')
        #plt.subplot(223)
        #plt.imshow(args.kin.remap('sb'), cmap='jet', origin='lower')
        #plt.subplot(224)
        #plt.imshow(args.kin.remap('sb')==0, cmap='jet', origin='lower')
        #return


    #apply beam smearing if beam is given
    conv = ConvolveFFTW(args.kin.spatial_shape)
    #try: conv
    #except: conv = None

    if args.kin.beam_fft is not None:
        if hasattr(args, 'smearing') and not args.smearing: pass
        else: 
            sbmodel, velmodel, sigmodel = smear(velmodel, args.kin.beam_fft, sb=args.kin.remap('sb'), 
            sig=sigmodel, beam_fft=True, cnvfftw=conv, verbose=True)
        if debug:
            plt.subplot(222)
            plt.imshow(velmodel, cmap='jet', origin='lower', vmin=-200, vmax=200)

    #remasking after convolution
    if args.kin.vel_mask is not None: velmodel = np.ma.array(velmodel, mask=args.kin.remap('vel_mask'))
    if args.kin.sig_mask is not None: sigmodel = np.ma.array(sigmodel, mask=args.kin.remap('sig_mask'))
    if debug:
        plt.subplot(223)
        plt.imshow(velmodel, cmap='jet', origin='lower', vmin=-200, vmax=200)

    #rebin data
    binvel = np.ma.MaskedArray(args.kin.bin(velmodel), mask=args.kin.vel_mask)
    if sigmodel is not None: binsig = np.ma.MaskedArray(args.kin.bin(sigmodel), mask=args.kin.sig_mask)
    else: binsig = None

    #return a 2D array for plotting reasons
    if plot:
        velremap = args.kin.remap(binvel, masked=True)
        if sigmodel is not None: 
            sigremap = args.kin.remap(binsig, masked=True)
            if debug:
                plt.subplot(224)
                plt.imshow(velremap, cmap='jet', origin='lower', vmin=-200, vmax=200)
            return velremap, sigremap
        return velremap

    return binvel, binsig
