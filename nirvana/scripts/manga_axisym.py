"""
Script that runs the axisymmetric, least-squares fit for MaNGA data.
"""
import os
import argparse

from IPython import embed

from matplotlib import pyplot

from .. import data
from ..models import axisym

# TODO: Setup a logger
# TODO: Need to test different modes.
#   Tested so far:
#       - Fit Gas or stars
#       - Fit with psf and velocity dispersion
#   Need to test:
#       - Fit without psf
#       - Fit with covariance
#       - Fit without velocity dispersion

#import warnings
#warnings.simplefilter('error', RuntimeWarning)

def parse_args(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('plate', default=None, type=int, 
                        help='MaNGA plate identifier (e.g., 8138)')
    parser.add_argument('ifu', default=None, type=int, 
                        help='MaNGA ifu identifier (e.g., 12704)')
    parser.add_argument('--daptype', default='HYB10-MILESHC-MASTARHC2', type=str,
                        help='DAP analysis key used to select the data files.  This is needed '
                             'regardless of whether or not you specify the directory with the '
                             'data files (using --root).')
    parser.add_argument('--dr', default='MPL-10', type=str,
                        help='The MaNGA data release.  This is only used to automatically '
                             'construct the directory to the MaNGA galaxy data (see also '
                             '--redux and --analysis), and it will be ignored if the root '
                             'directory is set directly (using --root).')
    parser.add_argument('--redux', default=None, type=str,
                        help='Top-level directory with the MaNGA DRP output.  If not defined and '
                             'the direct root to the files is also not defined (see --root), '
                             'this is set by the environmental variable MANGA_SPECTRO_REDUX.')
    parser.add_argument('--analysis', default=None, type=str,
                        help='Top-level directory with the MaNGA DAP output.  If not defined and '
                             'the direct root to the files is also not defined (see --root), '
                             'this is set by the environmental variable MANGA_SPECTRO_ANALYSIS.')
    parser.add_argument('--root', default=None, type=str,
                        help='Path with *all* fits files required for the fit.  This includes ' \
                             'the DRPall file, the DRP LOGCUBE file, and the DAP MAPS file.  ' \
                             'The LOGCUBE file is only required if the beam-smearing is ' \
                             'included in the fit.')
    parser.add_argument('--verbose', default=0, type=int,
                        help='Verbosity level.  0=only status output written to terminal; 1=show '
                             'fit result QA plot; 2=full output.')
    parser.add_argument('--odir', type=str, default=os.getcwd(), help='Directory for output files')
    parser.add_argument('--nodisp', dest='disp', default=True, action='store_false',
                        help='Only fit the velocity field (ignore velocity dispersion)')
    parser.add_argument('--nopsf', dest='smear', default=True, action='store_false',
                        help='Ignore the map PSF (i.e., ignore beam-smearing)')
    parser.add_argument('--covar', default=False, action='store_true',
                        help='Include the nominal covariance in the fit')
    parser.add_argument('--fix_cen', default=False, action='store_true',
                        help='Fix the dynamical center coordinate to the galaxy center')
    parser.add_argument('--fix_inc', default=False, action='store_true',
                        help='Fix the inclination to the guess inclination based on the '
                             'photometric ellipticity')
    parser.add_argument('-t', '--tracer', default='Gas', type=str,
                        help='The tracer to fit; must be either Gas or Stars.')
    parser.add_argument('--rc', default='HyperbolicTangent', type=str,
                        help='Rotation curve parameterization to use: HyperbolicTangent or PolyEx')
    parser.add_argument('--dc', default='Exponential', type=str,
                        help='Dispersion profile parameterization to use: Exponential, ExpBase, '
                             'or Const.')
    parser.add_argument('--min_vel_snr', default=None, type=float,
                        help='Minimum S/N to include for velocity measurements in fit; S/N is '
                             'calculated as the ratio of the surface brightness to its error')
    parser.add_argument('--min_sig_snr', default=None, type=float,
                        help='Minimum S/N to include for dispersion measurements in fit; S/N is '
                             'calculated as the ratio of the surface brightness to its error')
    parser.add_argument('--max_vel_err', default=None, type=float,
                        help='Maximum velocity error to include in fit.')
    parser.add_argument('--max_sig_err', default=None, type=float,
                        help='Maximum velocity dispersion error to include in fit '
                             '(ignored if dispersion not being fit).')
    parser.add_argument('--min_unmasked', default=None, type=int,
                        help='Minimum number of unmasked spaxels required to continue fit.')
    parser.add_argument('--coherent', default=False, action='store_true',
                        help='After the initial rejection of S/N and error limits, find the '
                             'largest coherent region of adjacent spaxels and only fit that '
                             'region.')
    parser.add_argument('--screen', default=False, action='store_true',
                        help='Indicate that the script is being run behind a screen (used to set '
                             'matplotlib backend).') 

    # TODO: Other options:
    #   - Fit with least-squares vs. dynesty
    #   - Type of rotation curve
    #   - Type of dispersion profile
    #   - Include the surface-brightness weighting

    return parser.parse_args() if options is None else parser.parse_args(options)


def main(args):

    # Running the script behind a screen, so switch the matplotlib backend
    if args.screen:
        pyplot.switch_backend('agg')

    #---------------------------------------------------------------------------
    # Setup
    #  - Check the input
    if args.tracer not in ['Gas', 'Stars']:
        raise ValueError('Tracer to fit must be either Gas or Stars.')
    #  - Check that the output directory exists, and if not create it
    if not os.path.isdir(args.odir):
        os.makedirs(args.odir)
    #  - Set the output root name
    oroot = f'nirvana-manga-axisym-{args.plate}-{args.ifu}-{args.tracer}'

    #---------------------------------------------------------------------------
    # Read the data to fit
    if args.tracer == 'Gas':
        kin = data.manga.MaNGAGasKinematics.from_plateifu(args.plate, args.ifu,
                                                          daptype=args.daptype, dr=args.dr,
                                                          redux_path=args.redux,
                                                          cube_path=args.root,
                                                          image_path=args.root,
                                                          analysis_path=args.analysis,
                                                          maps_path=args.root,
                                                          ignore_psf=not args.smear,
                                                          covar=args.covar,
                                                          positive_definite=True)
    elif args.tracer == 'Stars':
        kin = data.manga.MaNGAStellarKinematics.from_plateifu(args.plate, args.ifu,
                                                              daptype=args.daptype, dr=args.dr,
                                                              redux_path=args.redux,
                                                              cube_path=args.root,
                                                              image_path=args.root,
                                                              analysis_path=args.analysis,
                                                              maps_path=args.root,
                                                              ignore_psf=not args.smear,
                                                              covar=args.covar,
                                                              positive_definite=True)
    else:
        # NOTE: Should never get here given the check above.
        raise ValueError(f'Unknown tracer: {args.tracer}')

    # Setup the metadata
    galmeta = data.manga.MaNGAGlobalPar(args.plate, args.ifu, redux_path=args.redux, dr=args.dr,
                                        drpall_path=args.root)
    #---------------------------------------------------------------------------

    # Run the iterative fit
    disk, p0, fix, vel_mask, sig_mask \
            = axisym.axisym_iter_fit(galmeta, kin, rctype=args.rc, dctype=args.dc,
                                     fitdisp=args.disp, max_vel_err=args.max_vel_err,
                                     max_sig_err=args.max_sig_err, min_vel_snr=args.min_vel_snr,
                                     min_sig_snr=args.min_sig_snr, fix_cen=args.fix_cen,
                                     fix_inc=args.fix_inc, min_unmasked=args.min_unmasked,
                                     select_coherent=args.coherent, verbose=args.verbose)

    # Plot the final residuals
    dv_plot = os.path.join(args.odir, f'{oroot}-vdist.png')
    ds_plot = os.path.join(args.odir, f'{oroot}-sdist.png')
    axisym.disk_fit_resid_dist(kin, disk, disp=args.disp, vel_mask=vel_mask, vel_plot=dv_plot,
                               sig_mask=sig_mask, sig_plot=ds_plot)

    # Create the final fit plot
    fit_plot = os.path.join(args.odir, f'{oroot}-fit.png')
    axisym.axisym_fit_plot(galmeta, kin, disk, fix=fix, ofile=fit_plot)

    # Write the output file
    data_file = os.path.join(args.odir, f'{oroot}.fits.gz')
    axisym.axisym_fit_data(galmeta, kin, p0, disk, data_file, vel_mask, sig_mask)


