"""
Provides a class to house and manipulate galaxy global parameters/metadata.

.. include common links, assuming primary doc root is up one directory
.. include:: ../include/links.rst
"""

from IPython import embed

import numpy as np

import astropy.units
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

from ..models.geometry import projected_polar
from .util import select_major_axis

class GlobalPar:
    r"""
    A class used to house and manipulate global galaxy properties.

    Any of the parameters can be None, but this may limit the usability of
    some class methods.

    Args:
        ra (:obj:`float`, optional):
            RA of the galaxy center
        dec (:obj:`float`, optional):
            DEC of the galaxy center
        mass (:obj:`float`, optional):
            Stellar mass of the galaxy in solar masses.
        z (:obj:`float`, optional):
            Bulk redshift of the galaxy.
        pa (:obj:`float`, optional):
            Characteristic galaxy photometric position angle in degrees.
        ell (:obj:`float`, optional):
            Characteristic galaxy isophotal ellipticity.
        reff (:obj:`float`, optional):
            Effective (half-light) radius of the galaxy in arcseconds.
        sersic_n (:obj:`float`, optional):
            Sersic index, :math:`n`, of the galaxy light profile.
        q0 (:obj:`float`, optional):
            Intrinsic oblateness of the galaxy.

    Raises:
        ValueError:
            Raised if the ellipticity or intrinsic oblateness is not in the
            range [0,1].
    """
    def __init__(self, ra=None, dec=None, mass=None, z=None, pa=None, ell=None, reff=None,
                 sersic_n=None, q0=0.2):

        # Check input
        if ell is not None and (ell > 1  or ell < 0):
            raise ValueError('Ellipticity must be 0.0 <= ell <= 1.0!')
        if q0 is not None and (q0 > 1 or q0 < 0):
            raise ValueError('Intrinsic oblateness must be 0.0 <= q0 <= 1.0!')

        # Assign values to attributes
        self.ra = ra
        self.dec = dec
        self.mass = mass
        self.z = z
        self.ell = ell
        self.reff = reff
        self.sersic_n = sersic_n
        self.q0 = q0

        self.pa = pa
        if self.pa is not None and (self.pa < -180. or self.pa >= 180.):
            warnings.warn('Converting position angle to be in the range -180 <= pa < 180.')
            while self.pa >= 180:
                self.pa -= 360.
            while self.pa < -180.:
                self.pa += 360.

    def guess_inclination(self, lb=0., ub=90.):
        r"""
        Return a guess inclination using the equation:

        .. math::
        
            \cos^2 i = \frac{ q^2 - q_0^2 }{ 1 - q_0^2 },

        where :math:`q` is the observed oblateness (the semi-minor to
        semi-major axis ratio, :math:`q = 1-\epsilon = b/a`) and
        :math:`q_0` is the intrinsic oblateness.
        
        If :math:`q < q_0`, the function returns a 90 degree inclination. If
        :attr:`q0` is None, the returned inclination is for an infinitely
        thin disk (:math:`q_0 = 0`).

        Args:
            lb (:obj:`float`, optional):
                Lower bound for the returned inclination.
            ub (:obj:`float`, optional):
                Upper bound for the returned inclination.

        Returns:
            :obj:`float`: Inclination estimate in degrees.

        Raises:
            ValueError:
                Raised if :math:`q_0 == 1.` such that the inclination is
                undefined, or if :attr:`ell` is None.
        """
        if self.ell is None:
            raise ValueError('Ellipticity is undefined.')
        if self.q0 is None or self.q0 == 0.0:
            return np.clip(np.degrees(np.arccos(1.0 - self.ell)), lb, ub)

        q = 1.0 - self.ell
        return np.clip(90. if q < self.q0 
                       else np.degrees(np.arccos(np.sqrt((q**2-self.q0**2) / (1.0 - self.q0**2)))),
                       lb, ub)

    def guess_kinematic_pa(self, x, y, v, r_range=None, wedge=30., return_vproj=False):
        r"""
        Use the input coordinates and velocity field to estimate the
        kinematic position angle.

        The estimate uses the existing ellipticity and position angle
        (presumably from the photometry) to deproject the line-of-sight
        velocities into the projected rotation curve using data within the
        specified radial range (``r_range``) and angular wedge around the
        major axis (``wedge``). The deprojected rotation velocities are then
        used to determine if the the kinematic position angle (specifically
        defined to be along the receding side of the rotation curve) should
        be flipped with respect to the photometric one. If the median
        rotation velocity is negative, the returned position angle is 180 deg
        flipped with respect to :attr:`pa`. This method does *not* estimate
        the PA directly from the velocity field alone.

        Args:
            x (`numpy.ndarray`_):
                On-sky x coordinates relative to the dynamical center of the
                galaxy.  East is toward +x, West is toward -x.
            y (`numpy.ndarray`_):
                On-sky y coordinates relative to the dynamical center of the
                galaxy.  North is toward +y, South is toward -y.
            v (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
                Line-of-sight velocity measurements at ``x`` and ``y``.
            r_range (array-like, optional):
                The lower and upper limit of the radial range over which to
                measure the median rotation velocity. If None, the radial
                range is from 1/5 to 2/3 of the radial range within the
                selected wedge around the major axis.
            wedge (:obj:`float`, optional):
                The :math:`\pm` wedge in degrees around the major axis used
                in the velocity deprojection.
            return_vproj (:obj:`bool`, optional):
                Return the estimate of the projected rotational velocity
                along the major axis, as well as the position angle.

        Returns:
            :obj:`float`, :obj:`tuple`: Provides the position angle estimate
            in degrees, as well as an estimate of the projected rotation
            velocity, if requested. The range of the returned PA is -180 to
            180 degrees.

        Raises:
            ValueError:
                Raised if :attr:`pa` or :attr:`ell` are None, or if the input
                arrays have mismatched shapes.
        """
        # Check input
        if self.pa is None or self.ell is None:
            raise ValueError('Position angle and/or ellipticity are undefined.')
        if x.shape != y.shape or x.shape != v.shape:
            raise ValueError('Input coordinate and velocity arrays must all have the same shape.')
        if r_range is not None and len(r_range) != 2:
                raise ValueError('Specified radial range must contain 2 and only 2 elements.')

        # Get the disk-plane polar coordinates
        r, th = projected_polar(x, y, np.radians(self.pa), np.radians(self.guess_inclination()))
        # Create a mask that selects data near the major axis
        gpm = select_major_axis(r, th, r_range=r_range, wedge=wedge)
        # Include any input mask
        gpm |= np.logical_not(np.ma.getmaskarray(v))

        # Get the median projected rotation speed
        vp = np.ma.median((v[gpm] - np.ma.median(v[gpm])) / np.cos(th[gpm]))

        # If the projected rotation is negative, flip the photometric PA by 180
        # degrees
        _pa = self.pa if vp > 0 else (self.pa - 180. if self.pa > 0 else self.pa + 180.)

        # Return the position angle and the projected rotation speed, if requested
        return (_pa, abs(vp)) if return_vproj else _pa

    def guess_central_dispersion(self, x, y, s, r_lim=1.25):
        r"""
        Use the input coordinates and velocity dispersion field to estimate
        the central velocity dispersion.

        For the moment this does a simple calculation of the on-sky radius
        and returns the mean dispersion of the pixels with the provided
        limiting radius.

        Args:
            x (`numpy.ndarray`_):
                On-sky x coordinates relative to the dynamical center of the
                galaxy.  East is toward +x, West is toward -x.
            y (`numpy.ndarray`_):
                On-sky y coordinates relative to the dynamical center of the
                galaxy.  North is toward +y, South is toward -y.
            s (`numpy.ndarray`_, `numpy.ma.MaskedArray`_):
                Line-of-sight velocity dispersion at ``x`` and ``y``.
            r_lim (:obj:`float`, optional):
                The upper limit for the *on-sky* radius used in the estimate.
                Units should match the ``x`` and ``y`` inputs.

        Returns:
            :obj:`float`: Provides the mean velocity dispersion within the
            provided aperture.

        Raises:
            ValueError:
                Raised if the input arrays have mismatched shapes.
        """
        # Check input
        if x.shape != y.shape or x.shape != s.shape:
            raise ValueError('Input arrays must all have the same shape.')
        return np.mean(s[x**2 + y**2 < r_lim**2])

    def kpc_per_arcsec(self):
        r"""
        Return the factor needed to convert the on-sky separation to a
        physical separation.

        The redshift must be available (:attr:`z`) for this calculation. If
        it is not available, the method issues a warning and returns -1.

        The calculation assumes a flat :math:`\Lambda{\rm CDM}` cosmology
        with :math:`\Omega_m = 0.3` (:math:`\Omega_\Lambda = 0.7`). The
        Hubble constant is assumed to be 100 km/s/Mpc such that the returned
        conversion is in units of :math:`h^{-1} {\rm kpc / arcsec}`.

        Returns:
            :obj:`float`: :math:`h^{-1} {\rm kpc / arcsec}` conversion factor
            at the galaxy redshift; always returns -1 if the redshift is not
            available.
        """
        if self.z is None or self.z <= 0:
            warnings.warn('Redshift not provided or less than 0.  Cannot determine kpc/arcsec.')
            return -1.
        H0 = 100 * astropy.units.km / astropy.units.s / astropy.units.Mpc
        cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)
        return np.radians(1/3600) * 1e3 * cosmo.angular_diameter_distance(self.z).value


